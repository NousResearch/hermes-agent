"""Reviewer gate + profile/skill validation for Kanban task creation and routing.

What these tests pin down (all of it new behaviour; every test here is red on
the base commit ``655010d6``):

* ``kanban_create``'s optional ``reviewer`` persists ``tasks.required_reviewer``
  and records it on the ``created`` event; omitting it changes nothing.
* Validation happens BEFORE every side effect: a bad reviewer profile, a
  missing forced skill, or a mixed valid/missing skill list leaves no task row,
  no dependency edge, no event and no workspace behind.
* Nonexistent profiles are reported as ``profile '<name>' was not found`` with
  the roster, ``kanban_discover`` guidance and ``Nothing changed`` — never as
  "not installed".
* A reviewer must carry the skill the dispatcher force-loads for review-phase
  startup (``sdlc-review``); an existing profile without it is rejected.
* ``request_review`` routes to the SAVED reviewer and refuses any override;
  the gate survives the request_changes -> implementer cycle.
* ``complete_task`` refuses an implementation run (or an unclaimed/spoofed
  caller) on a gated card, allows the trusted review run, and records an audit
  event when an operator explicitly overrides.
* ``kanban_complete`` is absent from the schema of a gated implementation run,
  while the ungated / review-phase schemas keep it.
* Reassignment preserves forced skills and the reviewer gate.
* Migration + readback expose ``required_reviewer``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_validation as kv
from hermes_cli import profiles as profiles_mod


PROFILES = ("default", "qa", "reviewer", "worker")


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME + empty board, with the profile roster pinned.

    The roster is stubbed (as the other review-lifecycle tests do) so profile
    existence is a decision of this test, not of whatever happens to be on the
    host. The skill library is deliberately left to the real resolver: a bare
    hermetic home falls back to the checkout's bundled tree, exactly like a
    profile ``seed_profile_skills`` has populated.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name in PROFILES)
    monkeypatch.setattr(profiles_mod, "list_profile_names", lambda: list(PROFILES))
    return home


def _seed_skill(home: Path, name: str) -> Path:
    """Give ``home`` a skill tree containing exactly ``name`` (and nothing else).

    A non-empty tree makes skill resolution strict for that home, so these
    tests exercise the real verdict rather than the bundled fallback.
    """
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test double\n---\n\nbody\n",
        encoding="utf-8",
    )
    return home / "skills"


def _counts(conn) -> dict:
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("tasks", "task_links", "task_events", "task_runs")
    }


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    return [e for e in out if e[0] == kind] if kind else out


def _frozen(conn, tid: str) -> dict:
    """Everything a refused reassignment must leave byte-for-byte alone: the
    whole task row (status, claim, ``current_run_id``, assignee, failure
    streak), the run table and the complete ordered event log."""
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    return {
        "task": tuple(row) if row is not None else None,
        "events": [tuple(r) for r in conn.execute(
            "SELECT kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        )],
        "runs": [tuple(r) for r in conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        )],
    }


# ---------------------------------------------------------------------------
# Creation: gate persisted, validated before every side effect
# ---------------------------------------------------------------------------


def test_create_with_reviewer_persists_gate_and_event(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        task = kb.get_task(conn, tid)
        assert task.required_reviewer == "reviewer"
        created = _events(conn, tid, "created")[0][1]
        assert created["required_reviewer"] == "reviewer"
        # Readback surfaces the gate on every task view.
        from hermes_cli.kanban_output import _task_to_dict

        assert _task_to_dict(task)["required_reviewer"] == "reviewer"


def test_create_without_reviewer_is_ungated(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain work", assignee="worker")
        assert kb.get_task(conn, tid).required_reviewer is None


def test_unknown_reviewer_profile_is_zero_writes(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.ProfileNotFoundError) as excinfo:
            kb.create_task(
                conn, title="gated work", assignee="worker", reviewer="ghost",
            )
        message = str(excinfo.value)
        assert "profile 'ghost' was not found" in message
        assert "not installed" not in message
        assert "Available profiles:" in message
        assert "kanban_discover" in message
        assert message.rstrip().endswith("Nothing changed.")
        assert _counts(conn) == before, "a refused create wrote to the board"


def test_reviewer_without_review_skill_is_zero_writes(board: Path) -> None:
    # A non-empty skill tree that simply lacks the dispatcher's review skill:
    # the profile exists, but cannot run the review phase.
    _seed_skill(board, "some-other-skill")
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.create_task(
                conn, title="gated work", assignee="worker", reviewer="reviewer",
            )
        message = str(excinfo.value)
        assert "reviewer" in message and "sdlc-review" in message
        assert "not found for profile" in message
        assert _counts(conn) == before


def test_valid_reviewer_with_bundled_review_skill_is_accepted(board: Path) -> None:
    # Bare home -> bundled fallback; the checkout ships sdlc-review, which is
    # what seed_profile_skills installs into every real profile.
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        assert kb.get_task(conn, tid).required_reviewer == "reviewer"


def test_absent_forced_skill_is_zero_writes(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.create_task(
                conn, title="specialist work", assignee="worker",
                skills=["definitely-not-a-skill"],
            )
        message = str(excinfo.value)
        assert "worker" in message and "definitely-not-a-skill" in message
        assert _counts(conn) == before
        assert not any((board / "workspaces").glob("*")), "a refused create made a workspace"


def test_mixed_valid_and_missing_skills_reject_atomically(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.create_task(
                conn, title="mixed work", assignee="worker",
                skills=["sdlc-review", "definitely-not-a-skill"],
            )
        message = str(excinfo.value)
        # The whole list is judged at once: the profile and ONLY the missing
        # names are reported, and nothing is written.
        assert "worker" in message
        assert "definitely-not-a-skill" in message
        assert "sdlc-review" not in message.split("Nothing changed.")[0].split(": ", 1)[-1]
        assert _counts(conn) == before


def test_external_shared_skill_is_accepted(board: Path) -> None:
    """A skill living in a shared/external dir (``skills.external_dirs``) is
    part of the profile's effective library."""
    shared = board.parent / "shared-skills"
    skill_dir = shared / "team-playbook"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: team-playbook\ndescription: shared\n---\n\nbody\n",
        encoding="utf-8",
    )
    # The skill tree is read from the ASSIGNEE's own home, so give that profile
    # a home whose config points at the shared tree (an absolute entry, as
    # ``skills.external_dirs`` documents).
    profile_dir = board / "profiles" / "worker"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        "skills:\n  external_dirs:\n    - {}\n".format(shared), encoding="utf-8",
    )
    _seed_skill(board, "sdlc-review")  # non-empty -> strict mode
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="shared work", assignee="worker",
            skills=["team-playbook"],
        )
        assert kb.get_task(conn, tid).skills == ["team-playbook"]


# ---------------------------------------------------------------------------
# request_review: saved routing, override rejection, capability
# ---------------------------------------------------------------------------


def test_request_review_routes_to_the_saved_reviewer(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", expected_run_id=run.current_run_id,
            with_reason=True,
        )
        assert ok is True, why
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.required_reviewer == "reviewer"
        payload = _events(conn, tid, "review_requested")[0][1]
        assert payload["reviewer"] == "reviewer"
        assert payload["implementer"] == "worker"


def test_request_review_forbids_a_reviewer_override(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        before = _events(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", reviewer="qa",
            expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is False
        assert "override is not allowed" in why
        assert "reviewer" in why and "qa" in why
        task = kb.get_task(conn, tid)
        assert (task.status, task.assignee) == ("running", "worker")
        assert _events(conn, tid) == before


def test_request_review_rejects_a_reviewer_without_the_review_skill(board: Path) -> None:
    _seed_skill(board, "some-other-skill")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ungated work", assignee="worker")
        run = kb.claim_task(conn, tid)
        before = _events(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", reviewer="qa",
            expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is False
        assert "sdlc-review" in why and "qa" in why
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid) == before


def test_review_changes_cycle_retains_the_gate(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        impl_run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="first pass", expected_run_id=impl_run.current_run_id,
        )
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None
        assert review_run.assignee == "reviewer"
        ok, implementer = kb.request_changes(conn, tid, reason="needs tests")
        assert ok is True and implementer == "worker"
        task = kb.get_task(conn, tid)
        # Back with the implementer, gate intact and reviewer unchanged.
        assert task.status in ("ready", "todo")
        assert task.required_reviewer == "reviewer"
        assert task.assignee == "worker"
        # Second handoff still routes to the saved reviewer, still no override.
        ok, why = kb.request_review(conn, tid, summary="second pass", with_reason=True)
        assert ok is True, why
        assert kb.get_task(conn, tid).assignee == "reviewer"


# ---------------------------------------------------------------------------
# complete_task: backend gate (authoritative lifecycle/run state)
# ---------------------------------------------------------------------------


def test_implementation_completion_is_refused(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        with pytest.raises(kb.ReviewerGateError) as excinfo:
            kb.complete_task(
                conn, tid, summary="all done",
                expected_run_id=run.current_run_id,
            )
        message = str(excinfo.value)
        assert "required reviewer" in message and "reviewer" in message
        assert "Nothing changed." in message
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.completed_at is None


def test_unclaimed_and_forced_completion_are_still_refused(board: Path) -> None:
    """Neither an unclaimed caller nor ``force=True`` gets past the gate:
    ``force`` only governs a live claim, and the phase is read from the board."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="done without claiming")
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="forced", force=True)
        assert kb.get_task(conn, tid).status in ("ready", "todo", "running")


def test_trusted_same_profile_review_run_can_complete(board: Path) -> None:
    """reviewer == implementer: the PHASE, not the profile string, decides."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="self-reviewed work", assignee="worker", reviewer="worker",
        )
        impl_run = kb.claim_task(conn, tid)
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="done", expected_run_id=impl_run.current_run_id)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=impl_run.current_run_id,
        )
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None and review_run.assignee == "worker"
        assert kb.complete_task(
            conn, tid, summary="approved by reviewer",
            expected_run_id=review_run.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "done"


def test_wrong_profile_claiming_review_is_refused(board: Path) -> None:
    """A live review run that is NOT the saved reviewer's cannot close the card."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None
        # Forge the identity the way a spoofing caller would: the run still
        # belongs to the card, but not to the saved reviewer.
        conn.execute(
            "UPDATE task_runs SET profile = 'qa' WHERE id = ?", (review_run.current_run_id,),
        )
        conn.commit()
        with pytest.raises(kb.ReviewerGateError) as excinfo:
            kb.complete_task(
                conn, tid, summary="approved",
                expected_run_id=review_run.current_run_id,
            )
        assert "not the saved reviewer's review run" in str(excinfo.value)


def test_explicit_override_completes_and_is_audited(board: Path) -> None:
    """The audited operator-recovery contract under the revised trust boundary.

    The backend boolean IS the deliberate manual override: CLI
    ``--override-reviewer`` and the dashboard's ``review_gate_override`` are
    operator surfaces, and upstream grants those surfaces no caller-role
    auth (same as ``--force`` / ``created_by``) — so a caller reaching the
    backend with the flag is acting as an operator by definition. What must
    NOT exist is the flag in an AGENT tool payload (schema test +
    payload-smuggling test below) and every consumption is recorded as
    exactly ONE ``reviewer_gate_overridden`` event.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        assert kb.complete_task(
            conn, tid, summary="operator override",
            expected_run_id=run.current_run_id, review_gate_override=True,
        )
        assert kb.get_task(conn, tid).status == "done"
        assert len(_events(conn, tid, "reviewer_gate_overridden")) == 1


def test_ungated_completion_is_unchanged(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain work", assignee="worker")
        run = kb.claim_task(conn, tid)
        assert kb.complete_task(
            conn, tid, summary="done", expected_run_id=run.current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert _events(conn, tid, "reviewer_gate_overridden") == []


# ---------------------------------------------------------------------------
# Reassignment preserves skills and the gate
# ---------------------------------------------------------------------------


def test_reassign_refuses_a_profile_without_the_forced_skill(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="specialist work", assignee="worker",
            skills=["sdlc-review"],
        )
        kb.assign_task(conn, tid, "reviewer")
        _seed_skill(board, "some-other-skill")  # strict mode from here on
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.assign_task(conn, tid, "qa")
        assert "qa" in str(excinfo.value) and "sdlc-review" in str(excinfo.value)
        assert kb.get_task(conn, tid).assignee == "reviewer"


def test_reassign_cannot_move_a_review_phase_card_off_its_reviewer(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        with pytest.raises(ValueError) as excinfo:
            kb.assign_task(conn, tid, "qa")
        assert "required reviewer" in str(excinfo.value)
        assert kb.get_task(conn, tid).assignee == "reviewer"


# ---------------------------------------------------------------------------
# reclaim_first validates BEFORE it reclaims (refusal is a pure read)
# ---------------------------------------------------------------------------

def test_reassign_with_reclaim_refuses_a_missing_skill_without_dropping_the_claim(
    board: Path,
) -> None:
    """``reclaim_first=True`` used to reclaim FIRST and validate SECOND.

    ``assign_task``'s refusal contract is "a refusal changes nothing", and the
    CLI's own comment leans on it — but the reclaim that preceded the
    validation had already released the claim, flipped ``running`` back to
    ``ready``, closed the run and appended a ``reclaimed`` event. A typo'd or
    under-skilled destination therefore cost a live worker its run while the
    error message promised nothing had happened.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="specialist work", assignee="worker", skills=["sdlc-review"],
        )
        _seed_skill(board, "some-other-skill")  # strict mode from here on
        assert kb.claim_task(conn, tid) is not None
        before = _frozen(conn, tid)

        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.reassign_task(conn, tid, "qa", reclaim_first=True)
        assert "qa" in str(excinfo.value) and "sdlc-review" in str(excinfo.value)

        assert _frozen(conn, tid) == before, "a refused reclaim-reassign mutated the card"
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.claim_lock is not None
        assert task.assignee == "worker"


def test_reassign_with_reclaim_refuses_moving_a_live_review_run_off_its_reviewer(
    board: Path,
) -> None:
    """Same defect through the reviewer gate, and the harder direction.

    A card whose CURRENT status is ``running`` does not look review-gated at
    all — the gate only trips on ``status == "review"``. Releasing the claim is
    exactly what returns the card to ``review`` (``_retry_status_for_run``), so
    the refusal surfaced only after the reclaim had already mutated the card.
    Validating against the status the card WILL hold closes that gap.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated work", assignee="worker", reviewer="reviewer")
        run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        assert kb.claim_review_task(conn, tid) is not None  # live reviewer run
        before = _frozen(conn, tid)

        with pytest.raises(ValueError) as excinfo:
            kb.reassign_task(conn, tid, "qa", reclaim_first=True)
        assert "required reviewer" in str(excinfo.value)

        assert _frozen(conn, tid) == before, "a refused reclaim-reassign mutated the card"
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.claim_lock is not None
        assert task.assignee == "reviewer"


def test_reassign_with_reclaim_still_recovers_a_valid_destination(board: Path) -> None:
    """The escape hatch keeps working: a destination that honours the forced
    skill still reclaims the claim and takes the card over."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="specialist work", assignee="worker", skills=["sdlc-review"],
        )
        _seed_skill(board, "sdlc-review")  # destination can honour the skill
        assert kb.claim_task(conn, tid) is not None
        events_before = len(_frozen(conn, tid)["events"])

        assert kb.reassign_task(conn, tid, "qa", reclaim_first=True) is True
        task = kb.get_task(conn, tid)
        assert task.assignee == "qa"
        assert task.status == "ready"
        assert task.claim_lock is None
        kinds = [e[0] for e in _frozen(conn, tid)["events"]]
        assert kinds[events_before:] == ["reclaimed", "assigned"]


def test_reassign_with_reclaim_keeps_a_live_review_run_on_its_own_reviewer(
    board: Path,
) -> None:
    """Positive control for the gate path above: pointing a live review run at
    its OWN saved reviewer is allowed, reclaim and all."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated work", assignee="worker", reviewer="reviewer")
        run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        assert kb.claim_review_task(conn, tid) is not None
        events_before = len(_frozen(conn, tid)["events"])

        assert kb.reassign_task(conn, tid, "reviewer", reclaim_first=True) is True
        task = kb.get_task(conn, tid)
        assert task.assignee == "reviewer"
        assert task.status == "review"
        assert task.claim_lock is None
        kinds = [e[0] for e in _frozen(conn, tid)["events"]]
        assert kinds[events_before:] == ["reclaimed", "assigned"]


# ---------------------------------------------------------------------------
# Tool schema / dispatcher startup context
# ---------------------------------------------------------------------------


def test_kanban_complete_is_hidden_from_a_gated_implementation_run(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt
    from tools.registry import invalidate_check_fn_cache, registry

    def offered() -> set:
        return {
            d["function"]["name"]
            for d in registry.get_definitions({"kanban_complete"}, quiet=True)
        }

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    monkeypatch.setenv("HERMES_KANBAN_REQUIRED_REVIEWER", "reviewer")

    monkeypatch.setenv("HERMES_KANBAN_RUN_PHASE", "implementation")
    invalidate_check_fn_cache()
    assert kt._check_kanban_complete_mode() is False
    assert offered() == set(), "gated implementation run still received kanban_complete"

    monkeypatch.setenv("HERMES_KANBAN_RUN_PHASE", "review")
    invalidate_check_fn_cache()
    assert kt._check_kanban_complete_mode() is True
    assert offered() == {"kanban_complete"}

    # Ungated: neither var set, the tool is offered exactly as before.
    monkeypatch.delenv("HERMES_KANBAN_REQUIRED_REVIEWER")
    monkeypatch.delenv("HERMES_KANBAN_RUN_PHASE")
    invalidate_check_fn_cache()
    assert kt._check_kanban_complete_mode() is True
    assert offered() == {"kanban_complete"}


def test_dispatcher_review_gate_env_contract(board: Path) -> None:
    gated = kb.Task(
        id="t_deadbeef", title="t", body=None, assignee="worker", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None,
        claim_expires=None, tenant=None, required_reviewer="reviewer",
    )
    assert kbd.review_gate_env(gated) == {
        "HERMES_KANBAN_REQUIRED_REVIEWER": "reviewer",
        "HERMES_KANBAN_RUN_PHASE": "implementation",
    }
    gated.run_phase = "review"
    assert kbd.review_gate_env(gated)["HERMES_KANBAN_RUN_PHASE"] == "review"
    ungated = kb.Task(
        id="t_cafebabe", title="t", body=None, assignee="worker", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None,
        claim_expires=None, tenant=None,
    )
    assert kbd.review_gate_env(ungated) == {}


def test_worker_context_states_the_gate_once(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
            body="spec goes here",
        )
        context = kb.build_worker_context(conn, tid)
        assert context.count("Required reviewer: reviewer") == 1
        assert "kanban_request_review" in context
        # The gate is context, not body: the description itself is untouched.
        assert kb.get_task(conn, tid).body == "spec goes here"


# ---------------------------------------------------------------------------
# Migration / readback
# ---------------------------------------------------------------------------


def test_migration_adds_required_reviewer_to_a_legacy_board(board: Path) -> None:
    import sqlite3

    db_path = board / "kanban.db"
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "required_reviewer" in cols
        # Simulate a pre-gate board: drop the column, then reopen.
        conn.execute("ALTER TABLE tasks DROP COLUMN required_reviewer")
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES ('t_legacy', 'legacy card', 'ready', 0, 'scratch')"
        )
        conn.commit()
    finally:
        conn.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(db_path)
    with kbc.connect() as conn:
        task = kb.get_task(conn, "t_legacy")
        assert task is not None
        assert task.required_reviewer is None
        # New rows on the migrated board carry the gate again.
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        assert kb.get_task(conn, tid).required_reviewer == "reviewer"


# ---------------------------------------------------------------------------
# Agent-facing surface: profile wording + zero writes before the board opens
# ---------------------------------------------------------------------------


def test_tool_create_rejects_unknown_profile_with_shared_wording(board: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        before = _counts(conn)
    out = json.loads(kt._handle_create({"title": "typo", "assignee": "nope"}))
    assert out.get("ok") is not True
    message = out["error"]
    assert "profile 'nope' was not found" in message
    assert "not installed" not in message
    assert "Available profiles:" in message
    assert "kanban_discover" in message
    assert message.endswith("Nothing changed.")
    with kbc.connect() as conn:
        assert _counts(conn) == before


def test_tool_create_rejects_missing_skill_before_the_board_opens(board: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        before = _counts(conn)
    out = json.loads(kt._handle_create({
        "title": "specialist", "assignee": "worker",
        "skills": ["definitely-not-a-skill"],
    }))
    assert out.get("ok") is not True
    assert "worker" in out["error"] and "definitely-not-a-skill" in out["error"]
    with kbc.connect() as conn:
        assert _counts(conn) == before


def test_create_schema_declares_the_reviewer_gate() -> None:
    from tools.registry import registry

    properties = registry.get_schema("kanban_create")["parameters"]["properties"]
    assert "reviewer" in properties
    # The gate must not hand an agent the backend's recovery lever.
    complete = registry.get_schema("kanban_complete")["parameters"]["properties"]
    assert "review_gate_override" not in complete
    assert "force" not in complete


def test_delegate_child_cannot_close_a_gated_card(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegated permission protection still outranks the gate: a delegate
    child inherits HERMES_KANBAN_* but is never a run owner."""
    from agent import delegation_context
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        assert kb.claim_task(conn, tid) is not None
        before = _events(conn, tid)

    monkeypatch.setattr(delegation_context, "is_delegated_child_process_context", lambda: True)
    out = json.loads(kt._handle_complete({"task_id": tid, "summary": "done"}))
    assert out.get("ok") is not True
    assert "delegate_task child agents are not Kanban run owners" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid) == before


# ---------------------------------------------------------------------------
# Alternate routes: CLI / dashboard API / agent tool (override trust boundary)
# ---------------------------------------------------------------------------
#
# The accepted boundary (upstream trust model audit): the explicit reviewer
# override is an INTENTIONAL manual recovery on operator surfaces — CLI flag
# and dashboard field — deliberately absent from every agent tool payload,
# always audited, and never implied by ``--force``. Ordinary gated completion
# must be refused from authoritative phase state on every route; what "any
# caller with the flag" means is "an operator", because upstream grants those
# surfaces no caller-role auth (same as ``created_by`` / ``--force``).

def _parse_kanban(argv: list) -> argparse.Namespace:
    """Drive the real ``hermes kanban …`` parser (as run_slash/CLI do)."""
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers())
    return root.parse_args(["kanban", *argv])


def _live_gated_card(conn) -> tuple[str, int]:
    """A gated card claimed by a live implementation run; returns (tid, run_id)."""
    tid = kb.create_task(
        conn, title="gated work", assignee="worker", reviewer="reviewer",
    )
    run = kb.claim_task(conn, tid)
    assert run is not None and run.current_run_id is not None
    return tid, int(run.current_run_id)


def test_cli_complete_is_refused_even_with_force_and_run_ownership(
    board: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    """``hermes kanban complete`` on a live gated implementation run is refused
    from authoritative phase state: neither ``--force`` (live-claim guard)
    nor owning the live run (HERMES_KANBAN_TASK/RUN_ID) implies the reviewer
    override — and no audit event is written by a refusal."""
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)

    with kbc.connect() as conn:
        tid, run_id = _live_gated_card(conn)
        before = _counts(conn)

    # Plain operator-style completion.
    capsys.readouterr()
    rc = kc._cmd_complete(_parse_kanban(["complete", tid, "--summary", "done"]))
    assert rc == 1
    err = capsys.readouterr().err
    assert "required reviewer" in err and "Nothing changed." in err

    # --force governs the live-claim guard only; the gate fires first.
    rc = kc._cmd_complete(
        _parse_kanban(["complete", tid, "--summary", "done", "--force"]))
    assert rc == 1
    assert "required reviewer" in capsys.readouterr().err

    # Run ownership (the worker's own env) does not re-label the phase either.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    rc = kc._cmd_complete(
        _parse_kanban(["complete", tid, "--summary", "done", "--force"]))
    assert rc == 1
    assert "required reviewer" in capsys.readouterr().err

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.completed_at is None
        assert _counts(conn) == before, "a refused completion wrote to the board"
        assert _events(conn, tid, "reviewer_gate_overridden") == []


def test_cli_override_reviewer_completes_and_audits_once(
    board: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    """The explicit CLI flag is the audited operator recovery: it completes
    the live gated card and records EXACTLY ONE audit event."""
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)

    with kbc.connect() as conn:
        tid, _ = _live_gated_card(conn)

    capsys.readouterr()
    rc = kc._cmd_complete(
        _parse_kanban(["complete", tid, "--summary", "operator recovery",
                       "--override-reviewer"]))
    assert rc == 0
    assert f"Completed {tid}" in capsys.readouterr().out

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"
        assert len(_events(conn, tid, "reviewer_gate_overridden")) == 1


def test_agent_complete_payload_cannot_smuggle_an_override(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No override in the agent tool payload — two layers deep:

    1. the schema carries no force/override field (schema test above), and
       the registry rejects smuggled keys as unknown parameters before the
       handler even runs;
    2. the clean payload falls through to the ordinary gated refusal with
       zero writes — ``_handle_complete`` never forwards a force/override
       flag to the backend.
    """
    from tools import kanban_tools as kt

    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)

    with kbc.connect() as conn:
        tid, _ = _live_gated_card(conn)
        before = _counts(conn)

    out = json.loads(kt._handle_complete({
        "task_id": tid, "summary": "done",
        "force": True, "review_gate_override": True,
    }))
    assert out.get("ok") is not True
    assert "unknown parameter(s): force, review_gate_override" in out["error"]

    out = json.loads(kt._handle_complete({"task_id": tid, "summary": "done"}))
    assert out.get("ok") is not True
    assert "required reviewer" in out["error"]

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert _counts(conn) == before
        assert _events(conn, tid, "reviewer_gate_overridden") == []


# ---------------------------------------------------------------------------
# Dashboard API routes (same gate, force hardcoded, explicit flag only)
# ---------------------------------------------------------------------------

@pytest.fixture
def client(board: Path):
    """Dashboard plugin router mounted on a bare FastAPI app over the same
    isolated board (mirrors tests/plugins/test_kanban_dashboard_plugin.py)."""
    import importlib.util
    import sys as _sys

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_gate_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def test_dashboard_done_is_refused_until_the_explicit_flag_audits_once(
    board: Path, client,
) -> None:
    """The dashboard hardcodes ``force=True`` for ``done`` — the review gate
    still fires first: PATCH/bulk without ``review_gate_override`` are refused
    with nothing written; the explicit flag completes and audits exactly once."""
    with kbc.connect() as conn:
        tid, _ = _live_gated_card(conn)
        before = _counts(conn)

    # PATCH done — force is implicit on this route, the override is not.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "done", "result": "done", "summary": "done"},
    )
    assert r.status_code == 400, r.text
    assert "required reviewer" in r.json()["detail"]
    # Bulk route: same gate, refusal quoted per id.
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [tid], "status": "done", "result": "done", "summary": "done"},
    )
    assert r.status_code == 200, r.text
    entry = r.json()["results"][0]
    assert entry["ok"] is False and "required reviewer" in entry["error"]

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert _counts(conn) == before, "a refused dashboard completion wrote to the board"
        assert _events(conn, tid, "reviewer_gate_overridden") == []

    # The explicit operator flag completes and audits exactly once.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "done", "result": "done", "summary": "operator recovery",
              "review_gate_override": True},
    )
    assert r.status_code == 200, r.text
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"
        assert len(_events(conn, tid, "reviewer_gate_overridden")) == 1


def test_api_create_validation_matches_the_backend(
    board: Path, client,
) -> None:
    """Backend/CLI/API validation consistency: a bad reviewer or a missing
    forced skill is a 400 with the shared wording and ZERO writes."""
    with kbc.connect() as conn:
        before = _counts(conn)

    r = client.post("/api/plugins/kanban/tasks", json={
        "title": "x", "assignee": "worker", "reviewer": "ghost"})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "profile 'ghost' was not found" in detail
    assert "not installed" not in detail
    assert detail.rstrip().endswith("Nothing changed.")

    r = client.post("/api/plugins/kanban/tasks", json={
        "title": "x", "assignee": "worker", "skills": ["definitely-not-a-skill"]})
    assert r.status_code == 400, r.text
    assert "worker" in r.json()["detail"] and "definitely-not-a-skill" in r.json()["detail"]

    with kbc.connect() as conn:
        assert _counts(conn) == before, "a refused API create wrote to the board"


def test_cli_create_validation_matches_the_backend(
    board: Path, capsys,
) -> None:
    """Backend/CLI validation consistency (the API twin is above): refusals
    share the wording and write nothing — while an UNKNOWN ASSIGNEE on the CLI
    remains the deliberate pre-existing exclusion (the dispatcher buckets it
    as skipped_nonspawnable; profile-existence validation lives on the agent
    tool surface and the reviewer path)."""
    with kbc.connect() as conn:
        before = _counts(conn)

    rc = kc._cmd_create(_parse_kanban(
        ["create", "T", "--assignee", "worker", "--reviewer", "ghost"]))
    assert rc == 2
    err = capsys.readouterr().err
    assert "profile 'ghost' was not found" in err
    assert "not installed" not in err
    assert err.rstrip().endswith("Nothing changed.")

    rc = kc._cmd_create(_parse_kanban(
        ["create", "T", "--assignee", "worker", "--skill", "definitely-not-a-skill"]))
    assert rc == 2
    err = capsys.readouterr().err
    assert "worker" in err and "definitely-not-a-skill" in err

    with kbc.connect() as conn:
        assert _counts(conn) == before, "a refused CLI create wrote to the board"

    # Deliberate exclusion, pinned so it cannot drift silently.
    rc = kc._cmd_create(_parse_kanban(["create", "T", "--assignee", "ghost"]))
    assert rc == 0
    capsys.readouterr()
    with kbc.connect() as conn:
        created = [t for t in kb.list_tasks(conn) if t.title == "T"]
        assert created and created[-1].assignee == "ghost"


# ---------------------------------------------------------------------------
# Effective-skill resolver: the bundle fallback must match RUNTIME seeding
# ---------------------------------------------------------------------------

def _write_skill(root: Path, name: str) -> Path:
    """A minimal SKILL.md at ``root/<name>`` (frontmatter name = dir name)."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test double\n---\n\nbody\n",
        encoding="utf-8",
    )
    return skill_dir


def _live_profile(board: Path, name: str) -> Path:
    """A resolvable named profile home (identity marker, unseeded tree)."""
    home = board / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    return home


def test_bundle_fallback_honors_no_bundled_skills_marker(board: Path) -> None:
    """A profile that opted out of bundled skills (.no-bundled-skills) gets
    ONLY ESSENTIAL_SKILLS seeded at startup — so its empty library must not
    resolve to the whole checkout bundle: sdlc-review is runtime-absent and
    must be refused, while the essential skill itself stays available."""
    home = _live_profile(board, "optout")
    (home / ".no-bundled-skills").write_text("", encoding="utf-8")

    with pytest.raises(kv.MissingSkillsError) as excinfo:
        kv.require_skills("optout", ["sdlc-review"])
    message = str(excinfo.value)
    assert "optout" in message and "sdlc-review" in message

    # ESSENTIAL_SKILLS ("hermes-agent") is seeded even under the opt-out.
    kv.require_skills("optout", ["hermes-agent"])


def test_bundle_fallback_uses_the_env_bundled_source(
    board: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packaged installs seed from HERMES_BUNDLED_SKILLS, not from this
    checkout: an empty/absent env bundle seeds NOTHING (a checkout-only skill
    must be refused), and a skill the env bundle carries is what resolves."""
    env_bundle = tmp_path / "env-bundle"
    env_bundle.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(env_bundle))
    _live_profile(board, "packaged")

    # Empty source → nothing would be seeded → even a checkout skill refused.
    with pytest.raises(kv.MissingSkillsError):
        kv.require_skills("packaged", ["sdlc-review"])

    _write_skill(env_bundle, "env-source-skill")
    kv.require_skills("packaged", ["env-source-skill"])
    # The checkout is not consulted while the env source rules.
    with pytest.raises(kv.MissingSkillsError):
        kv.require_skills("packaged", ["sdlc-review"])


def test_bundle_fallback_skips_curator_suppressed_skills(board: Path) -> None:
    """Curator-pruned built-ins are never (re)seeded by sync_skills — the
    fallback must not resurrect them from the bundle either."""
    home = _live_profile(board, "pruned")
    (home / "skills").mkdir()
    (home / "skills" / ".curator_suppressed").write_text(
        "# pruned by the curator\nsdlc-review\n", encoding="utf-8")

    with pytest.raises(kv.MissingSkillsError):
        kv.require_skills("pruned", ["sdlc-review"])
    # A non-suppressed bundle skill still seeds.
    kv.require_skills("pruned", ["hermes-agent"])


def test_resolver_scopes_roots_to_the_target_profile(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolvable profile is judged ONLY against what ITS worker scans:
    its own tree (+ create_dir/external). The shared root library and the
    validating process's home are other profiles' trees — counting them
    would accept skills the spawned worker cannot load."""
    target = _live_profile(board, "targetp")
    _write_skill(target / "skills", "target-only-skill")
    # Foreign evidence in both tempting places:
    _write_skill(board / "skills", "root-only-skill")          # shared root library
    validator = _live_profile(board, "validator")
    _write_skill(validator / "skills", "validator-only-skill")  # validating home

    monkeypatch.setenv("HERMES_HOME", str(validator))
    names = {n.rsplit("/", 1)[-1] for n in kv.profile_skill_names("targetp")}
    assert "target-only-skill" in names
    assert "root-only-skill" not in names, "shared root library leaked into a named profile's library"
    assert "validator-only-skill" not in names, "validating home leaked into another profile's library"

    kv.require_skills("targetp", ["target-only-skill"])
    with pytest.raises(kv.MissingSkillsError):
        kv.require_skills("targetp", ["root-only-skill"])

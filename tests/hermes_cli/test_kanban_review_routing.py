"""Reviewer-separation routing: a review handoff never returns to its implementer.

``request_review`` is the single choke point every surface funnels through
(worker tool, CLI, dashboard PATCH). When the caller omits ``reviewer=`` the
old code fell back to the latest ``changes_requested`` provenance, which is
``None`` on a first review -- so the card stayed on its implementer and the
dispatcher spawned the implementer as the reviewer. These tests pin the
resolution order (explicit > provenance > policy), the self-review refusal,
the durable ``reviewer_source``/``reviewer`` records, and the dispatcher guard
that reroutes legacy cards already parked in ``review``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import profiles


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def installed_profiles(monkeypatch: pytest.MonkeyPatch):
    """Policy candidates resolve as installed (the temp home has no profiles)."""
    names = {"qa", "architect", "reviewer", "devmanager"}
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in names)
    return names


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    return [e for e in out if kind is None or e[0] == kind]


def _handoff(conn, tid):
    return _events(conn, tid, kind="review_requested")[-1][1]


# ---------------------------------------------------------------------------
# Pure resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementer", ["coder", "cloudops", "devstaff1", "dev-x"])
def test_policy_maps_implementer_patterns_to_qa(implementer: str) -> None:
    cfg = kb.REVIEW_ROUTING_DEFAULTS
    assert kb.resolve_default_reviewer(implementer, cfg, lambda n: True) == ("qa", "policy")


@pytest.mark.parametrize("implementer", ["qa", "security", "designer"])
def test_policy_defaults_other_implementers_to_architect(implementer: str) -> None:
    cfg = kb.REVIEW_ROUTING_DEFAULTS
    assert kb.resolve_default_reviewer(implementer, cfg, lambda n: True) == ("architect", "policy")


def test_policy_skips_the_implementer_even_when_it_is_the_mapping() -> None:
    # implementer == architect maps to architect (itself): separation must fall
    # through the fallback chain instead of returning the card to its author.
    assert kb.resolve_default_reviewer("architect", kb.REVIEW_ROUTING_DEFAULTS, lambda n: True) == (
        "qa", "policy",
    )


def test_policy_walks_fallbacks_in_order_and_skips_duplicates() -> None:
    cfg = dict(kb.REVIEW_ROUTING_DEFAULTS, implementer_patterns=["coder", "cloudops"],
               implementer_reviewer="qa", default_reviewer="architect",
               fallback_reviewers=["qa", "architect", "reviewer"])
    # qa not installed, architect installed -> architect wins despite being second.
    installed = {"architect"}
    assert kb.resolve_default_reviewer("cloudops", cfg, lambda n: n in installed) == (
        "architect", "policy",
    )
    # qa installed -> the mapped candidate wins over the earlier duplicate.
    installed = {"qa", "architect"}
    assert kb.resolve_default_reviewer("cloudops", cfg, lambda n: n in installed) == (
        "qa", "policy",
    )


def test_policy_fails_closed_when_no_candidate_profile_exists() -> None:
    assert kb.resolve_default_reviewer(
        "cloudops", kb.REVIEW_ROUTING_DEFAULTS, lambda n: False,
    ) == (None, None)


def test_policy_returns_none_for_an_unknown_implementer_with_no_fallback() -> None:
    cfg = dict(kb.REVIEW_ROUTING_DEFAULTS, default_reviewer="architect",
               fallback_reviewers=["architect"])
    assert kb.resolve_default_reviewer("coder", cfg, lambda n: False) == (None, None)


# ---------------------------------------------------------------------------
# Integration: request_review
# ---------------------------------------------------------------------------


def test_first_review_without_reviewer_reassigns_off_the_implementer(
    kanban_home: Path, installed_profiles,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl", assignee="cloudops")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        assert kb.request_review(
            conn, tid, summary="done", expected_run_id=claimed.current_run_id,
        ) is True

        assert kb.get_task(conn, tid).status == "review"
        assert kb.get_task(conn, tid).assignee == "qa"
        payload = _handoff(conn, tid)
        assert payload["implementer"] == "cloudops"
        assert payload["reviewer"] == "qa"
        assert payload["reviewer_source"] == "policy"


def test_re_review_keeps_the_provenance_reviewer(kanban_home: Path, installed_profiles) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cycle", assignee="cloudops")
        impl_run = kb.claim_task(conn, tid)
        assert impl_run is not None
        assert kb.request_review(conn, tid, summary="v1", expected_run_id=impl_run.current_run_id)

        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review.current_run_id,
        ) == (True, "cloudops")

        retry = kb.claim_task(conn, tid)
        assert retry is not None
        assert kb.request_review(conn, tid, summary="v2", expected_run_id=retry.current_run_id)

        payload = _handoff(conn, tid)
        assert payload["implementer"] == "cloudops"
        assert payload["reviewer"] == "qa"
        assert payload["reviewer_source"] == "provenance"
        assert kb.get_task(conn, tid).assignee == "qa"


def test_provenance_naming_the_implementer_is_ignored(kanban_home: Path, installed_profiles) -> None:
    """The observed t_2fafba53 loop: a changes_requested whose reviewer was the
    implementer must NOT route the next round back to the implementer."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="heal the loop", assignee="cloudops")
        impl_run = kb.claim_task(conn, tid)
        assert impl_run is not None
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="qa", expected_run_id=impl_run.current_run_id,
        )

        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="fix", expected_run_id=review.current_run_id,
        ) == (True, "cloudops")
        # Corrupt the provenance into the observed self-review shape.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET payload = '{\"reviewer\": \"cloudops\"}' "
                "WHERE task_id = ? AND kind = 'changes_requested'",
                (tid,),
            )

        retry = kb.claim_task(conn, tid)
        assert retry is not None
        assert kb.request_review(conn, tid, summary="v2", expected_run_id=retry.current_run_id)

        payload = _handoff(conn, tid)
        assert payload["reviewer"] == "qa"
        assert payload["reviewer_source"] == "policy"
        assert kb.get_task(conn, tid).assignee == "qa"


def test_explicit_self_review_is_refused_and_leaves_the_card_running(
    kanban_home: Path, installed_profiles,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no self review", assignee="cloudops")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        ok, reason = kb.request_review(
            conn, tid, summary="done", reviewer="cloudops",
            expected_run_id=claimed.current_run_id, with_reason=True,
        )
        assert ok is False
        assert reason is not None and "self-review" in reason
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).assignee == "cloudops"
        assert _events(conn, tid, kind="review_requested") == []


def test_allow_self_review_escape_hatch(kanban_home: Path, installed_profiles, monkeypatch) -> None:
    monkeypatch.setattr(
        kb, "review_routing_config",
        lambda: {**kb.REVIEW_ROUTING_DEFAULTS, "allow_self_review": True},
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="opt in", assignee="cloudops")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="done", reviewer="cloudops",
            expected_run_id=claimed.current_run_id,
        ) is True
        payload = _handoff(conn, tid)
        assert payload["reviewer"] == "cloudops"
        assert payload["reviewer_source"] == "explicit"


def test_handoff_fails_closed_when_no_reviewer_profile_exists(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="nobody to review", assignee="cloudops")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok, reason = kb.request_review(
            conn, tid, summary="done", expected_run_id=claimed.current_run_id,
            with_reason=True,
        )
        assert ok is False
        assert reason is not None and "no eligible reviewer" in reason
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid, kind="review_requested") == []


def test_completed_event_and_run_metadata_record_the_reviewer(
    kanban_home: Path, installed_profiles,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="approve me", assignee="cloudops")
        impl_run = kb.claim_task(conn, tid)
        assert impl_run is not None
        assert kb.request_review(conn, tid, summary="ready", expected_run_id=impl_run.current_run_id)

        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.complete_task(
            conn, tid, summary="LGTM", expected_run_id=review.current_run_id,
        ) is True

        run = kb.latest_run(conn, tid)
        assert run.outcome == "completed"
        assert run.metadata is not None and run.metadata["reviewer"] == "qa"
        completed = _events(conn, tid, kind="completed")[-1][1]
        assert completed["reviewer"] == "qa"


def test_worker_context_surfaces_the_reviewer(kanban_home: Path, installed_profiles) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="show me", assignee="cloudops")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(conn, tid, summary="done", expected_run_id=claimed.current_run_id)
        assert "Reviewer: qa" in kb.build_worker_context(conn, tid)


# ---------------------------------------------------------------------------
# Dispatcher guard: legacy card parked on its implementer
# ---------------------------------------------------------------------------


def test_dispatcher_reroutes_legacy_review_card_once(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda n: n in {"qa", "architect"})
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="legacy", assignee="cloudops")
        impl_run = kb.claim_task(conn, tid)
        assert impl_run is not None
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="qa", expected_run_id=impl_run.current_run_id,
        )
        # Simulate the pre-fix state: the card sits in review on its implementer.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review', assignee = 'cloudops', "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ?", (tid,),
            )

        spawned: list[tuple[str, str]] = []

        def _spawn(task, workspace, board):
            spawned.append((task.id, task.assignee or ""))

        result = kbd.DispatchResult()
        row = conn.execute("SELECT id, assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert kbd._dispatch_lane_task(
            conn, row, row["assignee"], result, lane="review", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=2, spawn_fn=_spawn,
            per_profile_cap=None, per_profile_running={},
        ) is True

        assert kb.get_task(conn, tid).assignee == "qa"
        assert spawned == [(tid, "qa")]
        rerouted = _events(conn, tid, kind="review_rerouted")
        assert len(rerouted) == 1
        assert rerouted[0][1] == {"from": "cloudops", "to": "qa", "implementer": "cloudops"}

        # The guard's own condition is now false: no second reroute event.
        assert kb.reroute_legacy_review(conn, tid, profile_exists=lambda n: True) is None
        assert len(_events(conn, tid, kind="review_rerouted")) == 1

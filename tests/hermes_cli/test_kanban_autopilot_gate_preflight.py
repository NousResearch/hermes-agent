"""P2b — autopilot preflight human-gate hardening (review-5 leftovers).

The autopilot skill script (``kanban_autopilot.py``) is the READ-ONLY
inspection surface for human gates. Two review-5 findings harden it:

1. **Comment-trail read errors fail closed** — when the comment trail cannot
   be read, ``_latest_pending_human_gate`` returns ``(None, err)`` and the
   old preflight silently passed (it only rejected when ``gate_id`` was not
   ``None``). Now a read error fails the ``human_gate_pending`` check with
   the error in the reason.
2. **Multiple armed gates fail closed naming ALL of them** — the old parser
   kept a single ``armed`` variable, so a second armed gate replaced the
   first and an approval for the last one released an earlier still-pending
   obligation. ``_latest_pending_human_gate`` now returns the FULL LIST of
   armed gate ids without later matching approvals, and the preflight check
   fails with every armed gate id in the detail when the list is non-empty.
3. **Preflight re-verifies the approval artifact on disk** — an approval
   marker whose recorded ``artifact=`` path does NOT exist on disk must not
   release the gate at preflight time (a forged approval naming a missing
   artifact fails closed). The matching-approval writer already asserts the
   artifact exists; preflight now re-checks at inspection time.

Contract (kept compatible): ``_latest_pending_human_gate`` still returns a
2-tuple; the first element is the list of armed gate ids (may hold several),
the second the comment-trail read error (or ``None``). Existing callers that
expect a single gate keep working: ``record_human_gate_approval`` uses the
first armed id; the preflight check uses the whole list.

The routing-metadata branch (``human_gate_pending`` key without/with an
``approval_artifact``) is unchanged — behaviour covered by
``tests/hermes_cli/test_kanban_blocked_completion_guard.py`` is pinned here
too, so the single-suite runs stay green.

Sandbox: the script lives OUTSIDE the repo tree, so — exactly like the ``ap``
fixture in ``test_kanban_blocked_completion_guard.py`` — it is copied into
pytest's tmp via subprocess and imported from the copy (the home_io_guard
refuses file I/O against the real operator home from inside a test). The
module is registered in ``sys.modules`` before exec (dataclasses resolve
string annotations through it under ``from __future__ import annotations``).
Boards under /home/moi/.hermes/kanban/ are READ-ONLY and never touched: every
test runs against an isolated tmp HERMES_HOME with its own kanban DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUTOPILOT_SCRIPTS = Path(
    "/home/moi/.hermes/profiles/orchestrator/skills/software-development/kanban-autopilot/scripts"
)
_has_autopilot = AUTOPILOT_SCRIPTS.exists()
pytestmark = pytest.mark.skipif(not _has_autopilot, reason="kanban-autopilot skill not present")


@pytest.fixture(scope="module")
def ap(tmp_path_factory):
    """The autopilot script imported from an isolated copy.

    The real script lives in the operator's Hermes home (outside this repo),
    so the home_io_guard refuses any file I/O against it from inside a test —
    including pytest's own linecache reads while building a traceback for an
    exception raised inside ``kanban_autopilot.py``. The guard's documented
    scope is 'Python filesystem calls, not arbitrary subprocess I/O', so the
    copy out happens via subprocess and the module is loaded from the
    temporary copy; tracebacks then resolve inside tmp_path.
    """
    if not _has_autopilot:
        pytest.skip("kanban-autopilot skill not present")
    import importlib.util
    import subprocess

    d = tmp_path_factory.mktemp("ap_gate_preflight")
    dst = d / "kanban_autopilot.py"
    subprocess.run(
        ["cp", str(AUTOPILOT_SCRIPTS / "kanban_autopilot.py"), str(dst)], check=True
    )
    spec = importlib.util.spec_from_file_location("kanban_autopilot_gate_preflight", dst)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module uses `from __future__ import
    # annotations`, and dataclasses resolves string annotations through
    # sys.modules[cls.__module__] — an unregistered module crashes init.
    sys.modules["kanban_autopilot_gate_preflight"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def conn(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (no real board touched)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def ap_env(tmp_path, monkeypatch, ap):
    """Isolated hermes home for the preflight checks: an ``orchestrator``
    profile dir and a hermes binary so checks 2-4 pass."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "orchestrator").mkdir(parents=True)
    (home / "bin").mkdir()
    (home / "bin" / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ap, "hermes_root", lambda: home)
    return home


def _minimal_routing_registry() -> dict:
    """Fail-closed hermetic registry: one enabled/dispatchable route."""
    return {
        "registry_version": "p2b-test",
        "kanban_stage_routing": {
            "CANONICAL_PROMOTION": {"default_routes": ["p2b_test_route"]},
        },
        "named_routes": {
            "p2b_test_route": {
                "provider": "ollama-cloud",
                "model": "glm-5.3-flash",
                "route_id": "p2b_test_route_id",
            },
        },
        "routes": {
            "p2b_test_route_id": {
                "provider": "ollama-cloud",
                "model": "glm-5.3-flash",
                "enabled": True,
                "dispatchable": True,
            },
        },
    }


def _profile_registry() -> dict:
    return {"profiles": [{"profile_key": "orchestrator"}]}


def _add_comment(conn, task_id, author, body):
    """Raw marker insert (the autopilot skill writes exactly this row shape)."""
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, author, body, 1),
    )
    conn.commit()


def _gate_check(res, name="human_gate_pending"):
    return [c for c in res.checks if c.name == name]


def _preflight(conn, ap, tid, dry_run=True):
    return ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={}, state_schema={}, dry_run=dry_run,
    )


PENDING = "HUMAN_GATE_PENDING:"
APPROVAL = "HUMAN_GATE_APPROVAL:"


# ---------------------------------------------------------------------------
# (a) a pending marker fails preflight naming the gate id
# ---------------------------------------------------------------------------

class TestPendingMarkerFailsPreflight:
    def test_pending_marker_fails_preflight_with_gate_id(self, conn, ap_env, ap):
        tid = kb.create_task(conn, title="gated card", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_A")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks, "pending marker must produce a human_gate_pending check"
        gate = checks[0]
        assert gate.passed is False
        assert "GATE_A" in gate.reason
        assert res.passed is False

    def test_pending_marker_empty_gate_id_still_fails(self, conn, ap_env, ap):
        """A blank-id pending marker is still armed (it names nothing, but the
        card is halted at a gate): preflight must fail closed."""
        tid = kb.create_task(conn, title="blank gate", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING}   ")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks
        assert checks[0].passed is False

    def test_no_marker_no_gate_check(self, conn, ap_env, ap):
        """A card with no gate markers carries no human_gate_pending check."""
        tid = kb.create_task(conn, title="plain card", assignee="orchestrator")
        _add_comment(conn, tid, "worker", "progress: half done")

        res = _preflight(conn, ap, tid)
        assert _gate_check(res) == []
        assert res.passed is True, res.to_dict()


# ---------------------------------------------------------------------------
# (b) approval with matching id + EXISTING artifact passes
# ---------------------------------------------------------------------------

class TestApprovalWithExistingArtifactPasses:
    def test_matching_approval_with_existing_artifact_passes(
        self, conn, ap_env, ap, tmp_path
    ):
        tid = kb.create_task(conn, title="approved card", assignee="orchestrator")
        artifact = tmp_path / "GATE_A.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_A")
        _add_comment(
            conn, tid, "human-operator", f"{APPROVAL} GATE_A artifact={artifact}"
        )

        res = _preflight(conn, ap, tid)
        # The matching approval released the only gate: no pending check.
        assert _gate_check(res) == []
        assert res.passed is True, res.to_dict()

    def test_matching_approval_by_any_author_releases_gate(
        self, conn, ap_env, ap, tmp_path
    ):
        """The approval may be authored by anyone (identity/authz is P2b's
        design decision on the write path; preflight judges the trail)."""
        tid = kb.create_task(conn, title="human approved", assignee="orchestrator")
        artifact = tmp_path / "review.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_X")
        _add_comment(conn, tid, "some-worker", f"{APPROVAL} GATE_X artifact={artifact}")

        res = _preflight(conn, ap, tid)
        assert _gate_check(res) == []
        assert res.passed is True, res.to_dict()

    def test_plain_comments_between_markers_are_ignored(
        self, conn, ap_env, ap, tmp_path
    ):
        tid = kb.create_task(conn, title="prose between", assignee="orchestrator")
        artifact = tmp_path / "g1.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "worker", "progress: half done")
        _add_comment(conn, tid, "human", f"{APPROVAL} g1 artifact={artifact}")

        res = _preflight(conn, ap, tid)
        assert _gate_check(res) == []

    def test_new_pending_after_approval_re_arms_gate(self, conn, ap_env, ap, tmp_path):
        """Trail ordering: pending g1, approval g1, NEW pending g1 — the gate
        is armed again by the later marker."""
        tid = kb.create_task(conn, title="re-armed", assignee="orchestrator")
        artifact = tmp_path / "g1.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "human", f"{APPROVAL} g1 artifact={artifact}")
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        assert "g1" in checks[0].reason


# ---------------------------------------------------------------------------
# (c) approval whose recorded artifact path is NOT on disk fails closed
# ---------------------------------------------------------------------------

class TestApprovalArtifactReverifiedOnDisk:
    def test_approval_with_missing_artifact_fails_preflight(self, conn, ap_env, ap):
        """A forged acceptance: the approval marker exists with a matching id
        but names an artifact path that is NOT on disk. The writer refuses to
        create such a row via record_human_gate_approval; a raw/foreign writer
        can still plant one, so preflight re-verifies at inspection time."""
        tid = kb.create_task(conn, title="forged approval", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_F")
        _add_comment(
            conn, tid, "anyone",
            f"{APPROVAL} GATE_F artifact=/nonexistent/approval-GATE_F.yaml",
        )

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks, "missing artifact must keep the gate armed at preflight"
        gate = checks[0]
        assert gate.passed is False
        assert "GATE_F" in gate.reason
        assert "artifact" in gate.reason.lower()
        assert res.passed is False

    def test_approval_without_artifact_component_fails_preflight(
        self, conn, ap_env, ap
    ):
        """``HUMAN_GATE_APPROVAL: <id>`` with no artifact= at all is exactly
        the shape review-5 proved a worker could plant: never a release."""
        tid = kb.create_task(conn, title="bare approval", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_B")
        _add_comment(conn, tid, "worker", f"{APPROVAL} GATE_B")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        assert "GATE_B" in checks[0].reason
        assert res.passed is False

    def test_approval_artifact_is_a_directory_fails(self, conn, ap_env, ap, tmp_path):
        """``Path.exists()`` is the on-disk assertion: a directory is not an
        approval artifact, so the gate stays armed."""
        tid = kb.create_task(conn, title="dir artifact", assignee="orchestrator")
        dir_artifact = tmp_path / "not_a_file"
        dir_artifact.mkdir()
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_D")
        _add_comment(conn, tid, "human", f"{APPROVAL} GATE_D artifact={dir_artifact}")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False

    def test_wrong_gate_approval_does_not_release_the_armed_gate(
        self, conn, ap_env, ap, tmp_path
    ):
        """An approval for a DIFFERENT gate id is noise for the armed gate."""
        tid = kb.create_task(conn, title="wrong id", assignee="orchestrator")
        artifact = tmp_path / "g2.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        _add_comment(conn, tid, "autopilot", f"{PENDING} g1")
        _add_comment(conn, tid, "human", f"{APPROVAL} g2 artifact={artifact}")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        assert "g1" in checks[0].reason


# ---------------------------------------------------------------------------
# (d) comment-trail READ ERROR fails the check with the error in the reason
# ---------------------------------------------------------------------------

class TestCommentReadErrorFailsClosed:
    def test_unreadable_trail_fails_with_err_in_reason(self, conn, ap_env, ap, monkeypatch):
        tid = kb.create_task(conn, title="read error", assignee="orchestrator")

        import sqlite3 as _sqlite3

        real_execute = conn.execute

        def boom_execute(sql, *a, **k):
            if "task_comments" in str(sql):
                raise _sqlite3.OperationalError("disk I/O error (probe)")
            return real_execute(sql, *a, **k)

        monkeypatch.setattr(conn, "execute", boom_execute)

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks, "a read error must surface a human_gate_pending check"
        gate = checks[0]
        assert gate.passed is False
        assert "disk I/O error" in gate.reason or "comment trail unreadable" in gate.reason
        assert res.passed is False

    def test_unreadable_trail_err_is_not_swallowed_by_later_checks(
        self, conn, ap_env, ap, monkeypatch
    ):
        """The failed check must survive to the final failure_reason (the
        first failing check names the block), not be folded away."""
        tid = kb.create_task(conn, title="read error 2", assignee="orchestrator")

        import sqlite3 as _sqlite3

        real_execute = conn.execute

        def boom_execute(sql, *a, **k):
            if "task_comments" in str(sql):
                raise _sqlite3.OperationalError("locked (probe)")
            return real_execute(sql, *a, **k)

        monkeypatch.setattr(conn, "execute", boom_execute)

        res = _preflight(conn, ap, tid)
        assert res.passed is False
        names = {c.name for c in res.checks if not c.passed}
        assert "human_gate_pending" in names


# ---------------------------------------------------------------------------
# (e) MULTIPLE armed gates fail closed naming ALL of them
# ---------------------------------------------------------------------------

class TestMultipleArmedGates:
    def test_two_armed_gates_fail_naming_both(self, conn, ap_env, ap):
        tid = kb.create_task(conn, title="two gates", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_1")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_2")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        gate = checks[0]
        assert "GATE_1" in gate.reason and "GATE_2" in gate.reason
        detail_ids = gate.detail.get("gates") if isinstance(gate.detail, dict) else None
        assert detail_ids, "detail must carry the armed gate list"
        assert set(detail_ids) == {"GATE_1", "GATE_2"}
        assert res.passed is False

    def test_earlier_gate_survives_approval_of_the_last_one(
        self, conn, ap_env, ap, tmp_path
    ):
        """The old single-armed-variable bug: pending G1, pending G2, approval
        G2 must NOT release G1 (review-5 probe: it returned no pending gate).
        G1 and G2 stay armed until EACH has its matching later approval."""
        tid = kb.create_task(conn, title="two gates one ok", assignee="orchestrator")
        artifact2 = tmp_path / "GATE_2.yaml"
        artifact2.write_text("decision: APPROVE\n", encoding="utf-8")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_1")
        _add_comment(conn, tid, "autopilot", f"{PENDING} GATE_2")
        _add_comment(conn, tid, "human", f"{APPROVAL} GATE_2 artifact={artifact2}")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        gate = checks[0]
        assert "GATE_1" in gate.reason
        assert "GATE_2" not in gate.reason  # G2 was legitimately released
        detail_ids = gate.detail.get("gates") if isinstance(gate.detail, dict) else None
        assert detail_ids == ["GATE_1"]

    def test_all_of_three_armed_named_when_none_approved(self, conn, ap_env, ap):
        tid = kb.create_task(conn, title="three gates", assignee="orchestrator")
        for gid in ("A1", "B2", "C3"):
            _add_comment(conn, tid, "autopilot", f"{PENDING} {gid}")

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        for gid in ("A1", "B2", "C3"):
            assert gid in checks[0].reason

    def test_parser_helper_returns_full_armed_list(self, conn, ap):
        """Direct contract check on the helper: list of ALL armed ids + err."""
        tid = kb.create_task(conn, title="helper contract", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} G1")
        _add_comment(conn, tid, "worker", "noise")
        _add_comment(conn, tid, "autopilot", f"{PENDING} G2")

        armed, err = ap._latest_pending_human_gate(conn, tid)
        assert err is None
        assert isinstance(armed, list)
        assert armed == ["G1", "G2"]

    def test_parser_helper_approval_with_blank_gate_id_does_not_release(
        self, conn, ap
    ):
        """An approval marker with a blank id cannot name the gate it closes;
        it must release nothing."""
        tid = kb.create_task(conn, title="blank approval", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} G9")
        _add_comment(conn, tid, "human", f"{APPROVAL}   artifact=/tmp/x.yaml")

        armed, err = ap._latest_pending_human_gate(conn, tid)
        assert armed == ["G9"]
        assert err is None


# ---------------------------------------------------------------------------
# (f)/(g) routing-metadata branch behaviour pinned (unchanged semantics)
# ---------------------------------------------------------------------------

def _set_routing_metadata(conn, task_id: str, mapping: dict) -> None:
    """Simulate a hypothetical board that carries a routing_metadata column."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "routing_metadata" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN routing_metadata TEXT")
    conn.execute(
        "UPDATE tasks SET routing_metadata = ? WHERE id = ?",
        (__import__("json").dumps(mapping), task_id),
    )
    conn.commit()


class TestRoutingMetadataBranchPinned:
    def test_metadata_pending_without_artifact_fails(self, conn, ap_env, ap):
        tid = kb.create_task(conn, title="meta gated", assignee="orchestrator")
        _set_routing_metadata(conn, tid, {"human_gate_pending": "META_GATE"})

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is False
        assert "META_GATE" in checks[0].reason
        assert res.passed is False

    def test_metadata_pending_with_artifact_passes(self, conn, ap_env, ap):
        tid = kb.create_task(conn, title="meta approved", assignee="orchestrator")
        _set_routing_metadata(conn, tid, {
            "human_gate_pending": "META_GATE",
            "approval_artifact": "/approvals/independent_poc_review_approved.yaml",
        })

        res = _preflight(conn, ap, tid)
        checks = _gate_check(res)
        assert checks and checks[0].passed is True
        assert res.passed is True, res.to_dict()

    def test_board_without_metadata_column_tolerated(self, conn, ap_env, ap):
        """P2 creates no schema: a board without the column must not break
        preflight (the branch falls through to the comment markers)."""
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        assert "routing_metadata" not in cols
        tid = kb.create_task(conn, title="plain", assignee="orchestrator")
        res = _preflight(conn, ap, tid)
        assert _gate_check(res) == []


# ---------------------------------------------------------------------------
# Writer helpers unchanged (record_human_gate_approval keeps its on-disk
# assertion; mark_human_gate_pending keeps its blank-id refusal)
# ---------------------------------------------------------------------------

class TestWritersUnchanged:
    def test_writer_approval_requires_existing_artifact(self, conn, ap, tmp_path):
        tid = kb.create_task(conn, title="writer", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} W1")
        with pytest.raises(ValueError):
            ap.record_human_gate_approval(conn, tid, "W1", "/nonexistent/w1.yaml")
        artifact = tmp_path / "W1.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        assert ap.record_human_gate_approval(conn, tid, "W1", str(artifact)) is True

    def test_writer_pending_refuses_blank_gate_id(self, conn, ap):
        tid = kb.create_task(conn, title="writer2", assignee="orchestrator")
        with pytest.raises(ValueError):
            ap.mark_human_gate_pending(conn, tid, "   ")

    def test_writer_second_gate_approval_released_first_gate_only(
        self, conn, ap, tmp_path
    ):
        """The multi-gate parser keeps the writer's single-gate semantics:
        with two gates armed the writer records the approval for the gate it
        names (first armed id is still accepted when it matches)."""
        tid = kb.create_task(conn, title="writer3", assignee="orchestrator")
        _add_comment(conn, tid, "autopilot", f"{PENDING} G1")
        _add_comment(conn, tid, "autopilot", f"{PENDING} G2")
        artifact = tmp_path / "G1.yaml"
        artifact.write_text("decision: APPROVE\n", encoding="utf-8")
        # The writer refuses to approve G1 while G1 is armed and it names G1
        # — armed list now contains both; the matching-id rule still applies.
        assert ap.record_human_gate_approval(conn, tid, "G1", str(artifact)) is True
        armed, err = ap._latest_pending_human_gate(conn, tid)
        assert err is None
        assert armed == ["G2"]

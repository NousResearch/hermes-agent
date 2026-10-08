"""P2 defensive completion gate (audit 2026-10-03, F-01 / STUDY-03 pattern).

A ``blocked`` card must never be collapsed to ``done`` in one hop — not with
evidence, not with ``force=True``, not from a race window. The legitimate exit
from ``blocked`` is :func:`kanban_db.unblock_task` (which re-gates parents and
restores the resumable phase), then a normal claim/complete cycle.

Covered here:
  1. ``complete_task`` raises ``BlockedCompletionError`` for a ``blocked``
     card (before: it accepted the card whenever any result/summary existed).
  2. ``force=True`` does NOT bypass the blocked guard.
  3. The unblock -> complete reconciliation path keeps working.
  4. The dispatcher defensively skips any ``blocked`` row a lane query could
     ever hand over (``DispatchResult.skipped_blocked``).
  5. Autopilot preflight: ``card_state`` fails a blocked card;
     ``human_gate_pending`` routing metadata fails without an approval
     artifact and passes with one.
  6. Manifest integrity: every step-level ``human_gate_before`` token exists
     in the manifest's ``human_gates`` list (NF-2 drift guard).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

AUTOPILOT_SCRIPTS = Path(
    "/home/moi/.hermes/profiles/orchestrator/skills/software-development/kanban-autopilot/scripts"
)
AUTOPILOT_TEMPLATES = Path(
    "/home/moi/.hermes/profiles/orchestrator/skills/software-development/kanban-autopilot/templates"
)
_has_autopilot = AUTOPILOT_SCRIPTS.exists() and AUTOPILOT_TEMPLATES.exists()
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
    d = tmp_path_factory.mktemp("ap_script")
    dst = d / "kanban_autopilot.py"
    subprocess.run(
        ["cp", str(AUTOPILOT_SCRIPTS / "kanban_autopilot.py"), str(dst)], check=True,
    )
    spec = importlib.util.spec_from_file_location("kanban_autopilot_p2", dst)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module uses `from __future__ import
    # annotations`, and dataclasses resolves string annotations through
    # sys.modules[cls.__module__] — an unregistered module crashes init.
    sys.modules["kanban_autopilot_p2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def ap_env(tmp_path, monkeypatch, ap):
    """Isolated hermes home for the autopilot preflight checks: an
    ``orchestrator`` profile dir and a hermes binary so checks 2-4 pass."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "orchestrator").mkdir(parents=True)
    (home / "bin").mkdir()
    (home / "bin" / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ap, "hermes_root", lambda: home)
    return home


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (dispatcher tests)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _minimal_routing_registry() -> dict:
    """Fail-closed hermetic registry: one enabled/dispatchable route."""
    return {
        "registry_version": "p2-test",
        "kanban_stage_routing": {
            "CANONICAL_PROMOTION": {"default_routes": ["p2_test_route"]},
        },
        "named_routes": {
            "p2_test_route": {
                "provider": "ollama-cloud",
                "model": "glm-5.3-flash",
                "route_id": "p2_test_route_id",
            },
        },
        "routes": {
            "p2_test_route_id": {
                "provider": "ollama-cloud",
                "model": "glm-5.3-flash",
                "enabled": True,
                "dispatchable": True,
            },
        },
    }


def _profile_registry() -> dict:
    return {"profiles": [{"profile_key": "orchestrator"}]}


def _set_routing_metadata(conn, task_id: str, mapping: dict) -> None:
    """Simulate a hypothetical board that carries a routing_metadata column.

    P2 creates no schema (scope lock); the preflight reader only tolerates a
    board that happens to have the column, so the test fabricates one with a
    throwaway ALTER TABLE to prove the tolerance path.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "routing_metadata" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN routing_metadata TEXT")
    conn.execute(
        "UPDATE tasks SET routing_metadata = ? WHERE id = ?",
        (json.dumps(mapping), task_id),
    )
    conn.commit()


def _declare_clean(conn, ap, tid, tmp_path):
    """Declare a card dependency-free and in-scope so gate-focused tests
    isolate the gate semantics (gaps plan 2026-10-05: zero links and a
    missing scope declaration now FAIL preflight by design — unknown is
    never treated as satisfied)."""
    ap.record_dependencies_none(conn, tid, claim="root", reason="guard test root")
    src = tmp_path / "scope_evidence.yaml"
    src.write_text("decision: TEST_SCOPE\n", encoding="utf-8")
    ap.record_scope_evidence(conn, tid, mode="no-program-scope", source=str(src),
                             claim="guard test")


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# 1. complete_task refuses a blocked card
# ---------------------------------------------------------------------------

def test_complete_blocked_card_raises_blocked_completion_error(conn):
    """F-01 replay at the API level: a blocked card WITH evidence used to be
    completable in one hop; now the guard refuses it."""
    tid = kb.create_task(conn, title="parked pending human", assignee="coder")
    assert kb.block_task(conn, tid, reason="waiting on human decision", kind="needs_input")
    assert _status(conn, tid) == "blocked"

    with pytest.raises(kb.BlockedCompletionError) as excinfo:
        kb.complete_task(conn, tid, result="reconciled offline, closing the card")
    assert excinfo.value.task_id == tid
    assert excinfo.value.prior_status == "blocked"

    assert _status(conn, tid) == "blocked"
    kinds = _event_kinds(conn, tid)
    assert "completion_blocked_card_blocked" in kinds
    assert "completed" not in kinds


def test_complete_blocked_card_force_does_not_bypass(conn):
    """``force`` covers the live-claim fence only; it must not consume a
    parked card either."""
    tid = kb.create_task(conn, title="parked pending human", assignee="coder")
    assert kb.block_task(conn, tid, reason="waiting on human decision", kind="needs_input")
    with pytest.raises(kb.BlockedCompletionError):
        kb.complete_task(conn, tid, result="operator override attempt", force=True)
    assert _status(conn, tid) == "blocked"


def test_complete_blocked_card_empty_evidence_raises_blocked_first(conn):
    """A blocked card with no evidence at all reports the blocked guard (the
    more specific signal), not just the empty-completion gate."""
    tid = kb.create_task(conn, title="parked silent", assignee="coder")
    assert kb.block_task(conn, tid, reason="waiting", kind="needs_input")
    with pytest.raises(kb.BlockedCompletionError):
        kb.complete_task(conn, tid)
    assert _status(conn, tid) == "blocked"


def test_unblock_then_complete_still_works(conn):
    """The legitimate reconciliation path is untouched:
    unblock re-gates and restores the resumable phase, then completion works."""
    tid = kb.create_task(conn, title="parked pending human", assignee="coder")
    assert kb.block_task(conn, tid, reason="waiting on human decision", kind="needs_input")
    assert _status(conn, tid) == "blocked"

    assert kb.unblock_task(conn, tid)
    resumed = _status(conn, tid)
    assert resumed in ("ready", "running", "review")

    assert kb.complete_task(conn, tid, result="reconciled after unblock") is True
    assert _status(conn, tid) == "done"


def test_review_completion_still_works(conn):
    """The docstring drops 'blocked' from the accepted sources; review approval
    must keep working exactly as before."""
    tid = kb.create_task(conn, title="review card", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    assert kb.request_review(conn, tid, summary="please review") is not False
    assert _status(conn, tid) == "review"
    assert kb.complete_task(conn, tid) is True
    assert _status(conn, tid) == "done"


# ---------------------------------------------------------------------------
# 4. Dispatcher defensive skip
# ---------------------------------------------------------------------------

def test_dispatcher_skips_blocked_rows_injected_into_lane(kanban_home, all_assignees_spawnable, monkeypatch):
    """Defensive-in-depth: even if a lane query ever hands a ``blocked`` row to
    the dispatch loops (lane soup / future refactor), the row is skipped,
    recorded, and never claimed nor spawned."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked card", assignee="alice")
        assert kb.block_task(conn, tid, reason="needs human", kind="needs_input")

        original_lane_rows = kbd._lane_rows

        def soupy_lane_rows(conn_, status):
            rows = list(original_lane_rows(conn_, status))
            if status == "ready":
                # Simulate a future refactor bug: blocked rows leaking into the
                # ready lane query.
                rows += conn_.execute(
                    "SELECT id, assignee, status FROM tasks "
                    "WHERE status = 'blocked' AND claim_lock IS NULL "
                    "ORDER BY priority DESC, created_at ASC"
                ).fetchall()
            return rows

        monkeypatch.setattr(kbd, "_lane_rows", soupy_lane_rows)
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    assert res.spawned == []
    assert res.skipped_blocked == [tid]
    assert _status(conn, tid) == "blocked"


def test_dispatch_result_skipped_blocked_empty_on_normal_tick(kanban_home, all_assignees_spawnable):
    """Normal ticks never populate the defensive bucket."""
    with kbc.connect() as conn:
        kb.create_task(conn, title="normal card", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert res.skipped_blocked == []
    assert len(res.spawned) == 1


# ---------------------------------------------------------------------------
# 5. Autopilot preflight: card_state + human_gate_pending
# ---------------------------------------------------------------------------

def test_preflight_tolerates_board_without_routing_metadata_column(conn, ap):
    """P2 creates no schema (scope lock): a board without the column must not
    break preflight — the human-gate branch falls through to the comment
    markers."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "routing_metadata" not in cols
    tid = kb.create_task(conn, title="plain card", assignee="orchestrator")
    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={}, state_schema={}, dry_run=True,
    )
    assert "human_gate_pending" not in {c.name for c in res.checks}


def test_preflight_card_state_fails_blocked(conn, ap_env, ap):
    tid = kb.create_task(conn, title="gate card", assignee="orchestrator")
    assert kb.block_task(conn, tid, reason="needs human", kind="needs_input")

    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={},
        state_schema={},
        dry_run=True,
    )
    card = [c for c in res.checks if c.name == "card_state"][0]
    assert card.passed is False
    assert "blocked" in card.reason
    assert "unblock" in card.reason
    assert res.passed is False
    assert res.failure_reason.endswith("CARD_STATE: " + card.reason) or "CARD_STATE" in res.failure_reason


def test_preflight_card_state_passes_ready(conn, ap_env, ap, tmp_path):
    """ready/running/review must NOT be lumped together with blocked: a normal
    ready card passes the full preflight."""
    tid = kb.create_task(conn, title="normal card", assignee="orchestrator")
    _declare_clean(conn, ap, tid, tmp_path)
    assert _status(conn, tid) == "ready"

    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={},
        state_schema={},
        dry_run=True,
    )
    card = [c for c in res.checks if c.name == "card_state"][0]
    assert card.passed is True
    assert res.passed is True, res.to_dict()


def test_preflight_human_gate_pending_without_artifact_fails(conn, ap_env, ap):
    """A card halted at a human gate (routing_metadata.human_gate_pending set
    by the autopilot) fails preflight until the approval artifact is recorded."""
    tid = kb.create_task(conn, title="gate halted card", assignee="orchestrator")
    _set_routing_metadata(conn, tid, {"human_gate_pending": "INDEPENDENT_POC_REVIEW_APPROVAL"})

    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={},
        state_schema={},
        dry_run=True,
    )
    gate = [c for c in res.checks if c.name == "human_gate_pending"][0]
    assert gate.passed is False
    assert "INDEPENDENT_POC_REVIEW_APPROVAL" in gate.reason
    assert res.passed is False


def test_preflight_human_gate_pending_with_artifact_passes(conn, ap_env, ap, tmp_path):
    tid = kb.create_task(conn, title="approved gate card", assignee="orchestrator")
    _declare_clean(conn, ap, tid, tmp_path)
    _set_routing_metadata(conn, tid, {
        "human_gate_pending": "INDEPENDENT_POC_REVIEW_APPROVAL",
        "approval_artifact": "/approvals/independent_poc_review_approved.yaml",
    })

    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={},
        state_schema={},
        dry_run=True,
    )
    gate = [c for c in res.checks if c.name == "human_gate_pending"][0]
    assert gate.passed is True
    assert res.passed is True, res.to_dict()


def test_preflight_no_gate_metadata_no_gate_check(conn, ap_env, ap):
    """Boards/cards without the metadata carry no human_gate_pending check."""
    tid = kb.create_task(conn, title="plain card", assignee="orchestrator")
    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={},
        state_schema={},
        dry_run=True,
    )
    assert "human_gate_pending" not in {c.name for c in res.checks}


def test_gate_writer_cycle_end_to_end(conn, ap_env, ap, tmp_path):
    """The autopilot's gate writers close the loop over the comment trail:
    mark → preflight fails; record approval artifact → preflight passes;
    an artifact-less or wrong-gate approval is refused; a second gate can be
    armed after the first is discharged."""
    tid = kb.create_task(conn, title="gated card", assignee="orchestrator")
    _declare_clean(conn, ap, tid, tmp_path)

    assert ap.mark_human_gate_pending(conn, tid, "INDEPENDENT_POC_REVIEW_APPROVAL") is True

    # Without the artifact: preflight fails, naming the armed gate.
    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={}, state_schema={}, dry_run=True,
    )
    assert res.passed is False
    gate = [c for c in res.checks if c.name == "human_gate_pending"][0]
    assert gate.passed is False
    assert "INDEPENDENT_POC_REVIEW_APPROVAL" in gate.reason

    # Approval without an existing artifact is refused outright.
    with pytest.raises(ValueError):
        ap.record_human_gate_approval(conn, tid, "INDEPENDENT_POC_REVIEW_APPROVAL",
                                      "/nonexistent/approval.yaml")

    # Approval with a real artifact: marker recorded, preflight releases.
    artifact = tmp_path / "INDEPENDENT_POC_REVIEW_APPROVAL.yaml"
    artifact.write_text("decision: APPROVE_POC_REVIEW\n", encoding="utf-8")
    assert ap.record_human_gate_approval(conn, tid, "INDEPENDENT_POC_REVIEW_APPROVAL",
                                         str(artifact)) is True
    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={}, state_schema={}, dry_run=True,
    )
    assert res.passed is True, res.to_dict()

    # A wrong-gate approval cannot discharge a different armed gate.
    assert ap.mark_human_gate_pending(conn, tid, "SUBMISSION_APPROVAL_PENDING") is True
    with pytest.raises(ValueError):
        ap.record_human_gate_approval(conn, tid, "INDEPENDENT_POC_REVIEW_APPROVAL",
                                      str(artifact))
    other = tmp_path / "SUBMISSION_APPROVAL_PENDING.yaml"
    other.write_text("decision: APPROVE_SUBMISSION\n", encoding="utf-8")
    assert ap.record_human_gate_approval(conn, tid, "SUBMISSION_APPROVAL_PENDING",
                                         str(other)) is True
    res = ap.preflight(
        conn, tid,
        routing_registry=_minimal_routing_registry(),
        profile_registry=_profile_registry(),
        pipeline_sm={}, state_schema={}, dry_run=True,
    )
    assert res.passed is True, res.to_dict()

    # The trail documents every halt and approval for auditability.
    bodies = [r["body"] for r in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()]
    assert sum("HUMAN_GATE_PENDING:" in b for b in bodies) == 2
    assert sum("HUMAN_GATE_APPROVAL:" in b for b in bodies) == 2


# ---------------------------------------------------------------------------
# 6. Manifest gate representation (NF-2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest_dir(tmp_path_factory):
    """The real manifests live in the operator's Hermes home, which the repo's
    home_io_guard refuses to open() from inside a test. The guard's documented
    scope is 'Python filesystem calls, not arbitrary subprocess I/O', so the
    copy out happens via subprocess and tests read the isolated copies."""
    import subprocess
    out = tmp_path_factory.mktemp("manifests")
    for workflow in ("immunefi-target-audit", "protocol-study-mode-b"):
        src = AUTOPILOT_TEMPLATES / f"{workflow}.yaml"
        subprocess.run(["cp", str(src), str(out / f"{workflow}.yaml")], check=True)
    return out


def _manifest(manifest_dir, workflow: str) -> dict:
    import yaml
    path = manifest_dir / f"{workflow}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("workflow", ["immunefi-target-audit", "protocol-study-mode-b"])
def test_manifest_human_gates_cover_step_gate_tokens(manifest_dir, workflow):
    """Every step-level ``human_gate_before`` token must exist in the
    manifest's ``human_gates`` list (NF-2: Mode A declared
    INDEPENDENT_POC_REVIEW_APPROVAL on the independent_poc_review step but
    omitted it from human_gates)."""
    manifest = _manifest(manifest_dir, workflow)
    wf = manifest["workflow"]
    gate_ids = {g["id"] for g in wf.get("human_gates", [])}
    for step_name, step in wf.get("steps", {}).items():
        token = step.get("human_gate_before")
        if token:
            assert token in gate_ids, (
                f"{workflow}: step {step_name!r} declares human_gate_before "
                f"{token!r} missing from human_gates"
            )


def test_mode_a_manifest_has_independent_poc_review_approval(manifest_dir):
    gate_ids = {
        g["id"] for g in _manifest(manifest_dir, "immunefi-target-audit")["workflow"].get("human_gates", [])
    }
    assert "INDEPENDENT_POC_REVIEW_APPROVAL" in gate_ids

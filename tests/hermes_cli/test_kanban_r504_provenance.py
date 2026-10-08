"""R5-04 regression tests — review 5, fixed in the round-5b follow-up.

`kanban_request_review` (and any worker-driven transition that closes a run
with caller metadata) must never carry the TRUSTED run-metadata keys
(``off_board_run``, ``resolved_route_provenance``,
``external_artifacts_preserved``) from caller metadata into ``_end_run`` /
``_synthesize_ended_run``: a worker that never spawned a dispatcher could
otherwise forge dispatcher-side route provenance on an off-board run, and an
``off_board_run`` forged envelope could downgrade/upgrade the origin story of
a closing run (review 5, R5-04 probe
``test_review_tool_can_forge_dispatcher_provenance_without_spawn``).

`complete_task` already strips all three keys (round-2 HIGH #2 policy);
`request_review` only stripped ``off_board_run`` — this file pins the fix:
ALL protected keys are stripped at every worker-driven run closer, while the
dispatcher's own write path (``kanban_db_dispatch`` claim/spawn) is untouched.

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixture conventions).
Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME/kanban home with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _claimed_task(conn, title="r5-04 card"):
    tid = kb.create_task(conn, title=title, assignee="coder")
    claimed = kb.claim_task(conn, tid, claimer="host:mock")
    assert claimed is not None, "claim must succeed for the transition path"
    return tid


def _run_metadata(conn, task_id):
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or not row["metadata"]:
        return {}
    try:
        data = json.loads(row["metadata"])
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


FORGED_PROVENANCE = {
    "route": "GPT_61_SOL_REVIEW",
    "provider": "openai",
    "model": "gpt-6.1-sol",
    "served_model": "gpt-6.1-sol",
    "route_id": "fake-route-000",
}


class TestR504ReviewToolCannotForgeDispatcherProvenance:
    def test_forged_provenance_never_reaches_the_closed_run(self, conn):
        """The exact review-5 probe shape: a claimed card, an off-board-style
        origin registered via the API, then ``request_review`` with a forged
        ``resolved_route_provenance``. The closed run must NOT carry it."""
        tid = _claimed_task(conn)
        kb.record_off_board_run(
            conn, tid, served_model="gpt-6.1-sol", provider="openai",
            launch_ref="ext://review5b/forge",
        )
        result = kb.request_review(
            conn, tid, summary="handoff",
            metadata={"resolved_route_provenance": dict(FORGED_PROVENANCE)},
            expected_run_id=None, force=True, with_reason=True,
        )
        ok, reason = result
        assert ok is True, reason
        meta = _run_metadata(conn, tid)
        assert "resolved_route_provenance" not in meta, meta
        # The real origin the API recorded survives untouched.
        assert meta.get("off_board_run", {}).get("served_model") == "gpt-6.1-sol"

    def test_forged_off_board_envelope_stripped_too(self, conn):
        """A caller cannot downgrade/upgrade the origin story via a forged
        ``off_board_run`` envelope through the review handoff either."""
        tid = _claimed_task(conn)
        result = kb.request_review(
            conn, tid, summary="handoff",
            metadata={"off_board_run": {"served_model": "totally-fake"}},
            expected_run_id=None, force=True, with_reason=True,
        )
        ok, reason = result
        assert ok is True, reason
        meta = _run_metadata(conn, tid)
        assert "off_board_run" not in meta, meta

    def test_forged_preserved_artifacts_key_stripped(self, conn):
        """``external_artifacts_preserved`` is a bind_external_artifacts-owned
        key: the review handoff must not let a caller pre-stage it (the P3b
        staging reads only THIS invocation's bindings, but the key itself
        must not ride caller metadata into the run either)."""
        tid = _claimed_task(conn)
        result = kb.request_review(
            conn, tid, summary="handoff",
            metadata={"external_artifacts_preserved": [{"stored_path": "/tmp/x"}]},
            expected_run_id=None, force=True, with_reason=True,
        )
        ok, reason = result
        assert ok is True, reason
        meta = _run_metadata(conn, tid)
        assert "external_artifacts_preserved" not in meta, meta

    def test_plain_metadata_still_flows_through(self, conn):
        """Backwards compatibility: ordinary handoff metadata is untouched."""
        tid = _claimed_task(conn)
        result = kb.request_review(
            conn, tid, summary="handoff",
            metadata={"notes": "implemented feature X", "artifacts": []},
            expected_run_id=None, force=True, with_reason=True,
        )
        ok, reason = result
        assert ok is True, reason
        meta = _run_metadata(conn, tid)
        assert meta.get("notes") == "implemented feature X", meta


class TestR504OtherWorkerClosersStripProtectedKeys:
    def test_block_task_carries_no_provenance_key(self, conn):
        """``block_task`` closes the run without metadata today; a future
        caller-supplied payload must not become a provenance channel. Pin the
        invariant at the helper level instead of the (metadata-less) call."""
        tid = _claimed_task(conn)
        ok = kb.block_task(conn, tid, kind="needs_input", reason="r5-04")
        assert ok is True
        meta = _run_metadata(conn, tid)
        assert "resolved_route_provenance" not in meta and "off_board_run" not in meta

    def test_strip_protected_run_metadata_covers_all_three(self):
        """The policy helper strips every protected key, in place."""
        meta = {
            "off_board_run": {"served_model": "x"},
            "resolved_route_provenance": dict(FORGED_PROVENANCE),
            "external_artifacts_preserved": [{"stored_path": "/y"}],
            "keep": "me",
        }
        out = kb._strip_protected_run_metadata(meta)
        assert out is meta  # in-place
        assert set(meta) == {"keep"}
        assert meta["keep"] == "me"

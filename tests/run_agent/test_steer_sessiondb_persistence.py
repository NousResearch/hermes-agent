"""Regression: mid-turn /steer must reach the durable SQLite SessionDB.

A tool result is incrementally flushed to ``state.db`` by
``AIAgent._flush_messages_to_session_db`` BEFORE a mid-turn ``/steer`` appends
its out-of-band marker to that same ``role:"tool"`` message dict. Because the
flush path dedups by ``id(msg)`` and ``SessionDB.append_message`` is INSERT-only,
a later flush used to SKIP the already-flushed dict — leaving the stale pre-steer
content in SQLite even though the live transcript and the model both saw the
steered content.

The fix makes the flush path track each flushed row id + content and UPDATE the
durable row when a flushed dict is mutated in place. These tests pin that
contract across all five post-flush mutation sites:

    1. concurrent per-tool drain   (tool_executor.py:835)
    2. sequential per-tool drain   (tool_executor.py:1478)
    3. concurrent batch-end drain  (tool_executor.py:848)
    4. sequential batch-end drain  (tool_executor.py:1518)
    5. pre-API drain               (conversation_loop.py:650-699)

In every case the durable SQLite tool row must end up containing the steer
marker, with NO stale pre-steer row and NO duplicate row.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.prompt_builder import STEER_MARKER_OPEN
from agent.tool_dispatch_helpers import make_tool_result_message
from hermes_state import SessionDB
from run_agent import AIAgent

STEER_TEXT = "please prefer the smaller fix"
SESSION_ID = "steer-db-test"


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(session_db, hermes_home: Path, session_id=SESSION_ID):
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=session_id,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._ensure_db_session()
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_rows(db, session_id=SESSION_ID):
    return [r for r in db.get_messages(session_id) if r.get("role") == "tool"]


def _assert_single_steered_row(db, live_tool_msg):
    """The durable transcript holds exactly one tool row, steered, matching live."""
    rows = _tool_rows(db)
    assert len(rows) == 1, (
        f"expected exactly one tool row (no stale + no duplicate), got "
        f"{[r['content'] for r in rows]}"
    )
    row_content = rows[0]["content"]
    assert STEER_MARKER_OPEN in row_content, "durable row missing the /steer marker"
    assert STEER_TEXT in row_content, "durable row missing the steer text"
    assert row_content == live_tool_msg["content"], (
        "durable row diverged from the content the model actually saw"
    )


# ---------------------------------------------------------------------------
# Root cause: a post-flush in-place mutation must UPDATE the existing row,
# not be silently skipped by id() dedup nor duplicated by a re-INSERT.
# ---------------------------------------------------------------------------
def test_post_flush_inplace_mutation_updates_existing_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            agent = _make_agent(db, Path(tmp) / "hermes-home")
            tool_msg = {
                "role": "tool",
                "content": "tool result",
                "tool_call_id": "c1",
                "tool_name": "web_search",
            }
            messages = [tool_msg]

            # First incremental flush writes the pre-steer row + tracks its id.
            agent._flush_messages_to_session_db(messages)
            assert [r["content"] for r in _tool_rows(db)] == ["tool result"]

            # /steer mutates the already-flushed dict in place.
            agent.steer(STEER_TEXT)
            agent._apply_pending_steer_to_tool_results(messages, 1)
            assert STEER_MARKER_OPEN in tool_msg["content"]

            # The next flush must UPDATE the existing row in place.
            agent._flush_messages_to_session_db(messages)
            _assert_single_steered_row(db, tool_msg)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Site 1 & 2: executor per-tool drains. Steer is pending BEFORE the per-tool
# drain fires; the subsequent flush (next loop iteration / final persist) must
# carry the steer to the durable row.
# ---------------------------------------------------------------------------
def test_concurrent_per_tool_drain_persists_steer():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            agent = _make_agent(db, Path(tmp) / "hermes-home")
            assistant = SimpleNamespace(content="", tool_calls=[_mock_tool_call(call_id="c1")])
            messages: list = []
            agent.steer(STEER_TEXT)  # pending before the per-tool drain (835)

            with (
                patch.object(agent, "_invoke_tool", side_effect=lambda *a, **k: "search result"),
                patch(
                    "agent.tool_executor.maybe_persist_tool_result",
                    side_effect=lambda **k: k["content"],
                ),
            ):
                agent._execute_tool_calls_concurrent(assistant, messages, "task-1")

            assert STEER_MARKER_OPEN in messages[-1]["content"]
            # Persistence catches up on the next flush (next iteration / final).
            agent._flush_messages_to_session_db(messages)
            _assert_single_steered_row(db, messages[-1])
        finally:
            db.close()


def test_sequential_per_tool_drain_persists_steer():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            agent = _make_agent(db, Path(tmp) / "hermes-home")
            assistant = SimpleNamespace(content="", tool_calls=[_mock_tool_call(call_id="c1")])
            messages: list = []
            agent.steer(STEER_TEXT)  # pending before the per-tool drain (1478)

            with (
                patch("run_agent.handle_function_call", side_effect=lambda *a, **k: "search result"),
                patch(
                    "agent.tool_executor.maybe_persist_tool_result",
                    side_effect=lambda **k: k["content"],
                ),
            ):
                agent._execute_tool_calls_sequential(assistant, messages, "task-1")

            assert STEER_MARKER_OPEN in messages[-1]["content"]
            agent._flush_messages_to_session_db(messages)
            _assert_single_steered_row(db, messages[-1])
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Site 3 & 4: executor batch-end drains. The steer must arrive AFTER the final
# per-tool drain but BEFORE the batch-end drain. enforce_turn_budget runs in
# exactly that window (lines 841/1512), so patching it to set the steer makes
# the batch-end drain (848/1518) the mutation site — already-flushed dict.
# ---------------------------------------------------------------------------
def test_concurrent_batch_end_drain_persists_steer():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            agent = _make_agent(db, Path(tmp) / "hermes-home")
            assistant = SimpleNamespace(content="", tool_calls=[_mock_tool_call(call_id="c1")])
            messages: list = []

            def _set_steer_at_budget(*_a, **_k):
                # Fires after the per-tool drain (835), before batch-end (848).
                agent.steer(STEER_TEXT)

            with (
                patch.object(agent, "_invoke_tool", side_effect=lambda *a, **k: "search result"),
                patch(
                    "agent.tool_executor.maybe_persist_tool_result",
                    side_effect=lambda **k: k["content"],
                ),
                patch("agent.tool_executor.enforce_turn_budget", side_effect=_set_steer_at_budget),
            ):
                agent._execute_tool_calls_concurrent(assistant, messages, "task-1")

            # The batch-end drain — not a per-tool drain — applied the steer.
            assert STEER_MARKER_OPEN in messages[-1]["content"]
            assert agent._pending_steer is None
            agent._flush_messages_to_session_db(messages)
            _assert_single_steered_row(db, messages[-1])
        finally:
            db.close()


def test_sequential_batch_end_drain_persists_steer():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            agent = _make_agent(db, Path(tmp) / "hermes-home")
            assistant = SimpleNamespace(content="", tool_calls=[_mock_tool_call(call_id="c1")])
            messages: list = []

            def _set_steer_at_budget(*_a, **_k):
                # Fires after the per-tool drain (1478), before batch-end (1518).
                agent.steer(STEER_TEXT)

            with (
                patch("run_agent.handle_function_call", side_effect=lambda *a, **k: "search result"),
                patch(
                    "agent.tool_executor.maybe_persist_tool_result",
                    side_effect=lambda **k: k["content"],
                ),
                patch("agent.tool_executor.enforce_turn_budget", side_effect=_set_steer_at_budget),
            ):
                agent._execute_tool_calls_sequential(assistant, messages, "task-1")

            assert STEER_MARKER_OPEN in messages[-1]["content"]
            assert agent._pending_steer is None
            agent._flush_messages_to_session_db(messages)
            _assert_single_steered_row(db, messages[-1])
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Site 5: pre-API drain. The conversation loop appends the marker inline
# (conversation_loop.py:650-699) instead of calling the shared helper, onto a
# tool dict that was already flushed during a PRIOR tool batch. The next flush
# must update that row and leave no stale pre-steer row behind.
#
# This drives the REAL ``run_conversation`` loop (not a mirror of the inline
# logic) so the test fails if the production pre-API drain branch ever drifts:
# iteration 1 runs a tool and durably flushes its result row; the /steer then
# arrives AFTER that batch (so the executor per-tool / batch-end drains cannot
# consume it); on iteration 2 the real pre-API drain is the only code that can
# carry the marker onto the already-flushed tool dict, and end-of-turn
# persistence must UPDATE that row in place.
# ---------------------------------------------------------------------------
def test_pre_api_drain_persists_steer_no_stale_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            agent = _make_agent(db, Path(tmp) / "hermes-home")
            tool_call = _mock_tool_call(call_id="c1")
            # Iteration 1 → tool call; iteration 2 → final text response.
            agent.client.chat.completions.create.side_effect = [
                _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
                _mock_response(content="done", finish_reason="stop"),
            ]

            captured: dict = {}

            def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
                # Stand in for iteration 1's tool batch: append the result and
                # durably flush it so id(tool_msg) lands in
                # _flushed_db_message_ids — the "already flushed in a prior
                # batch" precondition for the pre-API drain.
                tool_msg = make_tool_result_message("web_search", "search result", "c1")
                messages.append(tool_msg)
                agent._flush_messages_to_session_db(messages)
                captured["tool_msg"] = tool_msg
                # The /steer arrives AFTER the batch. The executor per-tool and
                # batch-end drains have already passed (and are bypassed by this
                # fake anyway), so ONLY iteration 2's real pre-API drain can
                # inject it onto this already-flushed dict.
                agent.steer(STEER_TEXT)

            with (
                patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
                patch.object(agent, "_spawn_background_review"),
            ):
                result = agent.run_conversation("look it up")

            tool_msg = captured["tool_msg"]
            assert result["final_response"] == "done"
            # The real pre-API drain consumed the steer and injected the marker.
            # (The executor drains were bypassed, so nothing else could have.)
            assert agent._pending_steer is None
            assert STEER_MARKER_OPEN in tool_msg["content"]
            # End-of-turn persistence UPDATED the prior row in place: exactly one
            # steered tool row, no stale pre-steer row, no duplicate.
            _assert_single_steered_row(db, tool_msg)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# The fix leans on the AFTER UPDATE FTS triggers to re-index the steered
# content. Pin the issue's stated search/audit/history impact: after the
# post-flush UPDATE the steer text is discoverable through the public search
# API, and it was NOT discoverable beforehand. On the old (skip-by-id) code the
# pre-steer assertion stays true forever, so this fails without the fix.
# ---------------------------------------------------------------------------
def test_steer_update_reindexes_fts_search():
    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        try:
            if not db._fts_enabled:
                pytest.skip("FTS5 unavailable in this sqlite build")

            agent = _make_agent(db, Path(tmp) / "hermes-home")
            tool_msg = {
                "role": "tool",
                "content": "tool result",
                "tool_call_id": "c1",
                "tool_name": "web_search",
            }
            messages = [tool_msg]

            # Pre-steer flush: "smaller" lives only in the steer text, which is
            # not in SQLite yet, so FTS cannot find it.
            agent._flush_messages_to_session_db(messages)
            assert db.search_messages("smaller", role_filter=["tool"]) == []

            # /steer mutates the already-flushed dict; the next flush UPDATEs the
            # durable row and the AFTER UPDATE trigger must re-index FTS.
            agent.steer(STEER_TEXT)
            agent._apply_pending_steer_to_tool_results(messages, 1)
            agent._flush_messages_to_session_db(messages)

            hits = db.search_messages("smaller", role_filter=["tool"])
            assert len(hits) == 1, "steer text not searchable after UPDATE — FTS is stale"
            assert hits[0]["session_id"] == SESSION_ID
            assert "smaller" in hits[0]["snippet"]
        finally:
            db.close()

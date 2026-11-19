"""Execute once, reconcile only persistence after a lost worker receipt ACK.

Independently implemented from the scenario suggested by Michał Dziwisz:
https://github.com/NousResearch/hermes-agent/pull/149137#issuecomment-6925948560
This is not a copy of, or validation of, the contributor's linked gist.
The model call is deliberately absent; real executor, agent transcript flush,
worker journal/RPC, WebSocket transport, and SQLite mutation remain in the path.
"""

import json
import time
from types import SimpleNamespace

import pytest

from tests.agent.test_worker_rpc_lost_ack_recovery import _fast_receipts

pytest_plugins = ("tests.agent.test_worker_rpc_lost_ack_recovery",)


@pytest.fixture
def effect_tool(tmp_path):
    """A test-owned registered tool; dispatch is not replaced with a mock."""
    from tools.registry import registry

    name = "worker_receipt_append_probe"
    path = tmp_path / "tool-effect.txt"

    def append_line(args, **kwargs):
        with path.open("a", encoding="utf-8") as stream:
            stream.write("executed\n")
        return json.dumps({"appended": True})

    registry.register(
        name=name,
        toolset="worker-receipt-probe",
        schema={"name": name, "description": "Append one test-owned effect line.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        handler=append_line,
    )
    try:
        yield name, path
    finally:
        registry.deregister(name)


def _execute_effect(store, db, scope, name):
    from agent.tool_executor import execute_tool_calls_sequential
    from run_agent import AIAgent

    call = {"id": "receipt-effect-call", "type": "function",
            "function": {"name": name, "arguments": "{}"}}
    # A real tool-result row needs its assistant-call antecedent. Seed that
    # separately so only the result is written through this worker receipt.
    db.append_messages_batch(scope["session_id"], [
        {"role": "assistant", "content": None, "tool_calls": [call]},
    ])
    messages = db.get_messages_as_conversation(scope["session_id"])
    initial_rows = len(db.get_messages(scope["session_id"]))
    assert initial_rows == 1

    # Construction is real but has no store: setup cannot consume a receipt.
    # The fixture already owns the session row, so bind its worker store before
    # invoking the real executor and the real incremental persistence method.
    agent = AIAgent(
        model="test-model", provider="openai", api_key="unused-test-key",
        base_url="http://127.0.0.1:9/v1", session_id=scope["session_id"],
        enabled_toolsets=["worker-receipt-probe"], quiet_mode=True,
        skip_context_files=True, skip_memory=True, skip_background_review=True,
        save_trajectories=False,
    )
    # Start with a genuinely persisted recent activity heartbeat. Otherwise the
    # executor emits an unrelated session.activity receipt before the result.
    db.touch_session_activity(scope["session_id"])
    agent._session_activity_last_persist_mono = time.monotonic()
    agent._session_db = store
    agent._session_db_created = True
    assistant = SimpleNamespace(tool_calls=[SimpleNamespace(
        id=call["id"], type="function", function=SimpleNamespace(**call["function"]),
    )])
    try:
        execute_tool_calls_sequential(
            agent, assistant, messages, "worker-receipt-probe", finalize=False,
        )
        return messages, bool(getattr(agent, "_incremental_persistence_failed", False))
    finally:
        # Do not introduce end-session writes into the transport assertion.
        agent._session_db = None
        agent.shutdown_memory_provider(None)
        agent.release_clients()


def test_lost_ack_reconciles_tool_result_without_reexecuting_tool(
        receipt_owner, effect_tool, monkeypatch, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC

    state, db, scopes = receipt_owner
    scope = scopes[0]
    # Real agent flushes also read compression lineage through worker receipts.
    # Lose only the tool-result ACK; ancillary reads remain real and acknowledged.
    state.delay_operation = "transcript.append"
    name, effect_path = effect_tool
    _fast_receipts(monkeypatch)
    store = RuntimeSessionStore(WorkerRPC(tmp_path), scope, tmp_path / "effect-outbox")
    try:
        messages, failed = _execute_effect(store, db, scope, name)
        assert not failed
        assert effect_path.read_text(encoding="utf-8").splitlines() == ["executed"]
        requests = [request for request in state.requests
                    if request["params"]["operation"] == "transcript.append"]
        assert len(requests) == 2
        assert [request["method"] for request in requests] == ["worker.persist"] * 2
        # JSON-RPC correlation ids differ on reconnect; the durable operation is identical.
        assert requests[0]["params"] == requests[1]["params"]
        assert requests[0]["params"]["operation"] == "transcript.append"
        sequence = requests[0]["params"]["sequence"]
        assert state.retry_seen[(scope["execution_id"], sequence)].is_set()
        rows = db.get_messages(scope["session_id"])
        assert len(rows) == 2  # one seeded assistant call plus one newly durable result
        results = [row for row in rows if row["role"] == "tool"]
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "receipt-effect-call"
        assert json.loads(results[0]["content"]) == {"appended": True}
        assert len([row for row in messages if row["role"] == "tool"]) == 1
        with db._read_ctx() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM worker_receipts WHERE execution_id=? AND sequence=?",
                (scope["execution_id"], sequence),
            ).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM worker_receipts").fetchone()[0] == len(state.committed)
        assert store.journal["pending"] == []
        assert store.failure is None
    finally:
        store._outbox_owner.close()
        store.rpc.close()


def test_stale_generation_refuses_tool_result_without_retry_or_reexecution(
        receipt_owner, effect_tool, monkeypatch, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC

    state, db, scopes = receipt_owner
    scope = dict(scopes[0], generation=1)  # owner registered generation zero
    name, effect_path = effect_tool
    _fast_receipts(monkeypatch)
    store = RuntimeSessionStore(WorkerRPC(tmp_path), scope, tmp_path / "stale-effect-outbox")
    try:
        messages, failed = _execute_effect(store, db, scope, name)
        assert failed
        assert effect_path.read_text(encoding="utf-8").splitlines() == ["executed"]
        assert len(state.requests) == 1
        assert state.requests[0]["method"] == "worker.persist"
        assert state.requests[0]["params"]["generation"] == 1
        assert store.failure == "stale_generation"
        assert len(store.journal["pending"]) == 1
        assert len([row for row in messages if row["role"] == "tool"]) == 1
        rows = db.get_messages(scope["session_id"])
        assert len(rows) == 1 and rows[0]["role"] == "assistant"
        with db._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM worker_receipts").fetchone()[0] == 0
    finally:
        store._outbox_owner.close()
        store.rpc.close()

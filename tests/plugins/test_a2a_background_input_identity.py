"""Background polling preserves the destination owner's idempotent input identity."""

from gateway import session_a2a
from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter
from tests.waiting import wait_until


def test_background_resend_reuses_owner_input_and_persists_one_user_row(monkeypatch, tmp_path):
    owner_calls = []

    async def owner(home, **params):
        owner_calls.append((home, params))
        return {"status": "terminal", "outcome": "completed",
                "result": {"final_response": "finished"}}

    monkeypatch.setattr(session_a2a, "forward_to_owner", owner)
    monkeypatch.setattr("plugins.platforms.a2a.adapter._profile_home", lambda _profile: str(tmp_path))
    monkeypatch.setattr(protocol, "_conv_dir", lambda: tmp_path / "conversations")
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
        "agents": {"worker": {"profile": "worker", "tenant": "team"}}
    }))
    agent = adapter._agents["worker"]
    params = {"configuration": {"returnImmediately": True},
              "message": protocol.text_message(protocol.ROLE_USER, "run the task")}
    records = []
    for request_id in (1, 2):
        task = adapter._rpc_message_send(request_id, params, "peer", agent=agent)["result"]
        task_id = task["id"]

        def settled():
            record = adapter.tasks.get(task_id)
            return (record is not None and record["state"] == protocol.STATE_COMPLETED
                    and task_id not in adapter._pending)

        assert wait_until(settled, timeout=5)
        records.append(adapter.tasks.get(task_id))

    assert len(owner_calls) == 2
    assert owner_calls[0][1]["context_id"] == owner_calls[1][1]["context_id"]
    input_id = owner_calls[0][1]["input_id"]
    assert input_id == owner_calls[1][1]["input_id"]
    assert all(record["input_id"] == input_id for record in records)
    rows = protocol.load_conversation(owner_calls[0][1]["context_id"], limit=0)
    assert len([row for row in rows if row["role"] == "user"]) == 1
    assert protocol.logged_input(owner_calls[0][1]["context_id"], input_id)

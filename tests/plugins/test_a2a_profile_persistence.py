"""Multiplex A2A writes stay with the adapter owner across HTTP thread hops."""
import json
import threading
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a import protocol


def test_multiplex_adapters_persist_under_own_homes(tmp_path, monkeypatch):
    launch, a, b = (tmp_path / name for name in ("launch", "a", "b"))
    for home in (launch, a, b):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.delenv("A2A_PORT", raising=False)
    adapters = []
    for home in (a, b, a):
        token = set_hermes_home_override(home)
        try:
            adapters.append(A2AAdapter(SimpleNamespace(extra={})))
        finally:
            reset_hermes_home_override(token)

    def incoming(adapter, context):
        params = {"message": protocol.text_message(protocol.ROLE_USER, context, context_id=context)}
        adapter._prepare_task(params, "authenticated-peer")
        adapter._record_outcome("completed-task", context, "authenticated-peer", protocol.STATE_COMPLETED, "ACK")

    # New OS threads have no caller ContextVar: they see the multiplex launch home.
    for adapter, context in zip(adapters, ("to-a", "to-b", "back-to-a")):
        thread = threading.Thread(target=incoming, args=(adapter, context))
        thread.start()
        thread.join()
    for home, own, foreign in ((a, ("to-a", "back-to-a"), "to-b"), (b, ("to-b",), "to-a")):
        for context in own:
            rows = [json.loads(line) for line in (home / "a2a_conversations" / f"{context}.jsonl").read_text().splitlines()]
            assert [row["role"] for row in rows] == ["user", "agent"]
            assert [row["text"] for row in rows] == [context, "ACK"]
        assert not (home / "a2a_conversations" / f"{foreign}.jsonl").exists()
        audit = [json.loads(line) for line in (home / "a2a_audit.jsonl").read_text().splitlines()]
        assert len(audit) == 2 * len(own)
    assert not (launch / "a2a_conversations").exists()
    assert not (launch / "a2a_audit.jsonl").exists()

"""A2A forwarded tasks: a resend stays a retry after the process-local id memory forgot it
(andrexibiza 5480715949 P2c)."""
from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def _adapter():
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"agents": {"dev": {"profile": "dev", "tenant": "dev"}}}))
    return adapter, adapter._agents["dev"]


def test_a_resend_evicted_from_the_process_memory_is_still_a_retry(monkeypatch):
    monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "2")
    adapter, agent = _adapter()
    adapter._MAX_FORWARDED_INPUTS = 1
    seen = []

    def forward(agent_arg, peer, context_id, framed_text, *, input_id):
        seen.append(input_id)
        return "owner reply", protocol.STATE_COMPLETED

    adapter._forward_to_profile = forward  # type: ignore[method-assign]
    first = protocol.text_message(protocol.ROLE_USER, "first", context_id="ctx-evict")
    other = protocol.text_message(protocol.ROLE_USER, "other", context_id="ctx-elsewhere")
    adapter._prepare_task({"tenant": "dev", "message": dict(first)}, "peer-x", agent=agent)
    adapter._prepare_task({"tenant": "dev", "message": other}, "peer-x", agent=agent)  # evicts "first"
    for _ in range(3):
        task, _ = adapter._prepare_task({"tenant": "dev", "message": dict(first)}, "peer-x", agent=agent)
        assert task["status"]["state"] == protocol.STATE_COMPLETED
    assert len({seen[0], *seen[2:]}) == 1
    log = protocol.load_conversation("ctx-evict", limit=0)
    assert [entry["role"] for entry in log].count("user") == 1
    # The resends never spent the context's turn budget: the next distinct message is turn 2.
    assert adapter._turns.track("ctx-evict") == 2

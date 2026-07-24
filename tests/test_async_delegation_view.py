"""Backend tests for the Async Delegation View (docked agents panel + steering).

Covers the two new gateway RPCs — ``delegation.async_list`` (read projection
of the async registry) and ``subagent.send`` (live steering) — plus the
``send_to_subagent`` helper they lean on.
"""

import tools.async_delegation as async_delegation
import tools.delegate_tool as delegate_tool
from tui_gateway import server


class _FakeSteerAgent:
    """Minimal stand-in for a live child ``AIAgent`` in the registry.

    Records every ``steer`` call so tests can assert exactly-once delivery
    and role-legal drain behaviour without spinning up a real agent loop.
    """

    def __init__(self, accept: bool = True):
        self.accept = accept
        self.steers: list[str] = []

    def steer(self, text: str) -> bool:
        self.steers.append(text)
        return self.accept


def _clear_async_records():
    with async_delegation._records_lock:
        async_delegation._records.clear()


def _register_async(delegation_id: str, status: str = "running"):
    with async_delegation._records_lock:
        async_delegation._records[delegation_id] = {
            "delegation_id": delegation_id,
            "goal": "patch token-bucket refill race",
            "role": "fixer",
            "model": "opus-4.8",
            "status": status,
            "depth": 1,
            "dispatched_at": 1.0,
            "completed_at": None if status == "running" else 2.0,
            # interrupt_fn must be stripped by list_async_delegations — assert it.
            "interrupt_fn": lambda: None,
        }


# ── delegation.async_list ────────────────────────────────────────────────


def test_async_list_shape_and_running_count():
    _clear_async_records()
    try:
        _register_async("d-run", "running")
        _register_async("d-done", "completed")

        resp = server._methods["delegation.async_list"]("r1", {})
        result = resp["result"]

        assert result["running"] == 1  # only the running record counts
        assert len(result["delegations"]) == 2
        # interrupt_fn (non-serialisable) must never leak into the payload.
        for d in result["delegations"]:
            assert "interrupt_fn" not in d
        goals = {d["delegation_id"]: d["goal"] for d in result["delegations"]}
        assert goals["d-run"] == "patch token-bucket refill race"
    finally:
        _clear_async_records()


def test_async_list_empty_registry():
    _clear_async_records()
    resp = server._methods["delegation.async_list"]("r1", {})
    result = resp["result"]
    assert result["running"] == 0
    assert result["delegations"] == []


# ── send_to_subagent helper ──────────────────────────────────────────────


def test_send_to_subagent_delivers_once():
    agent = _FakeSteerAgent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        ok = delegate_tool.send_to_subagent("b7c2", "prefer a sliding window")
        assert ok is True
        # Exactly one user turn queued — no double-append.
        assert agent.steers == ["prefer a sliding window"]
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_send_to_subagent_unknown_id_returns_false():
    assert delegate_tool.send_to_subagent("does-not-exist", "hi") is False


def test_send_to_subagent_empty_text_is_rejected():
    agent = _FakeSteerAgent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        assert delegate_tool.send_to_subagent("b7c2", "   ") is False
        assert agent.steers == []  # never reached the child
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_send_to_subagent_agent_without_steer_returns_false():
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": object()})
    try:
        assert delegate_tool.send_to_subagent("b7c2", "hi") is False
    finally:
        delegate_tool._unregister_subagent("b7c2")


# ── subagent.send RPC ────────────────────────────────────────────────────


def test_subagent_send_rpc_delivers_to_live_child():
    agent = _FakeSteerAgent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        resp = server._methods["subagent.send"](
            "r1", {"subagent_id": "b7c2", "text": "switch approach"}
        )
        assert resp["result"]["delivered"] is True
        assert resp["result"]["subagent_id"] == "b7c2"
        assert agent.steers == ["switch approach"]
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_subagent_send_rpc_dead_id_reports_not_delivered():
    resp = server._methods["subagent.send"](
        "r1", {"subagent_id": "ghost", "text": "hi"}
    )
    assert resp["result"]["delivered"] is False


def test_subagent_send_rpc_requires_id_and_text():
    missing_text = server._methods["subagent.send"]("r1", {"subagent_id": "b7c2"})
    assert "error" in missing_text
    missing_id = server._methods["subagent.send"]("r1", {"text": "hi"})
    assert "error" in missing_id
    blank_text = server._methods["subagent.send"](
        "r1", {"subagent_id": "b7c2", "text": "   "}
    )
    assert "error" in blank_text


# ── Flow / integration: steering a REAL AIAgent through the registry ──────
#
# These use a bare AIAgent (object.__new__, no __init__) — steer/_drain fall
# back to the lock-free path documented for test stubs, exercising the same
# _pending_steer slot the live conversation loop drains.


def _bare_agent():
    from run_agent import AIAgent

    return object.__new__(AIAgent)


def test_steer_reaches_child_via_registry_and_drains_exact_text():
    agent = _bare_agent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        assert delegate_tool.send_to_subagent("b7c2", "prefer a sliding window") is True
        # The loop drains this exact text at its next iteration boundary.
        assert agent._drain_pending_steer() == "prefer a sliding window"
        # Exactly once — a second drain is empty.
        assert agent._drain_pending_steer() is None
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_multiple_steers_concatenate_in_order():
    agent = _bare_agent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        delegate_tool.send_to_subagent("b7c2", "first")
        delegate_tool.send_to_subagent("b7c2", "second")
        assert agent._drain_pending_steer() == "first\nsecond"
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_send_after_unregister_returns_false():
    """A child that finished (unregistered) can no longer be steered."""
    agent = _bare_agent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    delegate_tool._unregister_subagent("b7c2")
    assert delegate_tool.send_to_subagent("b7c2", "too late") is False
    assert agent._drain_pending_steer() is None


def test_concurrent_sends_all_deliver():
    """Thread-safety: N concurrent steers all land (no lost update)."""
    import threading

    agent = _bare_agent()
    # Give it the real lock so the concurrent path (not the stub path) runs.
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        n = 50
        barrier = threading.Barrier(n)

        def _fire(i: int):
            barrier.wait()
            delegate_tool.send_to_subagent("b7c2", f"m{i}")

        threads = [threading.Thread(target=_fire, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        drained = agent._drain_pending_steer() or ""
        parts = [p for p in drained.split("\n") if p]
        assert len(parts) == n  # every steer survived
        assert {p for p in parts} == {f"m{i}" for i in range(n)}
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_steer_injection_preserves_role_alternation():
    """The loop appends the steer to the LAST tool message, never inserting a
    fresh user turn mid-tool — the invariant the async design was built on.

    Replicates the drain+inject from agent/conversation_loop.py to assert the
    resulting message sequence stays role-legal (…tool→assistant…, no user
    spliced between a tool result and the next assistant turn)."""
    from agent.prompt_builder import format_steer_marker

    agent = _bare_agent()
    delegate_tool._register_subagent({"subagent_id": "b7c2", "agent": agent})
    try:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ran ok"},
        ]
        delegate_tool.send_to_subagent("b7c2", "switch approach")

        steer = agent._drain_pending_steer()
        assert steer == "switch approach"
        # Inject exactly as the loop does: onto the last tool message.
        last_tool = next(m for m in reversed(messages) if m["role"] == "tool")
        last_tool["content"] += format_steer_marker(steer)

        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool"]  # no new user turn
        assert "switch approach" in messages[-1]["content"]
        # No two adjacent same-role messages (alternation intact).
        assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    finally:
        delegate_tool._unregister_subagent("b7c2")


def test_async_list_reflects_status_transition():
    _clear_async_records()
    try:
        _register_async("d1", "running")
        assert server._methods["delegation.async_list"]("r", {})["result"]["running"] == 1

        # Flip to completed — active_count drops, record still listed.
        with async_delegation._records_lock:
            async_delegation._records["d1"]["status"] = "completed"
            async_delegation._records["d1"]["completed_at"] = 2.0

        result = server._methods["delegation.async_list"]("r", {})["result"]
        assert result["running"] == 0
        assert len(result["delegations"]) == 1
        assert result["delegations"][0]["status"] == "completed"
    finally:
        _clear_async_records()

"""Contract for the synchronous facade over the async Claude Agent SDK.

The facade owns one event-loop thread per session, so these tests care about
threading behavior: messages must land on the caller's thread, the stream must
be drained past ``ResultMessage``, an interrupt must leave the session usable,
and teardown must leave no live thread.

``claude-agent-sdk`` is an optional extra. Every test here injects a fake
client through ``client_factory``, so the suite is green with or without the
real package installed — the facade only imports the SDK when it has to build
its own client.
"""

import asyncio
import threading
import time

import pytest

from agent.transports.claude_agent_session import (
    ClaudeAgentSession,
    ClaudeAgentSessionError,
    is_result_message,
)


# ---------------------------------------------------------------------------
# Fake SDK client
# ---------------------------------------------------------------------------


class _Msg:
    """Base for the fake message types; the facade dispatches on class name."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.tag!r})"


class AssistantMessage(_Msg):
    pass


class ResultMessage(_Msg):
    pass


class FakeClient:
    """Scriptable stand-in for ``ClaudeSDKClient``.

    ``scripts`` is a list of per-query message lists. Each entry may hold a
    float, which the fake awaits before emitting the next message — that is how
    a test makes trailing frames arrive *after* the ``ResultMessage``.
    """

    def __init__(self, *, options=None, scripts=None):
        self.options = options
        self._scripts = list(scripts or [])
        self.queries = []
        self.connected = False
        self.disconnected = False
        self.interrupts = 0
        self.models = []
        self.loops = set()
        self._pending = []

    async def connect(self, prompt=None):
        self.connected = True
        self.loops.add(asyncio.get_running_loop())

    async def disconnect(self):
        self.disconnected = True

    async def query(self, prompt, session_id="default"):
        self.loops.add(asyncio.get_running_loop())
        self.queries.append(prompt)
        self._pending = list(self._scripts.pop(0)) if self._scripts else []

    async def interrupt(self):
        self.interrupts += 1
        # A real interrupt still terminates the response with a result.
        self._pending = [ResultMessage("interrupted")]

    async def set_model(self, model=None):
        self.models.append(model)

    async def receive_messages(self):
        while True:
            if not self._pending:
                # Idle forever, like the real stream between responses.
                await asyncio.sleep(0.01)
                continue
            item = self._pending.pop(0)
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield item


def _session(scripts, **kwargs):
    client = FakeClient(scripts=scripts)
    kwargs.setdefault("trailing_drain_seconds", 0.2)
    session = ClaudeAgentSession(
        options_factory=lambda: {"marker": "opts"},
        client_factory=lambda **kw: client,
        **kwargs,
    )
    return session, client


def _collect(session, prompt, **kwargs):
    seen = []
    session.run_turn(prompt, on_message=seen.append, **kwargs)
    return seen


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def test_result_message_is_recognised_by_class_name():
    assert is_result_message(ResultMessage()) is True
    assert is_result_message(AssistantMessage()) is False


def test_start_is_idempotent_and_connects_exactly_once():
    session, client = _session([[ResultMessage()]])
    try:
        session.ensure_started()
        session.ensure_started()
        assert client.connected is True
        assert session.started is True
    finally:
        session.close()


def test_client_lives_on_one_loop_that_the_session_created():
    session, client = _session([[ResultMessage("a")], [ResultMessage("b")]])
    try:
        _collect(session, "one")
        _collect(session, "two")
        # Every await the client saw ran on the same, session-owned loop.
        assert len(client.loops) == 1
        assert asyncio.get_event_loop_policy() is not None
    finally:
        session.close()


def test_module_imports_without_the_optional_sdk():
    import agent.transports.claude_agent_session as mod

    assert "claude_agent_sdk" not in mod.__dict__


# ---------------------------------------------------------------------------
# Turn semantics
# ---------------------------------------------------------------------------


def test_messages_are_delivered_in_order_on_the_calling_thread():
    session, _client = _session(
        [[AssistantMessage("1"), AssistantMessage("2"), ResultMessage("3")]]
    )
    threads = []

    def _on_message(message):
        threads.append(threading.current_thread())

    try:
        seen = []
        session.run_turn(
            "hi",
            on_message=lambda m: (seen.append(m.tag), _on_message(m)),
        )
    finally:
        session.close()

    assert [m for m in seen] == ["1", "2", "3"]
    assert set(threads) == {threading.current_thread()}


def test_trailing_events_after_the_result_message_are_not_truncated():
    """Stopping at ResultMessage drops frames the CLI flushes right after it."""
    session, _client = _session(
        [
            [
                AssistantMessage("body"),
                ResultMessage("done"),
                0.02,
                AssistantMessage("trailing"),
            ]
        ]
    )
    try:
        seen = _collect(session, "hi")
    finally:
        session.close()

    assert [m.tag for m in seen] == ["body", "done", "trailing"]


def test_the_turn_ends_once_the_stream_goes_quiet_after_the_result():
    session, _client = _session([[ResultMessage("done")]])
    try:
        started = time.monotonic()
        seen = _collect(session, "hi")
        elapsed = time.monotonic() - started
    finally:
        session.close()

    assert [m.tag for m in seen] == ["done"]
    # Bounded by the trailing-drain grace, not by the turn deadline.
    assert elapsed < 5.0


def test_consecutive_turns_reuse_the_same_client():
    session, client = _session([[ResultMessage("a")], [ResultMessage("b")]])
    try:
        _collect(session, "first")
        _collect(session, "second")
    finally:
        session.close()

    assert client.queries == ["first", "second"]
    assert session.turn_count == 2


def test_an_sdk_error_propagates_to_the_caller():
    class Boom(FakeClient):
        async def query(self, prompt, session_id="default"):
            raise RuntimeError("cli exploded")

    client = Boom()
    session = ClaudeAgentSession(
        options_factory=dict, client_factory=lambda **kw: client
    )
    try:
        with pytest.raises(RuntimeError, match="cli exploded"):
            _collect(session, "hi")
    finally:
        session.close()


def test_a_turn_that_never_completes_raises_timeout_rather_than_hanging():
    session, _client = _session([[AssistantMessage("partial")]])
    try:
        with pytest.raises(TimeoutError):
            _collect(session, "hi", timeout=0.4)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


def test_interrupt_then_drain_leaves_the_session_usable():
    session, client = _session(
        [
            # First turn stalls; the interrupt below supplies its terminator.
            [AssistantMessage("working"), 5.0, AssistantMessage("never")],
            [ResultMessage("second-turn")],
        ]
    )
    try:
        session.ensure_started()

        def _interrupt_soon():
            time.sleep(0.15)
            session.request_interrupt()

        threading.Thread(target=_interrupt_soon, daemon=True).start()
        first = _collect(session, "long task", timeout=10.0)

        # The interrupted response was drained to its terminal result before
        # the next query went out — skipping that drain wedges the session.
        assert client.interrupts == 1
        assert is_result_message(first[-1])

        second = _collect(session, "follow-up", timeout=10.0)
        assert [m.tag for m in second] == ["second-turn"]
        assert client.queries == ["long task", "follow-up"]
    finally:
        session.close()


def test_interrupt_before_start_is_a_no_op():
    session, _client = _session([[ResultMessage()]])
    assert session.request_interrupt() is False


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def _session_threads():
    return [t for t in threading.enumerate() if t.name == "hermes-claude-agent-sdk"]


def test_close_is_idempotent_and_leaves_no_live_thread():
    before = len(_session_threads())
    session, client = _session([[ResultMessage()]])
    _collect(session, "hi")
    assert len(_session_threads()) == before + 1

    session.close()
    session.close()
    session.close()

    assert client.disconnected is True
    assert session.closed is True
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(_session_threads()) > before:
        time.sleep(0.02)
    assert len(_session_threads()) == before


def test_a_closed_session_refuses_to_restart():
    session, _client = _session([[ResultMessage()]])
    session.close()
    with pytest.raises(ClaudeAgentSessionError):
        session.ensure_started()


def test_close_without_ever_starting_is_safe():
    session, client = _session([[ResultMessage()]])
    session.close()
    assert client.connected is False
    assert client.disconnected is False


def test_context_manager_starts_and_closes():
    session, client = _session([[ResultMessage("x")]])
    with session as live:
        assert live.started is True
    assert client.disconnected is True
    assert session.closed is True


# ---------------------------------------------------------------------------
# Session id (PR5 seam)
# ---------------------------------------------------------------------------


def test_session_id_is_recorded_for_resume():
    session, _client = _session([[ResultMessage()]])
    try:
        assert session.session_id is None
        session.note_session_id("sdk-session-abc")
        assert session.session_id == "sdk-session-abc"
        session.note_session_id(None)
        assert session.session_id == "sdk-session-abc"
    finally:
        session.close()


def test_set_model_round_trips_to_the_client():
    session, client = _session([[ResultMessage()]])
    try:
        session.ensure_started()
        assert session.set_model("claude-opus-4-5") is True
        assert client.models == ["claude-opus-4-5"]
    finally:
        session.close()

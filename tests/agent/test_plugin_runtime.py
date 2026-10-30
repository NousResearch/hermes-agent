"""Unit coverage for the plugin control bridge (`hermes_cli.plugin_runtime`)."""

from __future__ import annotations

import pytest

from hermes_cli import plugin_runtime


class _Agent:
    def __init__(self, session_id: str = "session-1"):
        self.session_id = session_id
        self.platform = "cli"
        self._current_turn_id = "turn-1"
        self._resolved_is_coding = True
        self.steered: list[tuple[str, str]] = []
        self.redirected: list[str] = []
        self.interrupts: list[str] = []

    def steer(self, text: str, *, kind: str = "user") -> bool:
        self.steered.append((text, kind))
        return True

    def redirect(self, text: str) -> bool:
        self.redirected.append(text)
        return True

    def interrupt(self, *, tool_reason: str = "") -> None:
        self.interrupts.append(tool_reason)


class _LegacyAgent(_Agent):
    """A host whose ``steer`` predates the ``kind`` keyword."""

    def steer(self, text: str) -> bool:
        self.steered.append((text, "user"))
        return True


class _PreserveAgent(_Agent):
    """A minimal agent with the transcript + flush surface ``preserve`` needs."""

    def __init__(self, session_id: str = "session-1"):
        super().__init__(session_id)
        self.messages: list[dict] = [{"role": "assistant", "content": "done"}]
        self.flushed: list[list[dict]] = []
        self.plugin_message_callback = None

    def _flush_messages_to_session_db(self, messages):
        self.flushed.append(list(messages))


@pytest.fixture(autouse=True)
def _clean_runtime():
    plugin_runtime.clear_session("session-1")
    plugin_runtime.clear_session("session-2")
    yield
    plugin_runtime.clear_session("session-1")
    plugin_runtime.clear_session("session-2")


def test_unbound_session_refuses_to_pretend_it_delivered():
    result = plugin_runtime.steer("session-1", "look at this")
    assert result == {"ok": False, "mode": "", "reason": "no_live_turn"}
    assert plugin_runtime.turn_state("session-1")["live"] is False


def test_bind_publishes_only_the_running_turn():
    agent = _Agent()
    assert plugin_runtime.bind_turn(agent) is True
    state = plugin_runtime.turn_state("session-1")
    assert state["live"] is True
    assert state["turn_id"] == "turn-1"
    assert state["coding"] is True
    agent._current_turn_id = "turn-2"
    assert plugin_runtime.turn_state("session-1")["turn_id"] == "turn-2"
    plugin_runtime.unbind_turn(agent)
    assert plugin_runtime.live_turn("session-1") is False


def test_steer_marks_advice_and_never_uses_the_user_correction_path():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    result = plugin_runtime.steer("session-1", "stop and re-read the diff")
    assert result["ok"] is True and result["mode"] == "steer"
    assert agent.steered == [("stop and re-read the diff", "adviser")]
    # redirect would persist a "user correction"; plugin advice must not borrow that.
    assert agent.redirected == []

    legacy = _LegacyAgent(session_id="session-2")
    plugin_runtime.bind_turn(legacy)
    result = plugin_runtime.steer("session-2", "fallback text")
    assert result["ok"] is True and legacy.steered == [("fallback text", "user")]


def test_one_continuation_at_a_time_and_its_status_is_observable():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    first = plugin_runtime.request_continuation("session-1", "first advice", source="minir")
    assert first["ok"] is True
    second = plugin_runtime.request_continuation("session-1", "second advice", source="minir")
    assert second == {"ok": False, "token": "", "reason": "already_pending"}
    assert plugin_runtime.continuation_status("session-1", first["token"]) == "pending"

    taken = plugin_runtime.take_continuation("session-1")
    assert taken["text"] == "first advice"
    assert plugin_runtime.continuation_status("session-1", first["token"]) == "taken"
    assert plugin_runtime.take_continuation("session-1") is None


def test_unconsumed_continuation_expires_at_turn_end():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    queued = plugin_runtime.request_continuation("session-1", "late advice", source="minir")
    plugin_runtime.unbind_turn(agent)
    assert plugin_runtime.continuation_status("session-1", queued["token"]) == "expired"


def test_discard_marks_the_token_expired_not_taken():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    queued = plugin_runtime.request_continuation("session-1", "over budget", source="minir")
    assert plugin_runtime.discard_continuation("session-1")["token"] == queued["token"]
    assert plugin_runtime.continuation_status("session-1", queued["token"]) == "expired"
    assert plugin_runtime.take_continuation("session-1") is None


def test_advisory_row_does_not_claim_user_authority():
    from agent.prompt_builder import STEER_DISPLAY_KIND, advisory_user_row, steer_row

    advice = advisory_user_row("[adviser — advisory only]\nre-read the diff")
    assert advice["role"] == "user" and advice["display_kind"] == "adviser"
    assert "OUT-OF-BAND USER MESSAGE" not in advice["content"]

    operator = steer_row("hello")
    assert operator["display_kind"] == STEER_DISPLAY_KIND
    assert "OUT-OF-BAND USER MESSAGE" in operator["content"]


def test_text_is_bounded():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    result = plugin_runtime.steer("session-1", "x" * (plugin_runtime.MAX_TEXT_CHARS + 100))
    assert result["ok"] is True
    assert len(agent.steered[0][0]) <= plugin_runtime.MAX_TEXT_CHARS


def test_interrupt_queues_advice_before_stopping_the_tool_batch():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    result = plugin_runtime.interrupt("session-1", "stop, the premise is wrong", source="minir")

    assert result["ok"] is True and result["mode"] == "interrupt"
    assert agent.steered == [("stop, the premise is wrong", "adviser")]
    assert agent.interrupts == ["minir"]


def test_post_turn_is_merged_into_the_surface_contract():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    first = plugin_runtime.queue_post_turn("session-1", "[adviser] finish the test", source="minir")
    assert first["ok"] is True
    assert plugin_runtime.queue_post_turn("session-1", "second")["reason"] == "already_pending"

    result = {"pending_steer": "leftover operator text"}
    plugin_runtime.attach_post_turn(agent, result)
    assert result["pending_steer"] == "leftover operator text\n[adviser] finish the test"
    assert result["pending_steer_kind"] == "adviser"


def test_session_flags_are_readable_only_while_live():
    agent = _Agent()
    plugin_runtime.bind_turn(agent)
    assert plugin_runtime.set_flag("session-1", "plan_mode", True) is True
    assert plugin_runtime.turn_state("session-1")["flags"] == {"plan_mode": True}

    plugin_runtime.unbind_turn(agent)
    assert plugin_runtime.set_flag("session-1", "plan_mode", False) is False


def test_preserve_writer_survives_the_turn_and_appends_a_card():
    agent = _PreserveAgent()
    seen: list[dict] = []
    agent.plugin_message_callback = seen.append
    plugin_runtime.bind_turn(agent)

    # Captured while the turn is live; the review that preserves runs after it.
    handle = plugin_runtime.preserve_handle("session-1")
    assert callable(handle)
    plugin_runtime.unbind_turn(agent)

    result = handle("[adviser] keep this card", {"preserved": True, "severity": "concern"})

    assert result["ok"] is True
    row = agent.messages[-1]
    assert row["role"] == "user" and row["display_kind"] == "adviser"
    assert row["display_metadata"] == {"preserved": True, "severity": "concern"}
    assert agent.flushed and agent.flushed[-1][-1] is row
    assert seen and seen[-1]["plugin_message_id"] == row["plugin_message_id"]
    assert "keep this card" in seen[-1]["content"]


def test_preserve_one_shot_refuses_after_the_turn():
    agent = _PreserveAgent()
    assert plugin_runtime.preserve("session-1", "too late") == {
        "ok": False, "reason": "no_live_turn"}
    plugin_runtime.bind_turn(agent)
    assert plugin_runtime.preserve("session-1", "on time")["ok"] is True
    assert agent.messages[-1]["display_kind"] == "adviser"

"""The native adviser continuation is drained by the stop gate before plugin continuations."""

from __future__ import annotations

from agent.turn_stop_gates import _adviser_continuation


class _Adviser:
    def __init__(self, pending):
        self.pending = pending

    def take_continuation(self):
        pending, self.pending = self.pending, None
        return pending


class _Agent:
    def __init__(self, adviser):
        self.adviser = adviser


def test_pending_adviser_continuation_is_returned_once():
    agent = _Agent(_Adviser({"text": "[adviser] fix it", "source": "reviewer"}))

    pending = _adviser_continuation(agent)
    assert pending is not None and pending["source"] == "reviewer"
    assert _adviser_continuation(agent) is None


def test_agent_without_adviser_is_a_noop():
    assert _adviser_continuation(_Agent(None)) is None


def test_stop_gate_captures_terminal_and_delivers_the_continuation(monkeypatch):
    import agent.turn_stop_gates as gates

    for name, fake in {
        "_late_lsp_verdicts": lambda agent: None,
        "_verify_on_stop_nudge": lambda agent: None,
        "_pre_verify_nudge": lambda agent, response, attempt: None,
        "_kanban_stop_nudge": lambda agent, messages: None,
    }.items():
        monkeypatch.setattr(gates, name, fake)

    class _Adviser:
        def __init__(self):
            self.seen: list[tuple[list, bool]] = []
            self.pending = {"text": "[adviser — advisory only]\n- [blocker] x\n[end adviser]",
                            "source": "reviewer"}

        def on_step(self, messages, *, terminal):
            self.seen.append((list(messages), terminal))

        def take_continuation(self):
            pending, self.pending = self.pending, None
            return pending

    class _Agent:
        def __init__(self):
            self.session_id = "s"
            self._turn_file_mutation_paths = set()
            self._session_messages: list = []
            self.plugin_message_callback = None
            self.quiet_mode = True
            self.adviser = _Adviser()

        def _interim_content_was_streamed(self, text):
            return False

        def _emit_interim_assistant_message(self, message):
            return None

        def _flush_messages_to_session_db(self, messages, history=None):
            return None

    agent = _Agent()
    messages = [{"role": "user", "content": "go"}]
    final_msg = {"role": "assistant", "content": "done"}

    verdict = gates.apply_stop_gates(
        agent, final_msg, final_response="done", messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=None,
    )

    assert verdict.continue_turn is True
    assert agent.adviser.seen and agent.adviser.seen[0][1] is True
    assert agent.adviser.seen[0][0][-1] == {"role": "assistant", "content": "done"}
    assert messages[-1]["display_kind"] == "adviser"

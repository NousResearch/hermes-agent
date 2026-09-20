"""Vertical contracts for the Antigravity session runtime."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from run_agent import AIAgent


class _Projector:
    def __init__(self):
        self.projected_messages = []
        self.final_text = ""

    def feed(self, event):
        if event["type"] == "assistant":
            self.projected_messages.append({"role": "assistant", "content": event["text"]})
            self.final_text = event["text"]


class _Client:
    def __init__(self):
        self.calls = []
        self.cancelled = 0
        self.closed = 0

    def run_turn(self, prompt, conversation_id=None, event_callback=None, cancel_event=None):
        self.calls.append((prompt, conversation_id, cancel_event))
        event_callback({"type": "assistant", "text": "runtime answer"})
        return SimpleNamespace(final_text="", conversation_id="ag-conversation", events=[], usage={"total_tokens": 7}, interrupted=False, error=None)

    def cancel(self):
        self.cancelled += 1

    def close(self):
        self.closed += 1


def _agent(**kwargs):
    return AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai",
                   api_mode="antigravity_runtime", quiet_mode=True, skip_context_files=True,
                   skip_memory=True, skip_background_review=True, **kwargs)


def test_aiagent_runs_antigravity_turn_reuses_conversation_and_closes_once():
    client = _Client()
    agent = _agent(cwd="/tmp/ag-cwd")
    agent._antigravity_client_factory = lambda **_kw: client
    agent._antigravity_projector_factory = _Projector

    first = agent.run_conversation("first")
    second = agent.run_conversation("second")

    assert first["final_response"] == "runtime answer"
    assert first["completed"] is True
    assert first["agent_persisted"] is True
    assert second["antigravity_conversation_id"] == "ag-conversation"
    assert client.calls == [("first", None, client.calls[0][2]), ("second", "ag-conversation", client.calls[1][2])]
    assert agent._antigravity_session.cwd == "/tmp/ag-cwd"

    agent.close()
    agent.close()
    assert client.closed == 1
    assert agent._antigravity_session is None


def test_cold_agent_restores_antigravity_conversation_from_transcript_sidecar():
    client = _Client()
    agent = _agent(cwd="/tmp/ag-cwd")
    agent._antigravity_client_factory = lambda **_kw: client
    agent._antigravity_projector_factory = _Projector

    result = agent.run_conversation(
        "continue",
        conversation_history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "stored", "_antigravity": {"conversation_id": "ag-restored"}},
        ],
    )

    assert result["completed"] is True
    assert client.calls[0][1] == "ag-restored"
    agent.close()


def test_cumulative_usage_is_recorded_as_per_turn_delta():
    from agent.antigravity_runtime import _record_usage

    agent = SimpleNamespace(session_api_calls=0, session_prompt_tokens=0,
                            session_completion_tokens=0, session_total_tokens=0)
    first = _record_usage(agent, {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
    second = _record_usage(agent, {"input_tokens": 150, "output_tokens": 35, "total_tokens": 185})

    assert first["prompt_tokens"] == 100
    assert second["prompt_tokens"] == 50
    assert second["completion_tokens"] == 15
    assert agent.session_total_tokens == 185
    assert agent.session_prompt_tokens == 150
    assert agent.session_completion_tokens == 35


def test_failed_db_flush_is_not_reported_as_agent_persisted():
    client = _Client()
    agent = _agent()
    setattr(agent, "_antigravity_client_factory", lambda **_kw: client)
    setattr(agent, "_antigravity_projector_factory", _Projector)
    setattr(agent, "_session_db", object())
    setattr(agent, "_flush_messages_to_session_db", lambda _messages: False)

    result = agent.run_conversation("first")

    assert result["agent_persisted"] is False
    agent.close()


def test_unknown_approval_is_reported_as_error_and_preserves_projected_rows():
    from agent.transports.antigravity_cli import AntigravityCancelled
    from agent.transports.antigravity_session import AntigravitySession

    class ApprovalClient:
        def __init__(self):
            self.cancelled = 0

        def run_turn(self, _prompt, **kwargs):
            callback = kwargs["event_callback"]
            callback({"event": "step_update", "step_update": {
                "step_index": 1, "state": "DONE", "step_type": "tool", "tool_name": "view_file",
                "tool_info": {"parameters": {"AbsolutePath": "/tmp/a"}, "result": "ok"},
            }})
            callback({"event": "approval_requested", "approval": {"command": "rm -rf build"}})
            raise AntigravityCancelled("cancelled by fail-safe")

        def cancel(self):
            self.cancelled += 1

    client = ApprovalClient()
    result = AntigravitySession(cwd="/tmp", client=client).run_turn("x")

    assert result.interrupted is False
    assert result.should_retire is True
    assert result.error is not None
    assert "approval/permission" in result.error
    assert len(result.projected_messages) == 2
    assert client.cancelled == 1


def test_interrupt_cancels_active_transport_and_exits_busy_state():
    from agent.transports.antigravity_cli import AntigravityCancelled

    class BlockingClient(_Client):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.cancelled_event = threading.Event()

        def run_turn(self, prompt, conversation_id=None, event_callback=None, cancel_event=None):
            self.started.set()
            self.cancelled_event.wait(timeout=5)
            raise AntigravityCancelled("stopped")

        def cancel(self):
            super().cancel()
            self.cancelled_event.set()

    client = BlockingClient()
    agent = _agent()
    setattr(agent, "_antigravity_client_factory", lambda **_kw: client)
    result_holder = {}

    worker = threading.Thread(target=lambda: result_holder.update(agent.run_conversation("long task")))
    worker.start()
    assert client.started.wait(timeout=2)
    agent.interrupt()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert client.cancelled == 1
    assert result_holder["interrupted"] is True
    assert result_holder["completed"] is False
    agent.close()


def test_interrupt_while_idle_does_not_poison_next_turn_and_error_retires_session():
    client = _Client()
    agent = _agent()
    agent._antigravity_client_factory = lambda **_kw: client
    agent._antigravity_projector_factory = _Projector
    from agent.antigravity_runtime import _ensure_antigravity_session, run_antigravity_turn

    _ensure_antigravity_session(agent)
    agent._antigravity_session._get_client()  # active transport exists before a cross-thread stop
    agent.interrupt()
    assert client.cancelled == 0
    assert run_antigravity_turn(agent, user_message="x", original_user_message="x", messages=[], effective_task_id="t")["interrupted"] is True

    class BrokenClient(_Client):
        def run_turn(self, *args, **kwargs):
            return SimpleNamespace(final_text="", conversation_id=None, events=[], usage=None, interrupted=False, error="broken")

    broken = BrokenClient()
    agent.clear_interrupt()
    agent._close_antigravity_session()
    agent._antigravity_client_factory = lambda **_kw: broken
    result = run_antigravity_turn(agent, user_message="x", original_user_message="x", messages=[], effective_task_id="t")
    assert result["error"] == "broken"
    assert agent._antigravity_session is None
    assert broken.closed == 1

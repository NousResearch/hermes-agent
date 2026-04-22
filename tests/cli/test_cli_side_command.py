from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.cli.test_cli_new_session import _make_cli


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


class _FakeSideAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        messages = list(conversation_history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "side answer"},
        ]
        return {
            "final_response": "side answer",
            "messages": messages,
        }


def _prepare_cli():
    cli = _make_cli()
    cli.console = MagicMock()
    cli.conversation_history = [
        {"role": "user", "content": "main question"},
        {"role": "assistant", "content": "main answer"},
    ]
    cli._invalidate = MagicMock()
    return cli


def test_side_command_unavailable_before_session_starts(monkeypatch):
    cli = _make_cli()
    cli.console = MagicMock()
    monkeypatch.setattr("cli._cprint", lambda *args, **kwargs: None)

    result = cli.process_command("/side")

    assert result is True
    assert cli._side_state is None


def test_side_command_opens_ephemeral_side_state(monkeypatch):
    cli = _prepare_cli()
    monkeypatch.setattr("cli._cprint", lambda *args, **kwargs: None)
    cli._ensure_runtime_credentials = MagicMock(return_value=True)

    assert cli.process_command("/side") is True

    assert cli._side_state is not None
    assert cli._side_state["parent_session_id"] == cli.session_id
    assert cli._side_state["conversation_history"][-1]["content"].startswith("Side conversation boundary.")


def test_side_command_rejects_nested_side(monkeypatch):
    cli = _prepare_cli()
    cli._side_state = {"session_id": "side_123", "conversation_history": [], "running": False}
    printed = []
    monkeypatch.setattr("cli._cprint", lambda msg, *args, **kwargs: printed.append(msg))

    cli.process_command("/side another question")

    assert any("unavailable in side conversations" in msg for msg in printed)


def test_side_command_with_question_runs_in_side_thread(monkeypatch):
    cli = _prepare_cli()
    monkeypatch.setattr("cli._cprint", lambda *args, **kwargs: None)
    monkeypatch.setattr("cli.AIAgent", _FakeSideAgent)
    monkeypatch.setattr("cli.threading.Thread", _ImmediateThread)
    monkeypatch.setattr("cli.ChatConsole", lambda: MagicMock(print=MagicMock()))
    cli._ensure_runtime_credentials = MagicMock(return_value=True)
    cli._resolve_turn_agent_config = MagicMock(
        return_value={
            "model": "gpt-5.4",
            "runtime": {
                "api_key": "token",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "provider": "openai-codex",
                "api_mode": None,
                "command": None,
                "args": [],
            },
            "request_overrides": None,
        }
    )

    cli.process_command("/side explain the parser")

    assert cli._side_state is not None
    assert cli._side_state["running"] is False
    history = cli._side_state["conversation_history"]
    assert history[-2]["content"] == "explain the parser"
    assert history[-1]["content"] == "side answer"


def test_close_side_conversation_clears_state(monkeypatch):
    cli = _prepare_cli()
    monkeypatch.setattr("cli._cprint", lambda *args, **kwargs: None)
    cli._side_state = {"session_id": "side_123", "conversation_history": [], "running": False}
    cli._side_queue = [("queued", [])]

    closed = cli._close_side_conversation()

    assert closed is True
    assert cli._side_state is None
    assert cli._side_queue == []

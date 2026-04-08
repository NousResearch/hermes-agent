"""Tests for gateway /compress truthfulness."""

import sys
import types
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/compress", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_history(n_messages: int) -> list[dict]:
    history = []
    for i in range(n_messages):
        history.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"message {i}",
            }
        )
    return history


def _make_runner(history: list[dict], session_id: str = "sess-current"):
    runner = object.__new__(gateway_run.GatewayRunner)
    session_entry = MagicMock()
    session_entry.session_id = session_id
    session_entry.session_key = "telegram:12345:67890"

    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    store.load_transcript.return_value = history
    store.rewrite_transcript = MagicMock()
    store.update_session = MagicMock()
    store._save = MagicMock()

    runner.session_store = store
    return runner, session_entry


class _NoOpCompressor:
    protect_first_n = 3

    def _align_boundary_forward(self, messages, idx):
        return idx

    def _find_tail_cut_by_tokens(self, messages, head_end):
        return head_end


class _NoOpAgent:
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.session_id = kwargs["session_id"]
        self.context_compressor = _NoOpCompressor()
        self._print_fn = None
        self._compress_context_calls = 0

    def _compress_context(self, messages, system_message, *, approx_tokens=None):
        self._compress_context_calls += 1
        return messages, system_message


class _CompressibleCompressor:
    protect_first_n = 1

    def _align_boundary_forward(self, messages, idx):
        return idx

    def _find_tail_cut_by_tokens(self, messages, head_end):
        return 3


class _CompressingAgent:
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.session_id = kwargs["session_id"]
        self.context_compressor = _CompressibleCompressor()
        self._print_fn = None
        self._compress_context_calls = 0

    def _compress_context(self, messages, system_message, *, approx_tokens=None):
        self._compress_context_calls += 1
        self.session_id = "sess-compressed"
        return (
            [
                {"role": "user", "content": "summary"},
                {"role": "assistant", "content": "latest reply"},
            ],
            system_message,
        )


@pytest.mark.asyncio
async def test_compress_command_reports_noop_truthfully(monkeypatch):
    event = _make_event()
    runner, session_entry = _make_runner(_make_history(4))

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "openai/test-model")
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _NoOpAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    result = await runner._handle_compress_command(event)

    assert result == "Nothing to compress yet (the transcript is still all protected context)."
    assert _NoOpAgent.last_instance is not None
    assert _NoOpAgent.last_instance._compress_context_calls == 0
    runner.session_store.rewrite_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()
    runner.session_store._save.assert_not_called()
    assert session_entry.session_id == "sess-current"


@pytest.mark.asyncio
async def test_compress_command_relabels_token_estimate_on_success(monkeypatch):
    event = _make_event()
    runner, session_entry = _make_runner(_make_history(6))

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "openai/test-model")
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CompressingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    result = await runner._handle_compress_command(event)

    assert "🗜️ Compressed: 6 → 2 messages" in result
    assert "Rough transcript estimate:" in result
    assert "\n~" not in result
    assert _CompressingAgent.last_instance is not None
    assert _CompressingAgent.last_instance._compress_context_calls == 1
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-compressed",
        [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "latest reply"},
        ],
    )
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        last_prompt_tokens=0,
    )
    runner.session_store._save.assert_called_once()
    assert session_entry.session_id == "sess-compressed"

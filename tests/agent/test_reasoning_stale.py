"""Regression coverage for reasoning-only streaming stalls (#78807)."""

import threading
import time
from types import SimpleNamespace

import pytest


def _agent(tmp_path, monkeypatch, timeout):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        f"agent:\n  reasoning_only_stale_timeout: {timeout}\n",
        encoding="utf-8",
    )
    from run_agent import AIAgent

    return AIAgent(
        model="deepseek/deepseek-v4-flash",
        provider="deepseek",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _reasoning_chunk():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    reasoning_content="thinking", content=None, tool_calls=None
                ),
                finish_reason=None,
            )
        ],
        model="deepseek/deepseek-v4-flash",
        usage=None,
    )


def _content_chunk():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    reasoning_content=None, content="done", tool_calls=None
                ),
                finish_reason=None,
            )
        ],
        model="deepseek/deepseek-v4-flash",
        usage=None,
    )


class _Stream:
    def __init__(self, factory, aborted, stopped):
        self.factory = factory
        self.aborted = aborted
        self.stopped = stopped

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped.is_set():
            raise StopIteration
        if self.aborted["value"]:
            raise ConnectionError("aborted")
        time.sleep(0.01)
        return self.factory()

    def close(self):
        return None


def _wire(agent, monkeypatch, factory):
    aborted = {"value": False}
    stopped = threading.Event()
    aborts = []
    create_calls = []

    def create(**_kwargs):
        create_calls.append(_kwargs)
        return _Stream(factory, aborted, stopped)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_kwargs: client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda _client, reason=None: (
            aborts.append(reason),
            aborted.__setitem__("value", True),
        ),
    )
    monkeypatch.setattr(agent, "_buffer_diagnostic_status", lambda _message: None)
    return aborted, stopped, aborts, create_calls


def _kwargs():
    return {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_reasoning_only_stream_is_aborted_without_retry(tmp_path, monkeypatch):
    from agent import chat_completion_helpers

    agent = _agent(tmp_path, monkeypatch, 0.2)
    _aborted, _stopped, aborts, create_calls = _wire(
        agent, monkeypatch, _reasoning_chunk
    )
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")

    with pytest.raises(Exception):
        chat_completion_helpers.interruptible_streaming_api_call(agent, _kwargs())

    assert "reasoning_only_stale_kill" in aborts
    assert len(create_calls) == 1


def test_reasoning_then_content_is_not_aborted(tmp_path, monkeypatch):
    from agent import chat_completion_helpers

    agent = _agent(tmp_path, monkeypatch, 0.2)
    started = time.monotonic()
    content_sent = False

    def factory():
        nonlocal content_sent
        elapsed = time.monotonic() - started
        if elapsed < 0.1:
            return _reasoning_chunk()
        if not content_sent:
            content_sent = True
            return _content_chunk()
        time.sleep(0.01)
        return SimpleNamespace(choices=[], usage=None)

    _aborted, stopped, aborts, _create_calls = _wire(agent, monkeypatch, factory)
    result = []

    thread = threading.Thread(
        target=lambda: result.append(
            chat_completion_helpers.interruptible_streaming_api_call(agent, _kwargs())
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.5)
    stopped.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert "reasoning_only_stale_kill" not in aborts

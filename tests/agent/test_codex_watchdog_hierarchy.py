"""Behavior contract for Codex Responses stream watchdog ordering and activity."""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import httpx
import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_agent(tmp_path, monkeypatch, *, provider="openai-codex"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    from run_agent import AIAgent

    base_url = (
        "https://chatgpt.com/backend-api/codex"
        if provider == "openai-codex"
        else "https://api.x.ai/v1"
    )
    agent = AIAgent(
        model="gpt-5.5" if provider == "openai-codex" else "grok-4.3",
        provider=provider,
        api_key="test-key",
        base_url=base_url,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_buffer_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_emit_wait_notice", lambda *args, **kwargs: None)
    return agent


def _completed_event():
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(status="completed", id="resp-test", usage=None),
    )


class _EventStream:
    def __init__(self, events, *, delay=0.0):
        self._events = events
        self._delay = delay

    def __iter__(self):
        for event in self._events:
            if self._delay:
                time.sleep(self._delay)
            yield event

    def close(self):
        pass


def _install_stream(agent, monkeypatch, stream_factory):
    closes = []

    class Responses:
        def create(self, **kwargs):
            return stream_factory()

    client = SimpleNamespace(responses=Responses())
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda request_client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda request_client, reason=None: closes.append(reason),
    )
    return closes


@pytest.mark.parametrize(
    "events",
    [
        [SimpleNamespace(type="response.reasoning_text.delta", delta="thinking")],
        [SimpleNamespace(type="response.output_text.delta", delta="answer")],
        [SimpleNamespace(type="response.function_call_arguments.delta", delta="{}")],
        [
            SimpleNamespace(type="response.in_progress"),
            SimpleNamespace(type="response.usage", input_tokens=10),
            SimpleNamespace(type="codex.rate_limits", remaining=10),
        ],
    ],
    ids=["reasoning", "content", "tool", "usage-provider-status"],
)
def test_parsed_activity_can_stream_beyond_generic_stale_timeout(
    tmp_path, monkeypatch, events
):
    """Parsed reasoning/content/tool/provider-status events refresh one clock."""
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 0.35
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "5")
    closes = _install_stream(
        agent,
        monkeypatch,
        lambda: _EventStream(events * 7 + [_completed_event()], delay=0.1),
    )

    response = helpers.interruptible_api_call(
        agent, {"model": agent.model, "input": "hi"}
    )

    assert response.status == "completed"
    assert "stale_call_kill" not in closes
    assert "codex_ttfb_kill" not in closes
    assert "codex_stream_idle_kill" not in closes
    assert "codex_hard_timeout_kill" not in closes


def test_no_first_parsed_event_uses_ttfb_watchdog(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 5.0
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "2")
    stop = {"value": False}

    def no_parsed_events(api_kwargs, client=None, on_first_delta=None):
        while not stop["value"]:
            time.sleep(0.05)

    monkeypatch.setattr(agent, "_run_codex_stream", no_parsed_events)
    closes = _install_stream(agent, monkeypatch, lambda: _EventStream([]))

    try:
        with pytest.raises(TimeoutError, match="TTFB"):
            helpers.interruptible_api_call(
                agent, {"model": agent.model, "input": "hi"}
            )
    finally:
        stop["value"] = True

    assert "codex_ttfb_kill" in closes
    assert "codex_stream_idle_kill" not in closes


def test_stream_then_true_event_idle_uses_idle_watchdog(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 5.0
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "2")

    def stream_factory():
        def events():
            yield SimpleNamespace(type="response.in_progress")
            time.sleep(5)

        return _EventStream(events())

    closes = _install_stream(agent, monkeypatch, stream_factory)

    with pytest.raises(TimeoutError, match="after first byte"):
        helpers.interruptible_api_call(agent, {"model": agent.model, "input": "hi"})

    assert "codex_stream_idle_kill" in closes
    assert "codex_ttfb_kill" not in closes


def test_comment_only_and_raw_unparsed_bytes_do_not_establish_activity():
    from agent.codex_runtime import _consume_codex_event_stream

    seen = []
    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        _consume_codex_event_stream(
            iter([b": keepalive\n\n", b"raw unparsed bytes"]),
            model="gpt-5.5",
            on_event=seen.append,
        )

    assert seen == []


def _install_active_fake_stream(agent, monkeypatch, *, duration=5.0):
    sentinel = SimpleNamespace(status="completed")
    stop = {"value": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not stop["value"]:
            agent._codex_stream_last_event_ts = time.time()
            time.sleep(0.08)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)
    closes = _install_stream(agent, monkeypatch, lambda: _EventStream([]))
    return sentinel, closes, stop


def test_active_stream_is_killed_only_at_absolute_hard_ceiling(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 0.35
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0.75")
    _, closes, stop = _install_active_fake_stream(agent, monkeypatch)

    try:
        with pytest.raises(TimeoutError, match="hard ceiling"):
            helpers.interruptible_api_call(
                agent, {"model": agent.model, "input": "hi"}
            )
    finally:
        stop["value"] = True

    assert "codex_hard_timeout_kill" in closes
    assert "stale_call_kill" not in closes


def test_hard_ceiling_zero_override_remains_disabled(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 0.35
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0")
    sentinel, closes, _ = _install_active_fake_stream(
        agent, monkeypatch, duration=0.8
    )

    response = helpers.interruptible_api_call(
        agent, {"model": agent.model, "input": "hi"}
    )

    assert response is sentinel
    assert "codex_hard_timeout_kill" not in closes
    assert "stale_call_kill" not in closes


@pytest.mark.parametrize("provider", ["xai-oauth", "openai-codex"])
def test_hard_ceiling_applies_to_every_codex_responses_provider(
    tmp_path, monkeypatch, provider
):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch, provider=provider)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 5.0
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0.75")
    _, closes, stop = _install_active_fake_stream(agent, monkeypatch)

    try:
        with pytest.raises(TimeoutError, match="hard ceiling"):
            helpers.interruptible_api_call(
                agent, {"model": agent.model, "input": "hi"}
            )
    finally:
        stop["value"] = True

    assert "codex_hard_timeout_kill" in closes


def test_codex_transport_retry_is_preserved(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    calls = {"count": 0}

    class Responses:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                def broken():
                    raise httpx.ReadTimeout("retry me")
                    yield  # pragma: no cover

                return _EventStream(broken())
            return _EventStream([_completed_event()])

    client = SimpleNamespace(responses=Responses())

    response = agent._run_codex_stream({"model": agent.model}, client=client)

    assert response.status == "completed"
    assert calls["count"] == 2


def test_non_codex_request_keeps_generic_stale_watchdog(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    agent.api_mode = "chat_completions"
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 0.35
    )
    closes = []
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: time.sleep(5))
        )
    )
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda request_client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda request_client, reason=None: closes.append(reason),
    )

    with pytest.raises(TimeoutError, match="with no response"):
        helpers.interruptible_api_call(agent, {"model": agent.model, "messages": []})

    assert "stale_call_kill" in closes

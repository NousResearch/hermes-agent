"""Streaming provider-response terminal boundary regressions.

Once a streaming transport reaches its provider terminal (finish_reason /
message_stop / terminal response event), every later failure is Hermes-local:
no reconnect, no retry, no partial-stream "length" stub, no continuation —
the turn ends via the unified finalizer. Pre-terminal network failures keep
the baseline reconnect/retry behavior. Interruption semantics are unchanged.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

import run_agent
from run_agent import AIAgent

LOCAL_REASON = "local_post_response_error"


# ── chat_completions streaming fakes ─────────────────────────────────────────

def _chunk(text=None, finish=None, usage=None):
    delta = SimpleNamespace(
        content=text, reasoning_content=None, reasoning=None, tool_calls=None
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], model="test/model", usage=usage)


class _FakeStreamClient:
    """Non-Mock fake: passes the production isinstance(client, Mock) guard."""

    def __init__(self, stream_factory):
        self.calls = 0
        self._stream_factory = stream_factory
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return self._stream_factory()

    def close(self):
        pass


def _good_stream():
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return iter([_chunk("he"), _chunk("llo", finish="stop", usage=usage)])


# ── agent construction ───────────────────────────────────────────────────────

def _make_agent(fake, *, api_mode="chat_completions", provider="custom") -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="test-model",
            api_key="test-key-not-a-secret",
            base_url="https://test.invalid/v1",
            provider=provider,
            api_mode=api_mode,
            quiet_mode=False,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are a test double."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda text: None
    agent.client = fake
    return agent


def _run(agent, extra_patches=()):
    with ExitStack() as stack:
        stack.enter_context(patch("run_agent.jittered_backoff", return_value=0))
        stack.enter_context(patch("time.sleep", return_value=None))
        stack.enter_context(patch.object(agent, "_save_trajectory"))
        stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        stack.enter_context(patch.object(agent, "_persist_session"))
        for extra in extra_patches:
            stack.enter_context(extra)
        return agent.run_conversation("streaming boundary probe")


# ── chat_completions streaming ───────────────────────────────────────────────

def test_chat_streaming_normal_completion_single_call():
    fake = _FakeStreamClient(_good_stream)
    agent = _make_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["completed"] is True
    assert result["failed"] is False
    assert result["final_response"] == "hello"


def test_chat_streaming_post_terminal_local_error_calls_once():
    """Local failure after the terminal frame (finish_reason) must not retry."""
    fake = _FakeStreamClient(_good_stream)
    agent = _make_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            # _reset_stale_streak runs on the streaming success path, after
            # the terminal frame and final assembly — deterministic
            # post-boundary local failure.
            patch(
                "agent.chat_completion_helpers._reset_stale_streak",
                side_effect=RuntimeError("injected post-terminal local failure"),
            ),
        ),
    )
    assert fake.calls == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == LOCAL_REASON
    assert "API call failed" not in str(result["final_response"])


def test_chat_streaming_terminal_then_network_typed_local_error_does_not_retry():
    """Exception type must not override a position-proven terminal boundary."""

    def terminal_then_error():
        yield _chunk("done", finish="stop")
        raise httpx.ReadTimeout("raised after terminal frame")

    fake = _FakeStreamClient(terminal_then_error)
    agent = _make_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["failed"] is True
    assert result["turn_exit_reason"] == LOCAL_REASON


def test_chat_streaming_mid_stream_transport_error_keeps_baseline_retry(monkeypatch):
    """Pre-terminal transport failure: baseline retry behavior is preserved.

    The baseline issues 6 raw calls for this shape (3 outer attempts x the
    error classifier's within-attempt handling); the candidate must produce
    the same count, not fewer and not more. See BASELINE_COMPARISON.md.
    """
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    def failing_stream():
        yield _chunk(None)
        raise httpx.ReadTimeout("simulated mid-stream transport timeout")

    fake = _FakeStreamClient(failing_stream)
    agent = _make_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 6  # == locked-baseline count for this scenario
    assert result["failed"] is True
    assert result.get("turn_exit_reason") != LOCAL_REASON


def test_chat_streaming_mid_stream_local_error_keeps_baseline_retry(monkeypatch):
    """ACCEPTED BOUNDARY SEMANTICS (see ARCHITECTURE_DECISION.md): a
    Hermes-local failure BEFORE the terminal frame cannot be distinguished
    from a transport failure without exception-type guessing, so it keeps
    the baseline retry classification. This test pins that behavior so any
    future boundary change is deliberate."""
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    fake = _FakeStreamClient(_good_stream)
    agent = _make_agent(fake)
    real_touch = agent._touch_activity

    def picky_touch(msg=""):
        if msg == "receiving stream response":
            raise RuntimeError("injected mid-stream local failure")
        return real_touch(msg)

    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            patch.object(agent, "_touch_activity", picky_touch),
        ),
    )
    assert fake.calls > 1  # baseline retry behavior, by design
    assert result.get("turn_exit_reason") != LOCAL_REASON


def test_chat_streaming_network_error_before_boundary_then_success():
    fake = _FakeStreamClient(_good_stream)
    agent = _make_agent(fake)
    state = {"n": 0}

    def flaky(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("wire down before stream established")
        return _good_stream()

    fake.chat.completions.create = flaky
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert state["n"] == 2
    assert result["completed"] is True
    assert result["final_response"] == "hello"


def test_chat_streaming_empty_stream_stays_provider_anomaly(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    fake = _FakeStreamClient(lambda: iter([]))
    agent = _make_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    # EmptyStreamError = provider anomaly without a terminal frame: must not
    # be reported as a local post-response failure; provider-side retry
    # classification is preserved.
    assert fake.calls >= 2
    assert result.get("turn_exit_reason") != LOCAL_REASON


# ── Bedrock streaming ────────────────────────────────────────────────────────

def _bedrock_stream_events():
    return {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "he"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
        ]
    }


class _BedrockTerminalThenLocalError:
    def __iter__(self):
        yield {"messageStop": {"stopReason": "end_turn"}}
        raise RuntimeError("local event handling failed after messageStop")







# ── Anthropic streaming ──────────────────────────────────────────────────────

class _FakeAnthropicStream:
    def __init__(self, events, final_message=None, final_error=None):
        self._events = events
        self._final_message = final_message
        self._final_error = final_error
        self.response = None

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        if self._final_error is not None:
            raise self._final_error
        return self._final_message


class _AnthropicTerminalThenLocalError:
    response = None

    def __iter__(self):
        yield SimpleNamespace(type="message_stop")
        raise RuntimeError("local event handling failed after message_stop")

    def get_final_message(self):
        raise AssertionError("must not reach final assembly")


class _NonIterableAnthropicStream:
    response = None

    def __init__(self, final_message):
        self._final_message = final_message

    def get_final_message(self):
        return self._final_message


def _anthropic_final_message(text="hello"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="test-model",
    )


class _FakeAnthropicStreamCM:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc):
        return False


def _anthropic_events_with_stop():
    return [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(type="message_stop"),
    ]


def test_anthropic_streaming_post_messagestop_local_error_calls_once():
    agent = _make_agent(None, api_mode="anthropic_messages", provider="custom")
    stream_calls = {"n": 0}

    def fake_stream(**kwargs):
        stream_calls["n"] += 1
        return _FakeAnthropicStreamCM(
            _FakeAnthropicStream(
                _anthropic_events_with_stop(),
                final_error=RuntimeError("injected final-assembly local failure"),
            )
        )

    agent._anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(stream=fake_stream)
    )
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_try_refresh_anthropic_client_credentials", lambda: None),
            # #67142: the streaming path builds a per-request client via this
            # factory instead of using the shared _anthropic_client.
            patch.object(
                agent,
                "_create_request_anthropic_client",
                lambda **_: agent._anthropic_client,
            ),
        ),
    )
    assert stream_calls["n"] == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == LOCAL_REASON
    assert "API call failed" not in str(result["final_response"])




def test_anthropic_terminal_event_marks_before_later_iterator_error():
    agent = _make_agent(None, api_mode="anthropic_messages", provider="custom")
    stream_calls = {"n": 0}

    def fake_stream(**kwargs):
        stream_calls["n"] += 1
        return _FakeAnthropicStreamCM(_AnthropicTerminalThenLocalError())

    agent._anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(stream=fake_stream)
    )
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_try_refresh_anthropic_client_credentials", lambda: None),
            # #67142: per-request client factory — see test above.
            patch.object(
                agent,
                "_create_request_anthropic_client",
                lambda **_: agent._anthropic_client,
            ),
        ),
    )
    assert stream_calls["n"] == 1
    assert result["failed"] is True
    assert result["turn_exit_reason"] == LOCAL_REASON






# ── Codex Responses streaming ────────────────────────────────────────────────

class _FakeCodexClient:
    def __init__(self, events):
        self.calls = 0
        self._events = events
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        assert kwargs.get("stream") is True
        return list(self._events)

    def close(self):
        pass


def _codex_events_terminal(status="completed"):
    return [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(
            type=f"response.{status}",
            response=SimpleNamespace(
                status=status,
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                id="resp-1",
                incomplete_details=(
                    SimpleNamespace(reason="max_output_tokens")
                    if status == "incomplete"
                    else None
                ),
                error=SimpleNamespace(message="boom") if status == "failed" else None,
            ),
        ),
    ]


class _ExplosiveCodexTerminalResponse:
    status = "incomplete"
    id = "resp-explosive"
    incomplete_details = SimpleNamespace(reason="max_output_tokens")
    error = None

    @property
    def usage(self):
        raise RuntimeError("injected terminal-field local failure")


def _codex_events_terminal_field_error():
    return [
        SimpleNamespace(type="response.output_text.delta", delta="partial"),
        SimpleNamespace(
            type="response.incomplete",
            response=_ExplosiveCodexTerminalResponse(),
        ),
    ]


def _make_codex_agent(fake):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="gpt-5-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="codex-token",
            api_mode="codex_responses",
            provider="openai-codex",
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are Hermes."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent.client = fake
    return agent


def test_codex_streaming_normal_completion_single_call():
    fake = _FakeCodexClient(_codex_events_terminal("completed"))
    agent = _make_codex_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["completed"] is True
    assert result["final_response"] == "hello"


def test_codex_streaming_post_terminal_local_error_calls_once():
    """Terminal callback fires before local terminal-field materialization."""
    fake = _FakeCodexClient(_codex_events_terminal_field_error())
    agent = _make_codex_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == LOCAL_REASON
    assert "API call failed" not in str(result["final_response"])

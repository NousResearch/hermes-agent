"""Provider-path terminal boundary tests — real provider identities and
transport routing (no generic custom-provider substitutes).

Covers: Codex Responses terminal-event boundary, Kimi Coding (anthropic
messages) message_stop boundary, Kimi API / DeepSeek chat-completions
finish_reason boundary, and auxiliary Codex/Anthropic calls running on
detached attempt tokens. Terminal-after local error keeps wire call_count=1;
pre-terminal network errors keep baseline retry/reconnect.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import model_tools  # noqa: F401  (triggers plugin discovery)
import pytest

import run_agent
from run_agent import AIAgent

LOCAL_REASON = "local_post_response_error"


# ── shared fakes ─────────────────────────────────────────────────────────────

def _chunk(text=None, finish=None, usage=None):
    delta = SimpleNamespace(
        content=text, reasoning_content=None, reasoning=None, tool_calls=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
        model="test/model",
        usage=usage,
    )


def _good_stream(text="hello"):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return iter([_chunk(text[:2]), _chunk(text[2:], finish="stop", usage=usage)])


def _response(content, *, finish_reason="stop", tool_calls=None, reasoning_content=None):
    message = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning_content
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_call(name, call_id="call-provider-path-1"):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _tool_defs(*names):
    return [
        {"type": "function", "function": {"name": n, "description": "t", "parameters": {"type": "object", "properties": {}}}}
        for n in names
    ]


class _FakeStreamClient:
    def __init__(self, factory):
        self.calls = 0
        self.kwargs_seen = []
        self._factory = factory
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.kwargs_seen.append(kwargs)
        return self._factory()

    def close(self):
        pass


def _run(agent, extra_patches=(), prompt="provider path probe"):
    with ExitStack() as stack:
        stack.enter_context(patch("run_agent.jittered_backoff", return_value=0))
        stack.enter_context(patch("time.sleep", return_value=None))
        stack.enter_context(patch.object(agent, "_save_trajectory"))
        stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        stack.enter_context(patch.object(agent, "_persist_session"))
        for extra in extra_patches:
            stack.enter_context(extra)
        return agent.run_conversation(prompt)


# ── §A shared streaming terminal blocking ────────────────────────────────────

def _shared_agent(fake, *, quiet=False):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="test-model", api_key="test-key-not-a-secret",
            base_url="https://test.invalid/v1", provider="custom",
            quiet_mode=quiet, skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "s"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent._disable_streaming = quiet
    if not quiet:
        agent.stream_delta_callback = lambda t: None
    agent.client = fake
    return agent


def test_a1_terminal_then_runtime_error_no_stub_no_continuation():
    fake = _FakeStreamClient(lambda: _good_stream())
    agent = _shared_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            patch(
                "agent.chat_completion_helpers._reset_stale_streak",
                side_effect=RuntimeError("post-terminal local failure"),
            ),
        ),
    )
    assert fake.calls == 1
    assert result["api_calls"] == 1
    assert result["turn_exit_reason"] == LOCAL_REASON
    assert result["failed"] is True and result["completed"] is False
    # no length-stub, no continuation user row
    roles = [m.get("role") for m in result["messages"]]
    assert roles.count("user") == 1


def test_a2_terminal_then_network_typed_error_is_local():
    def terminal_then_error():
        yield _chunk("done", finish="stop")
        raise httpx.ReadTimeout("network-typed error after terminal")

    fake = _FakeStreamClient(terminal_then_error)
    agent = _shared_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["api_calls"] == 1
    assert result["turn_exit_reason"] == LOCAL_REASON


def test_a3_pre_terminal_network_error_then_success_retries():
    fake = _FakeStreamClient(lambda: _good_stream())
    agent = _shared_agent(fake)
    state = {"n": 0}

    def flaky(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("wire down before terminal")
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


def test_a4_interruption_stays_interruption():
    fake = _FakeStreamClient(lambda: _good_stream())
    agent = _shared_agent(fake)

    def cancel(msg=""):
        agent._interrupt_requested = True

    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            patch.object(agent, "_touch_activity", cancel),
        ),
    )
    reason = str(result.get("turn_exit_reason", ""))
    assert "interrupt" in reason.lower()
    assert reason != LOCAL_REASON


# ── Codex OAuth / Responses ─────────────────────────────────────────────────

class _FakeCodexClient:
    def __init__(self, events):
        self.calls = 0
        self.kwargs_seen = []
        self._events = events
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        self.kwargs_seen.append(kwargs)
        assert kwargs.get("stream") is True
        return list(self._events)

    def close(self):
        pass


def _codex_agent(fake):
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="gpt-5-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="codex-placeholder",
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
    agent.valid_tool_names = {"terminal"}
    agent.client = fake
    return agent


def test_b1_codex_oauth_credentials_applied_to_runtime_client():
    agent = _codex_agent(_FakeCodexClient([]))
    # session already on singleton tokens (matches resolver output below)
    agent.api_key = "fake-oauth-access-token"
    oauth_out = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "fake-oauth-access-token",
        "source": "hermes-auth-store",
        "auth_mode": "chatgpt",
        "last_refresh": 1784300000,
    }
    with patch(
        "hermes_cli.auth.resolve_codex_runtime_credentials", return_value=oauth_out
    ) as resolver:
        ok = agent._try_refresh_codex_client_credentials(force=True)
    assert ok is True
    assert resolver.called
    assert agent.api_key == "fake-oauth-access-token"
    assert agent.base_url == "https://chatgpt.com/backend-api/codex"


def test_b2_codex_responses_request_shape_and_final_text():
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                id="r1",
                incomplete_details=None,
                error=None,
            ),
        ),
    ]
    fake = _FakeCodexClient(events)
    agent = _codex_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["completed"] is True
    assert result["final_response"] == "hello"
    kw = fake.kwargs_seen[0]
    assert kw["model"] == "gpt-5-codex"
    assert isinstance(kw.get("instructions"), str) and kw["instructions"]
    assert isinstance(kw.get("input"), list) and kw["input"]
    assert kw.get("store") is False


def test_b3_codex_reasoning_commentary_toolcall_no_regression():
    commentary_item = SimpleNamespace(
        type="message", phase="commentary", status="completed",
        content=[SimpleNamespace(type="output_text", text="checking first")],
    )
    function_item = SimpleNamespace(
        type="function_call", id="fc_1", call_id="call_1",
        name="terminal", arguments="{}",
    )
    turn1 = [
        SimpleNamespace(type="response.reasoning_text.delta", delta="private plan"),
        SimpleNamespace(type="response.output_item.done", item=commentary_item),
        SimpleNamespace(type="response.output_item.done", item=function_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", usage=None, id="r1",
                                     incomplete_details=None, error=None),
        ),
    ]
    turn2 = [
        SimpleNamespace(type="response.output_text.delta", delta="done"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", usage=None, id="r2",
                                     incomplete_details=None, error=None),
        ),
    ]
    streams = [turn1, turn2]
    fake = _FakeCodexClient([])
    fake._events = None

    def _create(**kwargs):
        fake.calls += 1
        fake.kwargs_seen.append(kwargs)
        return list(streams.pop(0))

    fake.responses = SimpleNamespace(create=_create)
    agent = _codex_agent(fake)
    with patch("run_agent.handle_function_call", return_value="tool-out") as tool:
        result = _run(
            agent,
            extra_patches=(
                patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            ),
        )
    assert tool.call_count == 1
    assert fake.calls == 2
    assert result["completed"] is True
    assert result["final_response"] == "done"
    # reasoning + commentary preserved on the transcript
    assistant_rows = [m for m in result["messages"] if m.get("role") == "assistant"]
    joined = str(assistant_rows)
    assert "private plan" in joined or "checking first" in joined


def test_b4_codex_terminal_then_local_error_once():
    # Terminal response object whose fields raise locally when Hermes reads
    # them AFTER the terminal event was observed (on_terminal already fired).
    class _ExplodingTerminalResponse:
        def __getattr__(self, name):
            raise RuntimeError("post-terminal local failure")

    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hel"),
        SimpleNamespace(
            type="response.incomplete",
            response=_ExplodingTerminalResponse(),
        ),
    ]
    fake = _FakeCodexClient(events)
    agent = _codex_agent(fake)
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert fake.calls == 1
    assert result["turn_exit_reason"] == LOCAL_REASON


# ── Kimi Coding (api.kimi.com/coding, anthropic wire) ────────────────────────

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


class _FakeAnthropicCM:
    def __init__(self, stream):
        self._s = stream

    def __enter__(self):
        return self._s

    def __exit__(self, *e):
        return False


def _kimi_coding_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="kimi-k2.7-code",
            api_key="sk-kimi-test-not-real",
            base_url="https://api.kimi.com/coding",
            provider="kimi-coding",
            api_mode="anthropic_messages",
            quiet_mode=False,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "s"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent.valid_tool_names = {"terminal"}
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda t: None
    return agent


def test_b5_kimi_coding_key_prefix_routing_real_resolver():
    from hermes_cli.auth import _resolve_kimi_base_url

    assert _resolve_kimi_base_url("sk-kimi-abc", "https://api.moonshot.ai/v1", None) == "https://api.kimi.com/coding"
    assert _resolve_kimi_base_url("sk-legacy", "https://api.moonshot.ai/v1", None) == "https://api.moonshot.ai/v1"
    assert _resolve_kimi_base_url("sk-kimi-abc", "https://api.moonshot.ai/v1", "https://proxy.invalid/kimi") == "https://proxy.invalid/kimi"


def test_b6_kimi_coding_anthropic_client_construction():
    agent = _kimi_coding_agent()
    agent._rebuild_anthropic_client()
    client = agent._anthropic_client
    assert client.api_key == "sk-kimi-test-not-real"
    assert "api.kimi.com/coding" in str(client.base_url)
    headers = getattr(client, "_custom_headers", {}) or {}
    assert headers.get("User-Agent") == "claude-code/0.1.0"


def _kimi_stream_events(with_tool=False):
    events = [SimpleNamespace(type="message_start")]
    if with_tool:
        events += [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", id="t1", name="terminal"),
            ),
            SimpleNamespace(type="content_block_stop"),
        ]
    else:
        events += [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="kimi says hi"),
            ),
        ]
    events.append(SimpleNamespace(type="message_stop"))
    return events


def test_b7_kimi_coding_stream_turn_toolcall_and_final_text():
    final = SimpleNamespace(
        role="assistant",
        content=[
            SimpleNamespace(type="text", text="using tool"),
            SimpleNamespace(type="tool_use", id="t1", name="terminal", input={}),
        ],
        stop_reason="tool_use",
    )
    calls = {"n": 0}
    finals = [
        final,
        SimpleNamespace(
            role="assistant",
            content=[SimpleNamespace(type="text", text="kimi done")],
            stop_reason="end_turn",
        ),
    ]

    def fake_stream(**kwargs):
        calls["n"] += 1
        return _FakeAnthropicCM(
            _FakeAnthropicStream(_kimi_stream_events(with_tool=calls["n"] == 1), finals.pop(0))
        )

    agent = _kimi_coding_agent()
    agent._anthropic_client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))
    with patch("run_agent.handle_function_call", return_value="tool-out") as tool:
        result = _run(
            agent,
            extra_patches=(
                patch.object(agent, "_try_refresh_anthropic_client_credentials", lambda: None),
                # #67142: per-request client factory
                patch.object(
                    agent,
                    "_create_request_anthropic_client",
                    lambda **_: agent._anthropic_client,
                ),
            ),
        )
    assert tool.call_count == 1
    assert calls["n"] == 2
    assert result["completed"] is True
    assert result["final_response"] == "kimi done"


def test_b8_kimi_coding_terminal_then_local_error_once():
    calls = {"n": 0}

    def fake_stream(**kwargs):
        calls["n"] += 1
        return _FakeAnthropicCM(
            _FakeAnthropicStream(
                _kimi_stream_events(),
                final_error=RuntimeError("post-messageStop local failure"),
            )
        )

    agent = _kimi_coding_agent()
    agent._anthropic_client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_try_refresh_anthropic_client_credentials", lambda: None),
            # #67142: per-request client factory
            patch.object(
                agent,
                "_create_request_anthropic_client",
                lambda **_: agent._anthropic_client,
            ),
        ),
    )
    assert calls["n"] == 1
    assert result["turn_exit_reason"] == LOCAL_REASON


# ── Kimi API (api.moonshot.ai/v1, chat_completions) ──────────────────────────

def _kimi_api_agent(**overrides):
    kwargs = dict(
        model="kimi-k2-thinking",
        api_key="sk-test-not-real",
        base_url="https://api.moonshot.ai/v1",
        provider="kimi-coding",
        api_mode="chat_completions",
        quiet_mode=False,
        skip_context_files=True,
        skip_memory=True,
    )
    kwargs.update(overrides)
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(**kwargs)
    agent._cached_system_prompt = "s"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent.valid_tool_names = {"terminal"}
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda t: None
    return agent


def test_b9_kimi_api_request_construction_and_reasoning_params():
    fake = _FakeStreamClient(lambda: _good_stream())
    agent = _kimi_api_agent()
    agent.client = fake
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
        ),
    )
    assert result["completed"] is True
    # endpoint + key as configured at the agent layer
    assert agent.base_url == "https://api.moonshot.ai/v1"
    assert agent.api_key == "sk-test-not-real"
    # profile extras flow through the real kwargs builder
    built = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    assert built["model"] == "kimi-k2-thinking"
    assert built.get("extra_body", {}).get("thinking") == {"type": "enabled"} or built.get("reasoning_effort")
    # profile default UA header is registered on the provider profile
    from providers import get_provider_profile

    profile = get_provider_profile("kimi-coding")
    assert profile.default_headers.get("User-Agent") == "hermes-agent/1.0"


def test_b10_kimi_api_reasoning_toolcall_and_terminal_boundary():
    seq = [
        lambda: _response(
            None,
            tool_calls=[_tool_call("terminal")],
            reasoning_content="kimi private reasoning",
        ),
        lambda: _response("kimi final"),
    ]

    def factory():
        return seq.pop(0)()

    fake = _FakeStreamClient(factory)
    agent = _kimi_api_agent()
    agent.client = fake
    with patch("run_agent.handle_function_call", return_value="tool-out") as tool:
        result = _run(
            agent,
            extra_patches=(
                patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            ),
        )
    assert tool.call_count == 1
    assert fake.calls == 2
    assert result["completed"] is True
    assert result["final_response"] == "kimi final"
    rows = str(result["messages"])
    assert "kimi private reasoning" in rows


def test_b11_kimi_api_terminal_then_local_error_once():
    fake = _FakeStreamClient(lambda: _good_stream())
    agent = _kimi_api_agent()
    agent.client = fake
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            patch(
                "agent.chat_completion_helpers._reset_stale_streak",
                side_effect=RuntimeError("post-terminal local failure"),
            ),
        ),
    )
    assert fake.calls == 1
    assert result["turn_exit_reason"] == LOCAL_REASON


# ── DeepSeek (api.deepseek.com, dedicated profile) ───────────────────────────

def _deepseek_agent(model="deepseek-reasoner", **overrides):
    kwargs = dict(
        model=model,
        api_key="sk-deepseek-test-not-real",
        base_url="https://api.deepseek.com/v1",
        provider="deepseek",
        api_mode="chat_completions",
        quiet_mode=False,
        skip_context_files=True,
        skip_memory=True,
    )
    kwargs.update(overrides)
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(**kwargs)
    agent._cached_system_prompt = "s"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent.valid_tool_names = {"terminal"}
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda t: None
    return agent


def test_b12_deepseek_profile_and_reasoning_params():
    from providers import get_provider_profile

    profile = get_provider_profile("deepseek")
    assert profile.base_url == "https://api.deepseek.com/v1"
    extra_body, top = profile.build_api_kwargs_extras(
        reasoning_config={"effort": "high"}, model="deepseek-reasoner"
    )
    assert extra_body["thinking"] == {"type": "enabled"}
    assert top["reasoning_effort"] == "high"
    extra_body2, top2 = profile.build_api_kwargs_extras(model="deepseek-chat")
    assert extra_body2 == {} and top2 == {}

    agent = _deepseek_agent()
    built = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    assert built["model"] == "deepseek-reasoner"
    assert built.get("extra_body", {}).get("thinking") == {"type": "enabled"}


def test_b13_deepseek_reasoning_content_toolcall_and_final_text():
    seq = [
        lambda: _response(
            None,
            tool_calls=[_tool_call("terminal")],
            reasoning_content="deepseek reasoning_content",
        ),
        lambda: _response("deepseek final"),
    ]

    def factory():
        return seq.pop(0)()

    fake = _FakeStreamClient(factory)
    agent = _deepseek_agent()
    agent.client = fake
    with patch("run_agent.handle_function_call", return_value="tool-out") as tool:
        result = _run(
            agent,
            extra_patches=(
                patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            ),
        )
    assert tool.call_count == 1
    assert fake.calls == 2
    assert result["completed"] is True
    assert result["final_response"] == "deepseek final"
    assert "deepseek reasoning_content" in str(result["messages"])


def test_b14_deepseek_terminal_then_local_error_once():
    fake = _FakeStreamClient(lambda: _good_stream())
    agent = _deepseek_agent()
    agent.client = fake
    result = _run(
        agent,
        extra_patches=(
            patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake),
            patch(
                "agent.chat_completion_helpers._reset_stale_streak",
                side_effect=RuntimeError("post-terminal local failure"),
            ),
        ),
    )
    assert fake.calls == 1
    assert result["turn_exit_reason"] == LOCAL_REASON

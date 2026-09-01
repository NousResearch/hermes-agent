from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_constants import parse_reasoning_effort
from providers import get_provider_profile


def _kwargs_for(model: str, effort: str | None):
    profile = get_provider_profile("cliproxyapi")
    assert profile is not None
    return ChatCompletionsTransport().build_kwargs(
        model,
        [{"role": "user", "content": "hi"}],
        tools=None,
        provider_profile=profile,
        reasoning_config=parse_reasoning_effort(effort) if effort else None,
        timeout=30,
        max_tokens=None,
        ephemeral_max_output_tokens=None,
        max_tokens_param_fn=lambda n: {"max_tokens": n},
        request_overrides=None,
        session_id="test-session",
        base_url="http://127.0.0.1:8317/v1",
        supports_reasoning=False,
    )


def test_cliproxyapi_gpt_reasoning_effort_is_top_level():
    assert _kwargs_for("gpt-5.5", "high")["reasoning_effort"] == "high"
    assert _kwargs_for("gpt-5.5", "max")["reasoning_effort"] == "xhigh"


def test_cliproxyapi_claude_effort_and_disable_are_top_level():
    assert _kwargs_for("neko-opus-4.8", "max")["reasoning_effort"] == "max"
    assert _kwargs_for("some-provider/sonnet-5", "none")["reasoning_effort"] == "none"


def test_cliproxyapi_grok_effort_and_cache_affinity():
    kwargs = _kwargs_for("grok-4.5", "xhigh")
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"]["prompt_cache_key"] == "test-session"
    assert kwargs["extra_headers"] == {
        "X-Session-ID": "test-session",
        "x-grok-conv-id": "test-session",
    }


def test_cliproxyapi_grok_none_keeps_affinity_without_invalid_dial():
    kwargs = _kwargs_for("grok-4.5", "none")
    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["prompt_cache_key"] == "test-session"


def test_cliproxyapi_opencode_go_effort_mapping():
    assert _kwargs_for("kimi-k3", "low")["reasoning_effort"] == "max"
    assert _kwargs_for("glm-5.2", "xhigh")["reasoning_effort"] == "max"
    assert _kwargs_for("kimi-k2.7-code", "max")["reasoning_effort"] == "high"
    assert _kwargs_for("deepseek-v4-pro", "xhigh")["reasoning_effort"] == "high"
    assert _kwargs_for("qwen3.7-max", "none")["reasoning_effort"] == "none"


def test_cliproxyapi_unrelated_model_omits_reasoning_effort():
    kwargs = _kwargs_for("gemini-3.1-pro-high", "high")
    assert "reasoning_effort" not in kwargs

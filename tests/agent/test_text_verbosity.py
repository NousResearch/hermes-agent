"""Behavior contract for OpenAI Responses text verbosity."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.codex_responses_adapter import (
    _preflight_codex_api_kwargs,
    classify_responses_route,
    supports_openai_text_verbosity_route,
)
from agent.text_verbosity import (
    finalize_text_verbosity_request,
    parse_text_verbosity,
    supports_openai_text_verbosity,
)
from agent.transports import get_transport
from agent.transports.chat_completions import ChatCompletionsTransport
from run_agent import AIAgent


def _route_agent(provider: str, base_url: str, *, api_mode: str = "codex_responses"):
    return SimpleNamespace(
        provider=provider,
        base_url=base_url,
        _base_url_hostname="",
        _base_url_lower=base_url.lower(),
        api_mode=api_mode,
        text_verbosity="low",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("low", "low"),
        (" Medium ", "medium"),
        ("HIGH", "high"),
        ("", None),
        ("extra-short", None),
        (42, None),
    ],
)
def test_parse_text_verbosity(raw, expected):
    assert parse_text_verbosity(raw) == expected


@pytest.mark.parametrize(
    ("provider", "base_url", "canonical_codex", "direct_openai", "supported"),
    [
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            True,
            False,
            True,
        ),
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex/?foo=bar",
            True,
            False,
            True,
        ),
        (
            "openai-codex",
            "https://chatgpt.com/backend-api/codex-extra",
            False,
            False,
            False,
        ),
        (
            "openai-codex",
            "https://chatgpt.com/proxy/backend-api/codex",
            False,
            False,
            False,
        ),
        (
            "openai-codex",
            "https://api.x.ai/v1",
            False,
            False,
            False,
        ),
        (
            "custom",
            "https://api.openai.com/v1",
            False,
            True,
            True,
        ),
        (
            "custom",
            "https://proxy.api.openai.com/v1",
            False,
            False,
            False,
        ),
    ],
)
def test_shared_route_classifier_owns_text_verbosity_boundary(
    provider,
    base_url,
    canonical_codex,
    direct_openai,
    supported,
):
    agent = _route_agent(provider, base_url)
    route = classify_responses_route(agent)

    assert len(route) == 3
    assert supports_openai_text_verbosity_route(agent) is supported
    assert canonical_codex is (
        base_url.startswith("https://chatgpt.com/backend-api/codex")
        and supported
    )
    assert direct_openai is (
        base_url.startswith("https://api.openai.com/") and supported
    )


def test_final_dispatch_gate_rechecks_mutated_responses_model_and_route():
    canonical_agent = _route_agent(
        "openai-codex",
        "https://chatgpt.com/backend-api/codex",
    )
    unsupported_model = {
        "model": "gpt-4.1",
        "text": {"verbosity": "high", "format": {"type": "text"}},
    }
    finalize_text_verbosity_request(canonical_agent, unsupported_model)
    assert unsupported_model["text"] == {"format": {"type": "text"}}

    xai_agent = _route_agent("xai", "https://api.x.ai/v1")
    unsupported_route = {
        "model": "gpt-5.6-sol",
        "text": {"verbosity": "high", "format": {"type": "text"}},
    }
    finalize_text_verbosity_request(xai_agent, unsupported_route)
    assert unsupported_route["text"] == {"format": {"type": "text"}}


def test_final_dispatch_gate_preserves_supported_explicit_override():
    agent = _route_agent(
        "openai-codex",
        "https://chatgpt.com/backend-api/codex",
    )
    payload = {
        "model": "gpt-5.6-sol",
        "text": {"verbosity": "high", "format": {"type": "text"}},
    }

    finalize_text_verbosity_request(agent, payload)

    assert payload["text"] == {
        "verbosity": "high",
        "format": {"type": "text"},
    }


def test_final_dispatch_gate_strips_chat_completions_middleware_injection():
    agent = _route_agent(
        "openai-api",
        "https://api.openai.com/v1",
        api_mode="chat_completions",
    )
    payload = {
        "model": "gpt-5.6-sol",
        "text": {"verbosity": "high", "format": {"type": "text"}},
        "extra_body": {
            "text": {"verbosity": "low", "format": {"type": "json_schema"}}
        },
    }

    finalize_text_verbosity_request(agent, payload)

    assert payload["text"] == {"format": {"type": "text"}}
    assert payload["extra_body"]["text"] == {
        "format": {"type": "json_schema"}
    }


@pytest.mark.parametrize(
    ("model", "route_supported", "expected"),
    [
        ("gpt-5", True, True),
        ("openai/gpt-5.6-sol", True, True),
        ("gpt-5-mini", True, True),
        ("gpt-4.1", True, False),
        ("gpt-5.6-sol", False, False),
    ],
)
def test_text_verbosity_model_and_route_capability(model, route_supported, expected):
    assert (
        supports_openai_text_verbosity(
            model,
            route_supported=route_supported,
        )
        is expected
    )


@pytest.fixture
def responses_transport():
    import agent.transports.codex  # noqa: F401

    return get_transport("codex_responses")


@pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
def test_supported_responses_route_sets_and_preflights_verbosity(
    responses_transport,
    verbosity,
):
    kwargs = responses_transport.build_kwargs(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        text_verbosity=verbosity,
        text_verbosity_route_supported=True,
    )

    assert kwargs["text"] == {"verbosity": verbosity}
    assert responses_transport.preflight_kwargs(kwargs)["text"] == {
        "verbosity": verbosity
    }


def test_explicit_text_override_wins_without_losing_siblings(responses_transport):
    request_overrides = {
        "text": {
            "format": {"type": "text"},
            "verbosity": "high",
        },
        "service_tier": "priority",
    }

    kwargs = responses_transport.build_kwargs(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        text_verbosity="low",
        text_verbosity_route_supported=True,
        request_overrides=request_overrides,
    )

    assert kwargs["text"] == {
        "format": {"type": "text"},
        "verbosity": "high",
    }
    assert kwargs["service_tier"] == "priority"
    assert request_overrides["text"]["verbosity"] == "high"


def test_extra_body_text_override_wins_without_losing_siblings(responses_transport):
    kwargs = responses_transport.build_kwargs(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        text_verbosity="low",
        text_verbosity_route_supported=True,
        request_overrides={
            "extra_body": {
                "text": {
                    "format": {"type": "text"},
                    " verbosity ": "high",
                },
                "metadata": {"source": "test"},
            }
        },
    )

    assert kwargs["text"] == {
        "format": {"type": "text"},
        "verbosity": "high",
    }
    assert kwargs["extra_body"] == {"metadata": {"source": "test"}}


def test_effective_model_override_can_enable_supported_verbosity(responses_transport):
    kwargs = responses_transport.build_kwargs(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        text_verbosity="low",
        text_verbosity_route_supported=True,
        request_overrides={"model": "gpt-5.6-sol"},
    )

    assert kwargs["text"] == {"verbosity": "low"}


@pytest.mark.parametrize(
    "request_overrides",
    [
        {"model": "gpt-4.1", "text": {"verbosity": "high"}},
        {
            "extra_body": {
                "model": "gpt-4.1",
                "text": {"verbosity": "high", "format": {"type": "text"}},
            }
        },
    ],
)
def test_effective_model_override_cannot_bypass_capability(
    responses_transport,
    request_overrides,
):
    kwargs = responses_transport.build_kwargs(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        text_verbosity="low",
        text_verbosity_route_supported=True,
        request_overrides=request_overrides,
    )

    assert kwargs.get("text", {}).get("verbosity") is None


@pytest.mark.parametrize(
    "request_overrides",
    [
        {"text": {"verbosity": "high", "format": {"type": "text"}}},
        {
            "extra_body": {
                "text": {"verbosity": "high", "format": {"type": "text"}},
                "metadata": {"source": "test"},
            }
        },
    ],
)
def test_unsupported_route_strips_only_verbosity(
    responses_transport,
    request_overrides,
):
    kwargs = responses_transport.build_kwargs(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        text_verbosity="low",
        text_verbosity_route_supported=False,
        request_overrides=request_overrides,
    )

    assert kwargs.get("text", {}).get("verbosity") is None
    if "text" in kwargs:
        assert kwargs["text"]["format"] == {"type": "text"}


def test_chat_completions_cannot_send_responses_verbosity():
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="gpt-5.6-sol",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        request_overrides={
            "text": {
                "verbosity": "high",
                "format": {"type": "text"},
            },
            "extra_body": {
                "text": {
                    "verbosity": "low",
                    "format": {"type": "json_schema"},
                }
            },
        },
    )

    assert kwargs["text"] == {"format": {"type": "text"}}
    assert kwargs["extra_body"]["text"] == {
        "format": {"type": "json_schema"}
    }


@pytest.mark.parametrize("text_value", ["provider-extension", ["a", "b"], None])
def test_chat_completions_preserves_non_object_text_extensions(text_value):
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="custom-model",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        request_overrides={
            "text": text_value,
            "extra_body": {"text": text_value},
        },
    )

    assert "text" in kwargs
    assert kwargs["text"] == text_value
    assert kwargs["extra_body"]["text"] == text_value


@pytest.mark.parametrize("text_value", ["provider-extension", ["a", "b"], None])
def test_registered_provider_preserves_non_object_text_extensions(text_value):
    from providers import get_provider_profile

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="openai/gpt-4.1",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        provider_profile=get_provider_profile("openrouter"),
        request_overrides={
            "text": text_value,
            "extra_body": {"text": text_value},
        },
    )

    assert "text" in kwargs
    assert kwargs["text"] == text_value
    assert kwargs["extra_body"]["text"] == text_value


def test_preflight_normalizes_text_verbosity_and_preserves_text_format():
    payload = _preflight_codex_api_kwargs(
        {
            "model": "gpt-5.6-sol",
            "instructions": "system",
            "input": [{"role": "user", "content": "hi"}],
            "store": False,
            "text": {
                "verbosity": "HIGH",
                "format": {"type": "text"},
            },
        }
    )

    assert payload["text"] == {
        "verbosity": "high",
        "format": {"type": "text"},
    }


@pytest.mark.parametrize("verbosity", ["extra-short", 42, [], {}])
def test_preflight_rejects_malformed_explicit_verbosity(verbosity):
    with pytest.raises(
        ValueError,
        match="text.verbosity.*must be low, medium, or high",
    ):
        _preflight_codex_api_kwargs(
            {
                "model": "gpt-5.6-sol",
                "instructions": "system",
                "input": [{"role": "user", "content": "hi"}],
                "store": False,
                "text": {"verbosity": verbosity},
            }
        )


def _make_agent(text_verbosity, *, provider, model, base_url):
    config = {"agent": {"text_verbosity": text_verbosity}}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
    ):
        return AIAgent(
            model=model,
            provider=provider,
            api_key="test-key-1234567890",
            base_url=base_url,
            api_mode="codex_responses",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_real_config_file_reaches_canonical_codex_request(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "agent:\n  text_verbosity: low\n",
        encoding="utf-8",
    )
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="gpt-5.6-sol",
            provider="openai-codex",
            api_key="test-key-1234567890",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Hi"}])

    assert kwargs["text"] == {"verbosity": "low"}


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "expected"),
    [
        (
            "openai-codex",
            "gpt-5.6-sol",
            "https://chatgpt.com/backend-api/codex",
            "low",
        ),
        (
            "openai-api",
            "gpt-5.5",
            "https://api.openai.com/v1",
            "low",
        ),
        (
            "openai-codex",
            "gpt-5.6-sol",
            "https://responses.example.com/v1",
            None,
        ),
        (
            "xai",
            "gpt-5.6-sol",
            "https://api.x.ai/v1",
            None,
        ),
        (
            "openai-api",
            "gpt-4.1",
            "https://api.openai.com/v1",
            None,
        ),
    ],
)
def test_config_reaches_only_supported_responses_targets(
    provider,
    model,
    base_url,
    expected,
):
    agent = _make_agent(
        " Low ",
        provider=provider,
        model=model,
        base_url=base_url,
    )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Hi"}])

    assert kwargs.get("text", {}).get("verbosity") == expected


def test_invalid_config_warns_and_preserves_provider_default(caplog):
    agent = _make_agent(
        "extra-short",
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Hi"}])

    assert "text" not in kwargs
    assert "Invalid agent.text_verbosity" in caplog.text


@pytest.mark.parametrize("configured", ["", None])
def test_unset_config_preserves_provider_default(configured):
    agent = _make_agent(
        configured,
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "Hi"}])

    assert "text" not in kwargs

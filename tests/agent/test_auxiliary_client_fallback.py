from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as aux


@pytest.fixture(autouse=True)
def _reset_auxiliary_state():
    aux._reset_aux_unhealthy_cache()
    aux.shutdown_cached_clients()
    yield
    aux._reset_aux_unhealthy_cache()
    aux.shutdown_cached_clients()


def _client(label: str) -> MagicMock:
    client = MagicMock(name=f"{label}-client")
    client.base_url = f"https://{label}.example/v1"
    return client


def _response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ]
    )


def _payment_error() -> Exception:
    exc = Exception("Payment Required: insufficient credits")
    exc.status_code = 402  # type: ignore[attr-defined]
    return exc


def test_auto_detection_prefers_main_provider_before_fallback_chain():
    main_client = _client("main")

    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"),
        patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(main_client, "anthropic/claude-sonnet"),
        ) as resolve_main,
        patch("agent.auxiliary_client._try_openrouter") as try_openrouter,
    ):
        client, model = aux._resolve_auto()

    assert client is main_client
    assert model == "anthropic/claude-sonnet"
    resolve_main.assert_called_once()
    try_openrouter.assert_not_called()


@pytest.mark.parametrize(
    ("winning_rung", "expected_label", "expected_model", "expected_calls"),
    [
        ("openrouter", "openrouter", "or-model", ["openrouter"]),
        ("nous", "nous", "nous-model", ["openrouter", "nous"]),
        ("custom", "local/custom", "custom-model", ["openrouter", "nous", "custom"]),
        ("anthropic", "api-key", "claude-haiku", ["openrouter", "nous", "custom", "api-key"]),
        ("direct", "api-key", "glm-4.5-flash", ["openrouter", "nous", "custom", "api-key"]),
        ("none", None, None, ["openrouter", "nous", "custom", "api-key"]),
    ],
)
def test_auto_detection_chain_walks_all_provider_fallback_rungs(
    winning_rung: str,
    expected_label: str | None,
    expected_model: str | None,
    expected_calls: list[str],
):
    calls: list[str] = []

    def try_openrouter():
        calls.append("openrouter")
        if winning_rung == "openrouter":
            return _client("openrouter"), "or-model"
        return None, None

    def try_nous():
        calls.append("nous")
        if winning_rung == "nous":
            return _client("nous"), "nous-model"
        return None, None

    def try_custom():
        calls.append("custom")
        if winning_rung == "custom":
            return _client("custom"), "custom-model"
        return None, None

    def try_api_key_provider():
        calls.append("api-key")
        if winning_rung == "anthropic":
            return _client("anthropic"), "claude-haiku"
        if winning_rung == "direct":
            return _client("zai"), "glm-4.5-flash"
        return None, None

    with (
        patch("agent.auxiliary_client._read_main_provider", return_value=""),
        patch("agent.auxiliary_client._read_main_model", return_value=""),
        patch("agent.auxiliary_client._try_openrouter", side_effect=try_openrouter),
        patch("agent.auxiliary_client._try_nous", side_effect=try_nous),
        patch("agent.auxiliary_client._try_custom_endpoint", side_effect=try_custom),
        patch("agent.auxiliary_client._resolve_api_key_provider", side_effect=try_api_key_provider),
    ):
        client, model = aux._resolve_auto()

    assert calls == expected_calls
    assert model == expected_model
    if expected_label is None:
        assert client is None
    else:
        assert client is not None


def test_402_credit_exhaustion_falls_back_to_next_available_provider():
    primary = _client("openrouter")
    primary.base_url = "https://openrouter.ai/api/v1"
    primary.chat.completions.create.side_effect = _payment_error()

    fallback = _client("nous")
    fallback.chat.completions.create.return_value = _response("fallback response")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "or-model")),
        patch("agent.auxiliary_client._read_main_provider", return_value=""),
        patch("agent.auxiliary_client._try_openrouter") as try_openrouter,
        patch("agent.auxiliary_client._try_nous", return_value=(fallback, "nous-model")),
        patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)),
        patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)),
    ):
        result = aux.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summarize"}],
        )

    assert result.choices[0].message.content == "fallback response"
    assert fallback.chat.completions.create.called
    assert aux._is_provider_unhealthy("openrouter") is True
    try_openrouter.assert_not_called()


def test_provider_alias_resolution_covers_common_names():
    assert aux._normalize_aux_provider("claude") == "anthropic"
    assert aux._normalize_aux_provider("claude-code") == "anthropic"
    assert aux._normalize_aux_provider("grok") == "xai"
    assert aux._normalize_aux_provider("x.ai") == "xai"
    assert aux._normalize_aux_provider("github-models") == "copilot"


def test_explicit_vision_provider_override_skips_auto_chain():
    override_client = _client("openrouter")

    with (
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"provider": "openrouter", "model": "openrouter/vision-model"},
        ),
        patch("agent.auxiliary_client._try_openrouter", return_value=(override_client, "default-vision")),
        patch(
            "agent.auxiliary_client._read_main_provider",
            side_effect=AssertionError("auto vision chain should not inspect main provider"),
        ),
    ):
        provider, client, model = aux.resolve_vision_provider_client()

    assert provider == "openrouter"
    assert client is override_client
    assert model == "openrouter/vision-model"

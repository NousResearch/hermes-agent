"""Regression tests for OpenRouter attribution in auxiliary provider resolution."""

from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import resolve_provider_client


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _resolved_headers(**kwargs):
    mock_openai = MagicMock()
    mock_openai.return_value = MagicMock(name="openrouter-client")
    with patch("agent.auxiliary_client.OpenAI", mock_openai):
        client, _ = resolve_provider_client(**kwargs)

    assert client is not None
    return mock_openai.call_args.kwargs.get("default_headers", {})


def test_custom_openrouter_endpoint_gets_attribution_headers():
    headers = _resolved_headers(
        provider="custom",
        model="openai/gpt-4o-mini",
        explicit_api_key="test-key",
        explicit_base_url="https://openrouter.ai/api/v1",
    )

    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-Title"] == "Hermes Agent"


def test_api_key_provider_routed_to_openrouter_gets_attribution_headers():
    headers = _resolved_headers(
        provider="alibaba",
        model="qwen/qwen3-coder",
        explicit_api_key="test-key",
        explicit_base_url="https://openrouter.ai/api/v1",
    )

    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-Title"] == "Hermes Agent"


def test_named_custom_openrouter_provider_gets_attribution_headers(_isolated_config):
    (_isolated_config / "config.yaml").write_text(
        "model:\n  default: test-model\n"
        "custom_providers:\n"
        "  - name: named-openrouter\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    api_key: test-key\n",
        encoding="utf-8",
    )

    headers = _resolved_headers(
        provider="named-openrouter",
        model="openai/gpt-4o-mini",
    )

    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-Title"] == "Hermes Agent"


def test_openrouter_user_headers_override_attribution_defaults(_isolated_config):
    (_isolated_config / "config.yaml").write_text(
        "model:\n"
        "  default: test-model\n"
        "  default_headers:\n"
        "    HTTP-Referer: https://example.test/custom\n"
        "    X-Title: Custom App\n",
        encoding="utf-8",
    )

    headers = _resolved_headers(
        provider="custom",
        model="openai/gpt-4o-mini",
        explicit_api_key="test-key",
        explicit_base_url="https://openrouter.ai/api/v1",
    )

    assert headers["HTTP-Referer"] == "https://example.test/custom"
    assert headers["X-Title"] == "Custom App"

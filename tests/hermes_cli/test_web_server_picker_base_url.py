"""Tests for the dashboard model picker (`POST /api/model/set`) and its
interaction with `custom_providers`.

Regression: the endpoint used to unconditionally clear ``model.base_url``
after a picker action, on the assumption that the resolver would pick "the
provider's own default." For user-defined custom providers declared in
``custom_providers:``, the user-supplied ``base_url`` IS the provider's
own default, so clearing it makes the resolver fall back to OpenRouter
and the next agent call returns ``HTTP 401: Missing Authentication header``.

These tests pin the contract:
  * Switching model WITHIN a custom provider preserves ``model.base_url``.
  * Switching to a built-in provider (anthropic / openai / etc.) still
    clears ``model.base_url`` (existing behaviour — built-ins have hardcoded
    endpoints in the resolver registry).
"""

import yaml
import pytest


@pytest.fixture
def _client(_isolate_hermes_home):
    """FastAPI TestClient with isolated HERMES_HOME + auth header."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _write_config(home, content: str):
    """Write a config.yaml under HERMES_HOME and return its path."""
    cfg_path = home / "config.yaml"
    cfg_path.write_text(content)
    return cfg_path


def _read_model(home) -> dict:
    cfg = yaml.safe_load((home / "config.yaml").read_text()) or {}
    model = cfg.get("model")
    return model if isinstance(model, dict) else {}


class TestPickerPreservesCustomProviderBaseUrl:
    """`POST /api/model/set` with a custom-provider scope must keep base_url."""

    def test_switching_model_within_custom_provider_preserves_base_url(
        self, _client, _isolate_hermes_home
    ):
        """Picker selects a different model under the same custom provider.

        Before fix: base_url cleared → resolver falls back to OpenRouter → 401.
        After fix: base_url preserved → next agent call routes to the
        user's self-hosted LLM gateway (LiteLLM/vLLM/etc.).
        """
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        _write_config(home, (
            "custom_providers:\n"
            "- name: my-litellm\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test-custom\n"
            "  api_format: openai\n"
            "model:\n"
            "  default: gpt-4o-mini\n"
            "  provider: my-litellm\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test-custom\n"
        ))

        resp = _client.post("/api/model/set", json={
            "scope": "main",
            "provider": "my-litellm",
            "model": "claude-3-haiku",
        })
        assert resp.status_code == 200, resp.text

        model = _read_model(home)
        assert model.get("default") == "claude-3-haiku"
        assert model.get("provider") == "my-litellm"
        # The critical assertion — base_url must not be cleared.
        assert model.get("base_url") == "http://192.168.1.10:4000", (
            "Picker cleared base_url for a custom provider; this regresses "
            "the LiteLLM/vLLM/self-hosted-gateway use case."
        )

    def test_provider_name_lookup_is_case_insensitive(
        self, _client, _isolate_hermes_home
    ):
        """Custom provider names match case-insensitively (matches the
        resolver's case-insensitive lookup in providers.py)."""
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        _write_config(home, (
            "custom_providers:\n"
            "- name: My-LiteLLM\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test-custom\n"
            "model:\n"
            "  default: model-a\n"
            "  provider: My-LiteLLM\n"
            "  base_url: http://192.168.1.10:4000\n"
        ))

        # POST with lowercased provider name — should still match.
        resp = _client.post("/api/model/set", json={
            "scope": "main",
            "provider": "my-litellm",
            "model": "model-b",
        })
        assert resp.status_code == 200, resp.text
        assert _read_model(home).get("base_url") == "http://192.168.1.10:4000"


class TestPickerStillClearsBuiltInBaseUrl:
    """When switching to a built-in provider, base_url should still be
    cleared so the resolver can pick the registered default endpoint."""

    def test_switching_to_anthropic_clears_base_url(
        self, _client, _isolate_hermes_home
    ):
        """Built-in providers have hardcoded endpoints in providers.py — the
        old behaviour of clearing stale base_url is correct for them."""
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        _write_config(home, (
            "custom_providers: []\n"
            "model:\n"
            "  default: gpt-4o-mini\n"
            "  provider: openai\n"
            "  base_url: https://stale-openai-mirror.example.com/v1\n"
            "  api_key: sk-test\n"
        ))

        resp = _client.post("/api/model/set", json={
            "scope": "main",
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
        })
        assert resp.status_code == 200, resp.text

        model = _read_model(home)
        assert model.get("provider") == "anthropic"
        # Built-in provider — base_url SHOULD be cleared so the resolver
        # picks anthropic's hardcoded endpoint.
        assert model.get("base_url") in ("", None), (
            f"Built-in provider switch should clear stale base_url; got "
            f"{model.get('base_url')!r}"
        )

    def test_switching_with_empty_base_url_is_noop(
        self, _client, _isolate_hermes_home
    ):
        """If base_url is already empty, the picker leaves it that way."""
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        _write_config(home, (
            "custom_providers: []\n"
            "model:\n"
            "  default: gpt-4o\n"
            "  provider: openai\n"
            "  base_url: ''\n"
            "  api_key: sk-test\n"
        ))

        resp = _client.post("/api/model/set", json={
            "scope": "main",
            "provider": "openai",
            "model": "gpt-4o-mini",
        })
        assert resp.status_code == 200, resp.text
        assert _read_model(home).get("base_url") in ("", None)

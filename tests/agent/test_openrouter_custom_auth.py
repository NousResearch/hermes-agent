"""Auth-wiring regression tests for the OpenRouter custom-endpoint path.

These tests cover the OCR / describe OpenRouter 401 defect:

  * The OCR tool (``tools/vision_tools.py::_ocr_one_image_data_url``) calls
    ``async_call_llm(provider="openrouter", base_url=OPENROUTER_BASE_URL, ...)``
    with an explicit base_url but NO api_key. An explicit base_url forces
    ``provider="custom"`` in ``_resolve_task_provider_model``, and the custom
    branch of ``resolve_provider_client`` previously resolved the key as
    ``explicit_api_key or OPENAI_API_KEY or "no-key-required"`` — it NEVER read
    ``OPENROUTER_API_KEY`` → OpenRouter returns 401.

  * The describe path (``auxiliary.vision`` configured with
    ``provider: openrouter, base_url: openrouter.ai, api_key: none``) has the
    SAME failure: YAML parses ``none`` as the literal string "none", which is
    truthy, so it was sent verbatim as the bearer token → 401.

The fix lives in the custom branch of ``resolve_provider_client``:
  - sentinel values ("none"/"null"/"no-key-required"/...) are treated as unset,
  - when the host is openrouter.ai, OPENROUTER_API_KEY is the env fallback.

Unlike the behaviour tests in tests/tools/test_ocr_image.py (which mock
``async_call_llm`` entirely), these tests exercise the REAL auth-resolution
wiring and assert the resolved client carries a NON-empty OpenRouter key — not
"no-key-required" and not OPENAI_API_KEY.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_constants import OPENROUTER_BASE_URL


_OPENROUTER_SENTINEL_KEY = "sk-or-test-REGRESSION-KEY"
_OPENAI_DECOY_KEY = "sk-openai-DECOY-must-not-win"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start from a known-empty credential environment for each test."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def _resolved_api_key(client) -> str:
    """Extract the bearer key the resolved client will actually send."""
    raw = getattr(client, "api_key", None)
    if callable(raw):  # e.g. Azure Entra bearer provider — not expected here
        raw = raw()
    return str(raw or "")


# ---------------------------------------------------------------------------
# OCR path: explicit OpenRouter base_url + NO api_key
# ---------------------------------------------------------------------------


class TestOcrPathResolvesOpenRouterKey:
    def test_explicit_openrouter_base_no_key_resolves_env_openrouter_key(self, monkeypatch):
        """The exact OCR call shape: provider=custom, openrouter base_url, no key.

        With only OPENROUTER_API_KEY in env, the resolved client must carry that
        key — NOT "no-key-required" and NOT OPENAI_API_KEY.
        """
        from agent.auxiliary_client import resolve_provider_client

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)

        client, _model = resolve_provider_client(
            "custom",
            model="qwen/qwen3-vl-32b-instruct",
            explicit_base_url=OPENROUTER_BASE_URL,
            explicit_api_key=None,
            is_vision=True,
        )

        assert client is not None
        key = _resolved_api_key(client)
        assert key == _OPENROUTER_SENTINEL_KEY
        assert key != "no-key-required"

    def test_openrouter_key_wins_over_openai_key(self, monkeypatch):
        """When both env keys are set, OpenRouter host must use OPENROUTER_API_KEY."""
        from agent.auxiliary_client import resolve_provider_client

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)
        monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_DECOY_KEY)

        client, _model = resolve_provider_client(
            "custom",
            model="qwen/qwen3-vl-32b-instruct",
            explicit_base_url=OPENROUTER_BASE_URL,
            explicit_api_key=None,
            is_vision=True,
        )

        assert client is not None
        assert _resolved_api_key(client) == _OPENROUTER_SENTINEL_KEY

    def test_resolution_through_task_layer_carries_no_key(self):
        """_resolve_task_provider_model with explicit base_url forces custom + no key.

        This is the layer the OCR tool actually flows through: async_call_llm →
        _resolve_task_provider_model(base_url=...) → resolve_provider_client.
        Asserts base_url forces provider=custom and leaves api_key unset so the
        custom branch's env fallback (OPENROUTER_API_KEY) takes over.
        """
        from agent.auxiliary_client import _resolve_task_provider_model

        provider, _model, base_url, api_key, _mode = _resolve_task_provider_model(
            task="vision",
            provider="openrouter",
            model="qwen/qwen3-vl-32b-instruct",
            base_url=OPENROUTER_BASE_URL,
            api_key=None,
        )
        assert provider == "custom"
        assert base_url == OPENROUTER_BASE_URL
        assert not api_key  # None/empty → env fallback wins downstream


# ---------------------------------------------------------------------------
# Describe path: auxiliary.vision openrouter + api_key: "none"
# ---------------------------------------------------------------------------


class TestDescribePathResolvesOpenRouterKey:
    _VISION_CFG = {
        "auxiliary": {
            "vision": {
                "provider": "openrouter",
                "model": "qwen/qwen3-vl-32b-instruct",
                "base_url": OPENROUTER_BASE_URL,
                "api_key": "none",  # literal string "none", as YAML parses it
                "timeout": 120,
            }
        }
    }

    def test_literal_none_string_is_not_sent_as_key(self, monkeypatch):
        """api_key: none must never be sent verbatim as the bearer token."""
        from agent.auxiliary_client import resolve_provider_client

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)

        # Simulate what _resolve_task_provider_model returns for this config:
        # provider="custom", api_key="none" (the truthy sentinel bug).
        client, _model = resolve_provider_client(
            "custom",
            model="qwen/qwen3-vl-32b-instruct",
            explicit_base_url=OPENROUTER_BASE_URL,
            explicit_api_key="none",
            is_vision=True,
        )
        assert client is not None
        key = _resolved_api_key(client)
        assert key == _OPENROUTER_SENTINEL_KEY
        assert key != "none"
        assert key != "no-key-required"

    def test_describe_config_resolves_env_key_end_to_end(self, monkeypatch):
        """Full describe resolution: vision config → task layer → client.

        Drives _resolve_task_provider_model with the real auxiliary.vision
        config (api_key: none) then resolves the client, asserting the env
        OPENROUTER_API_KEY ends up on the client.
        """
        from agent import auxiliary_client as ac

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)

        with patch.object(ac, "_get_auxiliary_task_config",
                          return_value=self._VISION_CFG["auxiliary"]["vision"]):
            provider, model, base_url, api_key, _mode = ac._resolve_task_provider_model(
                task="vision",
            )
            client, _ = ac.resolve_provider_client(
                provider,
                model=model,
                explicit_base_url=base_url,
                explicit_api_key=api_key,
                is_vision=True,
            )

        assert client is not None
        key = _resolved_api_key(client)
        assert key == _OPENROUTER_SENTINEL_KEY
        assert key not in ("none", "no-key-required", "")


# ---------------------------------------------------------------------------
# Regression guard: non-OpenRouter custom hosts keep OPENAI_API_KEY behaviour
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Named custom-provider path: providers/custom_providers entry with api_key: none
# ---------------------------------------------------------------------------


class TestNamedCustomProviderResolvesOpenRouterKey:
    """The named-custom-provider branch (~L3610-3691 of auxiliary_client.py).

    This branch is reached when the requested provider name matches a
    config.yaml ``providers`` / ``custom_providers`` entry (resolved via
    ``hermes_cli.runtime_provider._get_named_custom_provider``). Before the
    hardening fix it lacked the sentinel strip the explicit_base_url branch
    has, so a named provider with ``api_key: none`` resolved ``custom_key=
    "none"`` and sent it verbatim as the bearer token → 401. After the fix the
    sentinel is stripped and, for an OpenRouter host, OPENROUTER_API_KEY wins.
    """

    def test_named_openrouter_provider_with_none_key_strips_sentinel(self, monkeypatch):
        """Named OpenRouter provider, api_key: none → OPENROUTER_API_KEY wins."""
        from agent.auxiliary_client import resolve_provider_client
        from hermes_cli import runtime_provider

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)
        monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_DECOY_KEY)

        entry = {
            "name": "my-openrouter",
            "base_url": OPENROUTER_BASE_URL,
            "api_key": "none",  # literal YAML string "none"
            "model": "qwen/qwen3-vl-32b-instruct",
        }
        monkeypatch.setattr(
            runtime_provider, "_get_named_custom_provider",
            lambda name: entry if name == "my-openrouter" else None,
        )

        client, _model = resolve_provider_client(
            "my-openrouter",
            model="qwen/qwen3-vl-32b-instruct",
            is_vision=True,
        )

        assert client is not None
        key = _resolved_api_key(client)
        assert key == _OPENROUTER_SENTINEL_KEY
        assert key != "none"
        assert key != "no-key-required"
        assert key != _OPENAI_DECOY_KEY  # OpenAI key must NOT win for OR host

    def test_named_non_openrouter_provider_does_not_get_openrouter_key(self, monkeypatch):
        """Named non-OpenRouter provider must NOT pick up OPENROUTER_API_KEY.

        With api_key: none stripped and no key_env, a non-OpenRouter named host
        falls through to the "no-key-required" placeholder — it must never
        receive OPENROUTER_API_KEY (or be sent the literal "none").
        """
        from agent.auxiliary_client import resolve_provider_client
        from hermes_cli import runtime_provider

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)

        entry = {
            "name": "my-local",
            "base_url": "http://localhost:1234/v1",
            "api_key": "none",
            "model": "local-model",
        }
        monkeypatch.setattr(
            runtime_provider, "_get_named_custom_provider",
            lambda name: entry if name == "my-local" else None,
        )

        client, _model = resolve_provider_client(
            "my-local",
            model="local-model",
        )

        assert client is not None
        key = _resolved_api_key(client)
        assert key != _OPENROUTER_SENTINEL_KEY  # OR key must NOT leak here
        assert key != "none"
        assert key == "no-key-required"


class TestNonOpenRouterUnaffected:
    def test_non_openrouter_custom_still_uses_openai_key(self, monkeypatch):
        """A non-OpenRouter custom host must NOT pick up OPENROUTER_API_KEY."""
        from agent.auxiliary_client import resolve_provider_client

        monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_SENTINEL_KEY)
        monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_DECOY_KEY)

        client, _ = resolve_provider_client(
            "custom",
            model="gpt-4o-mini",
            explicit_base_url="https://api.openai.com/v1",
            explicit_api_key=None,
        )
        assert client is not None
        assert _resolved_api_key(client) == _OPENAI_DECOY_KEY

    def test_local_server_with_no_keys_falls_back_to_no_key_required(self, monkeypatch):
        """No env keys + non-OpenRouter local host → 'no-key-required' sentinel."""
        from agent.auxiliary_client import resolve_provider_client

        client, _ = resolve_provider_client(
            "custom",
            model="local-model",
            explicit_base_url="http://localhost:1234/v1",
            explicit_api_key=None,
        )
        assert client is not None
        assert _resolved_api_key(client) == "no-key-required"

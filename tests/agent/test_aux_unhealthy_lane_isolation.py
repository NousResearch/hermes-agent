"""Lane isolation of the shared provider-unhealthy cache (#133196).

The unhealthy cache is written by auxiliary tasks (402 quarantines, missing
credentials, fallback candidate dead-ends) but READ on every provider-route
resolution — including the main session's own startup fallback
(``agent_init._routed_client_kwargs`` → ``resolve_provider_client("auto")``,
whose auto-route consults the same key space). One auxiliary task's failure
marker therefore hid the provider from the main conversation's routing and
its ``fallback_providers`` for the whole TTL: the reported symptom was
``No LLM provider configured`` on a freshly-launched agent.

Main-route resolution (provider resolution for the main conversation, not an
auxiliary task) must ignore these auxiliary-failure markers; auxiliary reads
must keep skipping unhealthy providers.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean (mirrors
    tests/agent/test_auxiliary_client.py's autouse fixture)."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


class TestAuxUnhealthyCacheLaneIsolation:
    """Lane isolation: aux-written unhealthy markers must not shape main-route
    provider resolution, and aux reads must keep honoring them."""

    def setup_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def teardown_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def test_aux_marked_provider_still_serves_main_route_resolution(self):
        """A 402 quarantine written by an auxiliary task must not hide the
        provider from main-route resolution (the main session's init shape:
        resolve under the main-lane opt-out): the main session launches even
        though aux marked the provider unhealthy within the TTL."""
        from agent.auxiliary_client import (
            _MainLaneOptOut,
            _mark_provider_unhealthy,
            resolve_provider_client,
        )

        main_client = MagicMock(name="main-client")
        _mark_provider_unhealthy("openrouter")  # default TTL: 10 minutes
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter"
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="main/model"
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=lambda provider, model=None, **kw: (
                (main_client, model or "main/model") if provider == "openrouter" else (None, None)
            ),
        ):
            with _MainLaneOptOut():
                client, _model = resolve_provider_client("auto", raw_codex=True)
        assert client is main_client

    def test_aux_marked_provider_skipped_for_auxiliary_task(self):
        """Auxiliary task reads are unchanged: a provider marked unhealthy by an
        aux failure is still skipped on the aux task route (the isolation must
        not un-skip it there)."""
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _resolve_auto_route,
        )

        aux_client = MagicMock(name="aux-fallback-client")
        resolved_providers = []

        def fake_resolve(provider, model=None, **kw):
            resolved_providers.append(provider)
            return aux_client, model or "fallback/model"

        _mark_provider_unhealthy("local/custom")
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter"
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="main/model"
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=fake_resolve,
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=[
                {"provider": "custom", "model": "qwen3.5:4b", "base_url": "http://127.0.0.1:11434/v1"},
                {"provider": "nous", "model": "aux-model"},
            ],
        ), patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", return_value=""
        ):
            client, _model, _label = _resolve_auto_route(main_runtime=None, task="session_search")
        assert client is aux_client
        # The quarantined entry was skipped without a resolve attempt.
        assert "custom" not in resolved_providers

    def test_fallback_providers_entry_marked_by_aux_still_serves_main_route(self):
        """A ``fallback_providers`` entry quarantined by an auxiliary failure must
        still resolve for the main session's init fallback (its Step-2 chain runs
        the same auto-route), instead of dying at init with
        ``No LLM provider configured``."""
        from agent.auxiliary_client import (
            _MainLaneOptOut,
            _mark_provider_unhealthy,
            resolve_provider_client,
        )

        fb_client = MagicMock(name="fallback-client")
        _mark_provider_unhealthy("local/custom")
        with patch(
            "agent.auxiliary_client._read_main_provider", return_value="openrouter"
        ), patch(
            "agent.auxiliary_client._read_main_model", return_value="main/model"
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=lambda provider, model=None, **kw: (
                (fb_client, model or "fb/model") if provider == "custom" else (None, None)
            ),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")
        ), patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=[
                {"provider": "custom", "model": "qwen3.5:4b", "base_url": "http://127.0.0.1:11434/v1"},
            ],
        ), patch(
            "hermes_cli.config.load_config_readonly", return_value={}
        ), patch(
            "agent.auxiliary_client._custom_health_base_url", return_value=""
        ):
            with _MainLaneOptOut():
                client, _model = resolve_provider_client("auto", raw_codex=True)
        assert client is fb_client

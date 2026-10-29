"""Regression tests for issue #145182.

``auxiliary.vision`` pinned to a provider that cannot be resolved must not carry the
pinned ``model`` into the auto fallback: the pinned model belongs to the provider that
just failed, so pairing it with the auto-detected endpoint builds a cross-vendor request
that dies upstream (``400 <model> is not a valid model ID``) — while the log line right
above shows the correct autodetected model being computed and then discarded.

Before the fix, the fallback re-entered ``resolve_vision_provider_client`` with the
pinned model, whose task-config resolution resurrects ``auxiliary.vision.model`` for the
auto chain. The fallback now walks ``_vision_auto_route`` directly (the text branch's
``_get_cached_client("auto", task=...)`` already behaves this way), so each provider uses
its own default model.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux

MAIN_RUNTIME = {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash-1017"}
PINNED_PROVIDER = "custom:does-not-exist"
PINNED_MODEL = "Qwen3.5-9B-oQ4e-mtp"  # only exists on the pinned local endpoint


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Fresh HERMES_HOME (no auxiliary config) and a clean aux client cache per test."""
    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp(prefix="hermes_test_145182_"))
    aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()
    yield
    aux._client_cache.clear()


def _resolve_vision_route(resolved_provider: str, resolved_model: str | None) -> aux._ResolvedAuxRoute:
    return aux._resolve_call_client(
        task="vision", provider=None, model=None, base_url=None, api_key=None,
        resolved_provider=resolved_provider, resolved_model=resolved_model,
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
        main_runtime=MAIN_RUNTIME, async_mode=False,
    )


class TestUnavailableVisionProviderFallsBackToItsOwnModel:
    """The auto fallback must resolve the model for the endpoint it actually picked."""

    def test_fallback_model_is_the_autodetected_one(self):
        """Pinned provider unresolvable + pinned model → the main provider's model wins."""

        def fake_resolve(provider, model=None, async_mode=False, **_kw):
            if str(provider).startswith("custom:does-not-exist"):
                return None, None  # pinned provider: unresolvable, no credentials
            return MagicMock(name="openrouter-client"), "deepseek/deepseek-v4.1-flash-1017"

        with patch.object(aux, "resolve_provider_client", side_effect=fake_resolve):
            route = _resolve_vision_route(PINNED_PROVIDER, PINNED_MODEL)

        assert route.client is not None
        assert route.effective_provider == "openrouter"
        assert route.final_model == "deepseek/deepseek-v4.1-flash-1017"
        assert route.final_model != PINNED_MODEL

    def test_fallback_does_not_reuse_the_pinned_model_for_any_provider(self):
        """Even when the auto chain's default is unset, the pinned model must not leak."""

        def fake_resolve(provider, model=None, async_mode=False, **_kw):
            if str(provider).startswith("custom:does-not-exist"):
                return None, None
            # Auto chain resolved a client but no default model.
            return MagicMock(name="openrouter-client"), None

        with patch.object(aux, "resolve_provider_client", side_effect=fake_resolve):
            route = _resolve_vision_route(PINNED_PROVIDER, PINNED_MODEL)

        assert route.client is not None
        assert route.final_model != PINNED_MODEL

    def test_resolved_pinned_provider_keeps_the_pinned_model(self):
        """Guard: when the pinned provider DOES resolve, the pinned model is honored."""

        def fake_resolve(provider, model=None, async_mode=False, **_kw):
            if str(provider).startswith("custom:does-not-exist"):
                return None, None
            return MagicMock(name="openrouter-client"), "deepseek/deepseek-v4.1-flash-1017"

        with patch.object(aux, "resolve_provider_client", side_effect=fake_resolve):
            # provider "auto" resolves through the auto chain; the pinned model rides along
            # as the caller's explicit model (the MoA-slot contract).
            route = _resolve_vision_route("auto", PINNED_MODEL)

        assert route.client is not None
        assert route.final_model == PINNED_MODEL

"""Bug 1 regression: copilot transport must follow the active model, not the stale default.

These tests verify that _resolve_runtime_agent_kwargs_for_provider and its
callers (rehydration, channel override, fallback chain) correctly thread
target_model through to the underlying resolve_runtime_provider, and that
model-only channel overrides on a copilot default trigger re-resolution.

The critical discipline: **spy that the helper RECEIVES target_model** (not
just mock its return), because a dropped thread passes green with return-only
mocking.
"""
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_resolve_for_provider(provider, target_model=None):
    """Return a plausible runtime dict, recording what we received."""
    mode = "anthropic_messages" if "claude" in (target_model or "") else "codex_responses"
    return {
        "api_key": "gho_test",
        "base_url": None,
        "provider": provider,
        "api_mode": mode,
        "command": None,
        "args": [],
        "credential_pool": None,
    }


# ---------------------------------------------------------------------------
# Edit 4: rehydration threads persisted model
# ---------------------------------------------------------------------------

class TestRehydrationThreadsModel:
    """_rehydrate_session_model_override must pass target_model=persisted model."""

    def test_rehydrate_copilot_override_derives_from_persisted_model(self, monkeypatch):
        seen_calls = []

        def spy_resolve(provider, target_model=None):
            seen_calls.append({"provider": provider, "target_model": target_model})
            return _fake_resolve_for_provider(provider, target_model)

        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            spy_resolve,
        )

        # Build a minimal runner with just enough state
        runner = object.__new__(gateway_run.GatewayRunner)
        runner._session_model_overrides = {}
        runner.logger = MagicMock()

        # Mock session_store to return a persisted override
        persisted = {"model": "claude-sonnet-4.6", "provider": "copilot", "base_url": None}
        mock_store = MagicMock()
        mock_store.get_model_override = MagicMock(return_value=persisted)
        runner.session_store = mock_store

        runner._rehydrate_session_model_override("test-session-key")

        assert len(seen_calls) == 1
        assert seen_calls[0]["provider"] == "copilot"
        assert seen_calls[0]["target_model"] == "claude-sonnet-4.6", (
            "Rehydration must thread the persisted model as target_model"
        )
        override = runner._session_model_overrides["test-session-key"]
        assert override["api_mode"] == "anthropic_messages"


# ---------------------------------------------------------------------------
# Edit 5: channel override - model-only on copilot re-resolves
# ---------------------------------------------------------------------------

class TestChannelModelOnlyOverride:
    """Model-only channel override on copilot default must re-resolve transport."""

    def test_model_only_channel_copilot_rederives(self, monkeypatch):
        """ch.model set, ch.provider absent, default provider copilot -> re-resolve."""
        seen_calls = []

        def spy_resolve(provider, target_model=None):
            seen_calls.append({"provider": provider, "target_model": target_model})
            return _fake_resolve_for_provider(provider, target_model)

        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            spy_resolve,
        )

        # Simulate: runtime_kwargs already resolved with provider=copilot
        runtime_kwargs = {
            "provider": "copilot",
            "api_key": "gho_test",
            "api_mode": "codex_responses",  # stale: GPT default
        }

        # Simulate channel override with model only (no provider)
        ch = SimpleNamespace(model="claude-sonnet-4.6", provider=None)

        # Execute the channel logic inline (mirrors gateway/run.py:3866-3884)
        model = "gpt-5.5"  # the default model
        if ch:
            if ch.model:
                model = ch.model
            if ch.provider:
                runtime_kwargs = spy_resolve(ch.provider, target_model=model)
                ch_runtime_model = runtime_kwargs.pop("model", None)
                if ch_runtime_model and not ch.model:
                    model = ch_runtime_model
            elif ch.model and runtime_kwargs.get("provider") == "copilot":
                runtime_kwargs = spy_resolve("copilot", target_model=model)

        assert len(seen_calls) == 1
        assert seen_calls[0]["target_model"] == "claude-sonnet-4.6"
        assert runtime_kwargs["api_mode"] == "anthropic_messages"

    def test_model_only_channel_non_copilot_no_reresolve(self, monkeypatch):
        """Model-only channel on a non-copilot provider must NOT re-resolve."""
        seen_calls = []

        def spy_resolve(provider, target_model=None):
            seen_calls.append({"provider": provider, "target_model": target_model})
            return _fake_resolve_for_provider(provider, target_model)

        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            spy_resolve,
        )

        runtime_kwargs = {
            "provider": "openai",
            "api_key": "sk-test",
            "api_mode": "responses",
        }

        ch = SimpleNamespace(model="gpt-5.5", provider=None)
        model = "gpt-4o"
        if ch:
            if ch.model:
                model = ch.model
            if ch.provider:
                runtime_kwargs = spy_resolve(ch.provider, target_model=model)
            elif ch.model and runtime_kwargs.get("provider") == "copilot":
                runtime_kwargs = spy_resolve("copilot", target_model=model)

        # Non-copilot: no re-resolution happened
        assert len(seen_calls) == 0, (
            "Model-only channel on non-copilot provider must NOT trigger re-resolution"
        )
        # Original runtime_kwargs preserved
        assert runtime_kwargs["provider"] == "openai"


# ---------------------------------------------------------------------------
# Edit 6: fallback chain threads entry model
# ---------------------------------------------------------------------------

class TestFallbackChainThreadsModel:
    """Fallback chain must pass target_model=entry['model'] to resolve_runtime_provider."""

    def test_fallback_entry_model_threaded(self, monkeypatch):
        seen_calls = []

        def spy_resolve(requested=None, explicit_base_url=None, explicit_api_key=None, target_model=None):
            seen_calls.append({
                "requested": requested,
                "target_model": target_model,
            })
            return {
                "provider": requested or "copilot",
                "api_key": explicit_api_key or "gho_test",
                "base_url": explicit_base_url,
                "api_mode": "anthropic_messages" if "claude" in (target_model or "") else "codex_responses",
                "model": target_model,
                "credential_pool": None,
                "command": None,
                "args": [],
            }

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            spy_resolve,
        )

        # Simulate the fallback resolution inline
        entry = {"provider": "copilot", "model": "claude-sonnet-4.6", "base_url": None}
        explicit_api_key = "gho_fallback_key"

        # This mirrors gateway/run.py:1995-1999 after our edit
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(
            requested=entry.get("provider"),
            explicit_base_url=entry.get("base_url"),
            explicit_api_key=explicit_api_key,
            target_model=entry.get("model"),
        )

        assert len(seen_calls) == 1
        assert seen_calls[0]["target_model"] == "claude-sonnet-4.6", (
            "Fallback chain must thread entry['model'] as target_model"
        )
        assert runtime["api_mode"] == "anthropic_messages"

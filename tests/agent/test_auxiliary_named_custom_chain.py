"""Regression tests for ``custom_providers[]`` enumeration in the auto chain.

Before this change, ``_resolve_auto``'s fallback chain only checked the
single-slot legacy custom endpoint (``model.base_url + OPENAI_API_KEY``).
Users who declared their endpoint exclusively as a named entry under
``custom_providers:`` saw the misleading "No auxiliary LLM provider
configured" warning even though a working entry was right there in the
config — they had to point each ``auxiliary.<task>.provider`` at the named
custom provider explicitly to wire it up.

The fix adds a ``_try_named_custom_providers`` step between Nous and the
legacy ``local/custom`` step, enumerating each ``custom_providers[]`` entry
and trying it via the same ``custom:<name>`` router used by explicit config.

Related: gh-13762, gh-22317.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear known env vars that would short-circuit."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Strip env vars that would let earlier chain steps (OpenRouter / custom
    # endpoint) succeed and mask the named-custom step under test.
    for var in ("OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


# ── Provider-chain composition ──────────────────────────────────────────────


class TestChainShape:
    """The chain ordering must remain stable for callers that rely on it."""

    def test_named_custom_step_present(self):
        from agent.auxiliary_client import _get_provider_chain
        labels = [label for label, _ in _get_provider_chain()]
        assert "named-custom" in labels

    def test_named_custom_runs_before_legacy_custom(self):
        from agent.auxiliary_client import _get_provider_chain
        labels = [label for label, _ in _get_provider_chain()]
        assert labels.index("named-custom") < labels.index("local/custom")

    def test_named_custom_runs_after_aggregators(self):
        """Aggregators (OpenRouter / Nous) keep their priority — they're the
        common case and a named custom provider should only be tried when
        those have no credentials."""
        from agent.auxiliary_client import _get_provider_chain
        labels = [label for label, _ in _get_provider_chain()]
        assert labels.index("openrouter") < labels.index("named-custom")
        assert labels.index("nous") < labels.index("named-custom")


# ── _try_named_custom_providers behavior ───────────────────────────────────


class TestTryNamedCustomProviders:
    """The new step iterates ``custom_providers[]`` and returns the first hit."""

    def test_returns_none_when_no_custom_providers(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "claude-opus-4-7", "provider": "anthropic"}})
        from agent.auxiliary_client import _try_named_custom_providers
        client, model = _try_named_custom_providers()
        assert client is None
        assert model is None

    def test_returns_first_working_entry(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "claude-opus-4-7", "provider": "anthropic"},
            "custom_providers": [
                {"name": "azure-tunnel", "base_url": "http://localhost:8787/v1",
                 "api_key": "dummy", "model": "gpt-5.4"},
            ],
        })
        mock_client = MagicMock()
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "gpt-5.4"),
        ) as mock_resolve:
            from agent.auxiliary_client import _try_named_custom_providers
            client, model = _try_named_custom_providers()

        assert client is mock_client
        assert model == "gpt-5.4"
        # Verify the router was called with the custom:<name> form
        called_provider = mock_resolve.call_args.args[0]
        assert called_provider == "custom:azure-tunnel"

    def test_skips_main_provider_to_avoid_double_attempt(self, tmp_path):
        """If the main provider is custom:foo, we already tried it in Step 1;
        re-trying it in the chain is wasted work."""
        _write_config(tmp_path, {
            "model": {"default": "gpt-5.4", "provider": "custom:azure-tunnel"},
            "custom_providers": [
                {"name": "azure-tunnel", "base_url": "http://localhost:8787/v1",
                 "api_key": "dummy", "model": "gpt-5.4"},
                {"name": "other-tunnel", "base_url": "http://localhost:9999/v1",
                 "api_key": "dummy", "model": "other-model"},
            ],
        })
        mock_client = MagicMock()
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "other-model"),
        ) as mock_resolve:
            from agent.auxiliary_client import _try_named_custom_providers
            client, model = _try_named_custom_providers()

        # Only "other-tunnel" should have been attempted
        assert client is mock_client
        called_providers = [c.args[0] for c in mock_resolve.call_args_list]
        assert called_providers == ["custom:other-tunnel"]

    def test_falls_through_when_entry_returns_no_client(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "claude-opus-4-7", "provider": "anthropic"},
            "custom_providers": [
                {"name": "broken", "base_url": "http://localhost:1/v1"},
                {"name": "working", "base_url": "http://localhost:8787/v1",
                 "api_key": "dummy", "model": "gpt-5.4"},
            ],
        })
        mock_client = MagicMock()
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(None, None), (mock_client, "gpt-5.4")],
        ) as mock_resolve:
            from agent.auxiliary_client import _try_named_custom_providers
            client, model = _try_named_custom_providers()

        assert client is mock_client
        assert model == "gpt-5.4"
        assert mock_resolve.call_count == 2

    def test_skips_malformed_entries(self, tmp_path):
        """Non-dict / nameless entries are silently skipped."""
        _write_config(tmp_path, {
            "model": {"default": "claude-opus-4-7", "provider": "anthropic"},
            "custom_providers": [
                "bogus-string-entry",          # not a dict
                {"base_url": "http://x"},      # missing name
                {"name": "real", "base_url": "http://localhost:8787/v1",
                 "api_key": "dummy", "model": "gpt-5.4"},
            ],
        })
        mock_client = MagicMock()
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, "gpt-5.4"),
        ) as mock_resolve:
            from agent.auxiliary_client import _try_named_custom_providers
            client, model = _try_named_custom_providers()

        assert client is mock_client
        assert mock_resolve.call_count == 1
        assert mock_resolve.call_args.args[0] == "custom:real"


# ── End-to-end: _resolve_auto picks up the named-custom entry ──────────────


class TestResolveAutoUsesNamedCustom:
    """The bug from the issue: main provider has no aux-usable client and the
    only configured endpoint is in custom_providers[]. The chain must reach it."""

    def test_anthropic_oauth_main_with_named_custom_fallback(self, tmp_path):
        """Reproduces the original report: main=anthropic (OAuth-only on Claude
        Code, no API key), custom_providers has a working tunnel — auto must
        find it instead of warning 'No auxiliary LLM provider configured'."""
        _write_config(tmp_path, {
            "model": {"default": "claude-opus-4-7", "provider": "anthropic"},
            "custom_providers": [
                {"name": "azure-tunnel", "base_url": "http://localhost:8787/v1",
                 "api_key": "dummy", "model": "gpt-5.4"},
            ],
        })
        named_client = MagicMock()

        def fake_resolve(provider, model=None, **kwargs):
            if provider == "custom:azure-tunnel":
                return named_client, "gpt-5.4"
            # Step 1 (main provider) returns no client — Anthropic OAuth-only
            return None, None

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=fake_resolve,
        ):
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()

        assert client is named_client
        assert model == "gpt-5.4"

    def test_main_model_still_wins_when_resolvable(self, tmp_path):
        """Step 1 (main provider) must keep priority over the chain when its
        client is available — the chain is a fallback only."""
        _write_config(tmp_path, {
            "model": {"default": "deepseek-chat", "provider": "deepseek"},
            "custom_providers": [
                {"name": "azure-tunnel", "base_url": "http://localhost:8787/v1",
                 "api_key": "dummy", "model": "gpt-5.4"},
            ],
        })
        main_client = MagicMock()

        def fake_resolve(provider, model=None, **kwargs):
            if provider == "deepseek":
                return main_client, "deepseek-chat"
            return None, None  # would be hit if the test fails

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=fake_resolve,
        ):
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()

        assert client is main_client
        assert model == "deepseek-chat"

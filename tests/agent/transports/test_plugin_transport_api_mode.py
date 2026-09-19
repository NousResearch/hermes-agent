"""A provider plugin's own transport must survive every api_mode gate.

A transport is selected by its ``api_mode`` string, and ``register_transport``
is the public seam a provider plugin uses to ship a dialect of its own. Three
separate sites validated that string against a closed literal — the agent init
ladder, the runtime-provider config gate, and the delegation resolver — so a
plugin's mode was rewritten to ``chat_completions`` at each one. The dialect
translation then never ran and the provider degraded to prose-only answers with
no error raised: a working transport looked like a chat model.

These tests pin the contract "a registered transport is accepted" at each gate,
and that an unregistered value is still rejected (the sets must not become open
doors).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.transports import register_transport, registered_api_modes
from agent.agent_init import _EXPLICIT_API_MODES, _has_registered_transport


class _FakeTransport:
    """Stands in for a provider plugin's transport class."""


@pytest.fixture
def plugin_mode():
    """A registered, plugin-supplied api_mode, cleaned up after the test."""
    from agent.transports import _REGISTRY

    mode = "testplugin_functions"
    assert mode not in _EXPLICIT_API_MODES, "fixture mode must not be a core literal"
    register_transport(mode, _FakeTransport)
    yield mode
    _REGISTRY.pop(mode, None)


class TestRegistryReportsRegisteredModes:
    def test_in_tree_modes_are_present(self):
        """Discovery runs before the read, so a cold registry is never empty."""
        modes = registered_api_modes()
        assert "chat_completions" in modes

    def test_plugin_mode_appears_after_registration(self, plugin_mode):
        assert plugin_mode in registered_api_modes()

    def test_unregistered_mode_is_absent(self):
        assert "no_such_transport_mode" not in registered_api_modes()


class TestAgentInitGate:
    def test_registered_plugin_mode_is_kept(self, plugin_mode):
        assert _has_registered_transport(plugin_mode) is True

    def test_unknown_mode_is_rejected(self):
        assert _has_registered_transport("no_such_transport_mode") is False

    def test_empty_mode_is_rejected(self):
        assert _has_registered_transport("") is False


class TestRuntimeProviderGate:
    def test_registered_plugin_mode_survives(self, plugin_mode):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode(plugin_mode) == plugin_mode

    def test_unknown_mode_still_rejected(self):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode("no_such_transport_mode") is None

    def test_legacy_alias_still_canonicalized(self):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode("openai") == "chat_completions"


class TestPluginProfileTransportResolution:
    """``get_provider``/``determine_api_mode`` must not degrade a plugin dialect.

    A plugin profile declares its own ``api_mode`` ("<name>_functions"), which the
    reverse ``TRANSPORT_TO_API_MODE`` lookup has no entry for. Falling back to
    ``openai_chat`` there made ``determine_api_mode`` report ``chat_completions``
    for the provider, so the agent never selected the plugin's transport and the
    dialect translation silently never ran.
    """

    def test_plugin_transport_is_its_own_api_mode(self, plugin_mode):
        """The reverse lookup must not swallow a mode the registry knows."""
        from hermes_cli.providers import TRANSPORT_TO_API_MODE, determine_api_mode

        assert plugin_mode not in TRANSPORT_TO_API_MODE
        # A registered mode survives; an unregistered one still degrades.
        assert plugin_mode in registered_api_modes()

    def test_unknown_transport_still_degrades_to_chat_completions(self):
        from hermes_cli.providers import TRANSPORT_TO_API_MODE

        # The property the fix relies on: only REGISTERED modes pass through.
        assert "no_such_transport_mode" not in registered_api_modes()
        assert TRANSPORT_TO_API_MODE.get("no_such_transport_mode") is None


class TestPluginProfileSurvivesEarlyDiscovery:
    """A profile registered LAST in discovery must still reach the registry.

    ``hermes_cli.config`` triggers provider discovery while it is still importing,
    so the profile list seen at that moment is partial. The registration pass in
    ``hermes_cli.auth`` ran on that partial list, and whichever profile happened
    to sort last (live: a user plugin after the built-ins) never landed in
    ``PROVIDER_REGISTRY`` — every consumer then reported it unauthenticated, which
    removed it from the model picker with no error raised anywhere.
    """

    def test_ensure_registers_a_profile_added_later(self, monkeypatch):
        from hermes_cli import auth

        seen: list[str] = []

        class _Late:
            name = "late_plugin_provider"
            auth_type = "api_key"
            env_vars = ("LATE_PLUGIN_KEY",)
            aliases = ()
            base_url = "https://late.example.invalid/v1"
            display_name = "Late"

        def _fake_list_providers():
            seen.append("called")
            return [_Late()]

        monkeypatch.setattr("providers.list_providers", _fake_list_providers, raising=False)
        auth.PROVIDER_REGISTRY.pop("late_plugin_provider", None)

        # One explicit pass must be enough to pick it up.
        auth.ensure_plugin_providers_registered()

        assert "late_plugin_provider" in auth.PROVIDER_REGISTRY
        assert seen, "the pass must actually enumerate providers"

    def test_repeated_calls_are_harmless(self, monkeypatch):
        from hermes_cli import auth

        monkeypatch.setattr("providers.list_providers", lambda: [], raising=False)
        auth.ensure_plugin_providers_registered()
        auth.ensure_plugin_providers_registered()


class TestStreamDeltaSeam:
    """The response-side twin of ``convert_messages``: a transport may own the
    shape of a streamed tool call, and the default must not alter the delta."""

    def test_default_transport_passes_the_delta_through(self):
        from agent.transports.base import ProviderTransport

        sentinel = SimpleNamespace(tool_calls=None, content="hi")
        # The ABC's default is the identity — text and tool_calls both survive.
        assert ProviderTransport.normalize_stream_delta(self, sentinel) is sentinel

    def test_chat_completions_delta_is_unaffected(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        delta = SimpleNamespace(content="hi", tool_calls=[SimpleNamespace()])
        assert ChatCompletionsTransport().normalize_stream_delta(delta) is delta

"""Unit tests for the CommandCode provider profiles.

CommandCode registers two profiles:

``commandcode``
    ``api_mode=chat_completions`` — OpenAI-compatible.  Defaults to
    ``deepseek/deepseek-v4-pro``. 20+ models via a single base URL.

``commandcode-anthropic``
    ``api_mode=anthropic_messages`` — Anthropic Messages API-compatible.
    Defaults to ``claude-sonnet-4-6``.  Requires Bearer auth recognition
    in ``agent/anthropic_adapter.py``.

Both share ``COMMANDCODE_API_KEY`` and ``https://api.commandcode.ai/provider/v1``.
"""

from __future__ import annotations

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def commandcode_profile():
    """Resolve the registered CommandCode (chat_completions) profile."""
    import model_tools  # noqa: F401 — triggers discovery
    import providers

    profile = providers.get_provider_profile("commandcode")
    assert profile is not None, "commandcode provider profile must be registered"
    return profile


@pytest.fixture
def commandcode_anthropic_profile():
    """Resolve the registered CommandCode Anthropic profile."""
    import model_tools  # noqa: F401 — triggers discovery
    import providers

    profile = providers.get_provider_profile("commandcode-anthropic")
    assert profile is not None, "commandcode-anthropic profile must be registered"
    return profile


# ── Chat Completions profile ──────────────────────────────────────────────────

class TestCommandCodeProfileIdentity:
    """Profile metadata matches the declared contract."""

    def test_name(self, commandcode_profile):
        assert commandcode_profile.name == "commandcode"

    def test_api_mode(self, commandcode_profile):
        assert commandcode_profile.api_mode == "chat_completions"

    def test_aliases(self, commandcode_profile):
        assert "commandcode-chat" in commandcode_profile.aliases

    def test_env_vars(self, commandcode_profile):
        assert "COMMANDCODE_API_KEY" in commandcode_profile.env_vars

    def test_base_url(self, commandcode_profile):
        assert commandcode_profile.base_url == "https://api.commandcode.ai/provider/v1"

    def test_display_name(self, commandcode_profile):
        assert "CommandCode" in commandcode_profile.display_name

    def test_has_fallback_models(self, commandcode_profile):
        assert len(commandcode_profile.fallback_models) >= 5
        # Should include the major families
        names = " ".join(commandcode_profile.fallback_models)
        assert "deepseek" in names
        assert "Qwen" in names
        assert "Kimi" in names
        assert "gemini" in names

    def test_default_aux_model(self, commandcode_profile):
        assert commandcode_profile.default_aux_model == "deepseek/deepseek-v4-flash"

    def test_signup_url(self, commandcode_profile):
        assert "commandcode" in commandcode_profile.signup_url.lower()

    def test_hostname_derived_from_base_url(self, commandcode_profile):
        assert commandcode_profile.get_hostname() == "api.commandcode.ai"


class TestCommandCodeProfileNoThinkingInterference:
    """Chat completions profile is a no-op for thinking config — it delegates
    to the underlying model's provider (DeepSeek, Qwen, etc.) for wire format.
    """

    def test_passthrough_no_reasoning_config(self, commandcode_profile):
        extra_body, top_level = commandcode_profile.build_api_kwargs_extras(
            reasoning_config=None, model="deepseek/deepseek-v4-pro"
        )
        # Chat completions profile doesn't inject thinking params — that's
        # the DeepSeek provider's job when routed through DeepSeek's own profile.
        # When routed through CommandCode, the underlying model API handles it.
        assert isinstance(extra_body, dict)
        assert isinstance(top_level, dict)
        # Default ProviderProfile returns ({}, {}).

    def test_passthrough_with_reasoning_config(self, commandcode_profile):
        extra_body, top_level = commandcode_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="deepseek/deepseek-v4-pro",
        )
        assert isinstance(extra_body, dict)
        assert isinstance(top_level, dict)


# ── Anthropic Messages profile ────────────────────────────────────────────────

class TestCommandCodeAnthropicProfileIdentity:
    """Anthropic-compatible profile metadata."""

    def test_name(self, commandcode_anthropic_profile):
        assert commandcode_anthropic_profile.name == "commandcode-anthropic"

    def test_api_mode(self, commandcode_anthropic_profile):
        assert commandcode_anthropic_profile.api_mode == "anthropic_messages"

    def test_aliases(self, commandcode_anthropic_profile):
        assert "commandcode-claude" in commandcode_anthropic_profile.aliases

    def test_env_vars(self, commandcode_anthropic_profile):
        assert "COMMANDCODE_API_KEY" in commandcode_anthropic_profile.env_vars

    def test_base_url(self, commandcode_anthropic_profile):
        assert commandcode_anthropic_profile.base_url == "https://api.commandcode.ai/provider/v1"

    def test_fallback_models_are_claude_family(self, commandcode_anthropic_profile):
        for model in commandcode_anthropic_profile.fallback_models:
            assert model.startswith("claude-"), (
                f"All anthropic fallback models should be claude-*: got {model}"
            )

    def test_default_aux_model(self, commandcode_anthropic_profile):
        assert commandcode_anthropic_profile.default_aux_model == "claude-haiku-4-5-20251001"

    def test_display_name_distinct_from_chat(self, commandcode_anthropic_profile):
        # The Anthropic profile should be distinguishable in /model picker
        assert "(Anthropic)" in commandcode_anthropic_profile.display_name

    def test_hostname_derived_from_base_url(self, commandcode_anthropic_profile):
        assert commandcode_anthropic_profile.get_hostname() == "api.commandcode.ai"


# ── Bearer Auth Recognition ───────────────────────────────────────────────────

class TestCommandCodeAnthropicBearerAuth:
    """``agent/anthropic_adapter.py`` must recognize CommandCode as a
    Bearer-auth endpoint, or the chat_completions transport falls back to
    ``x-api-key`` and gets a 401.
    """

    def test_requires_bearer_auth_recognizes_commandcode(self):
        from agent.anthropic_endpoints import _requires_bearer_auth

        assert _requires_bearer_auth("https://api.commandcode.ai/provider/v1") is True
        assert _requires_bearer_auth("https://api.commandcode.ai/provider/v1/models") is True
        assert _requires_bearer_auth("https://api.commandcode.ai/anthropic") is True

    def test_bearer_auth_does_not_affect_unrelated(self):
        from agent.anthropic_endpoints import _requires_bearer_auth

        # Native Anthropic still uses x-api-key
        assert _requires_bearer_auth("https://api.anthropic.com") is False
        # OpenRouter still uses Bearer through its own transport path
        assert _requires_bearer_auth("https://openrouter.ai/api/v1") is False

    def test_bearer_auth_case_insensitive(self):
        from agent.anthropic_endpoints import _requires_bearer_auth

        assert _requires_bearer_auth("https://API.COMMANDCODE.AI/provider/v1") is True


# ── Registry integrity ───────────────────────────────────────────────────────

class TestCommandCodeRegistryIntegrity:
    """Both profiles are discoverable and distinct."""

    def test_both_profiles_registered(self):
        import model_tools  # noqa: F401
        import providers

        chat = providers.get_provider_profile("commandcode")
        anth = providers.get_provider_profile("commandcode-anthropic")
        assert chat is not None
        assert anth is not None
        assert chat is not anth  # distinct profile instances

    def test_alias_lookup(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("commandcode-chat") is not None
        assert providers.get_provider_profile("commandcode-claude") is not None

    def test_unknown_returns_none(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("commandcode-nonexistent") is None


# ── Model list filtering ──────────────────────────────────────────────────────

class TestCommandCodeModelFiltering:
    """``fetch_models`` filtering contracts."""

    def test_anthropic_profile_filters_to_claude(self):
        """If we mock a response with mixed models, anthropic profile
        should only return claude-* models.
        """
        from plugins.model_providers.commandcode import CommandCodeAnthropicProfile

        profile = CommandCodeAnthropicProfile(
            name="test-cc-anth",
            api_mode="anthropic_messages",
            env_vars=("COMMANDCODE_API_KEY",),
            base_url="https://api.commandcode.ai/provider/v1",
        )

        # Don't actually hit the network — just test the filter logic.
        # The class has a fetch_models override that filters.
        # We verify the filter works by inspecting the method.
        import inspect

        source = inspect.getsource(profile.fetch_models)
        assert "startswith(\"claude-\")" in source or '"claude-" in m' in source, (
            "CommandCodeAnthropicProfile.fetch_models should filter to claude-* models"
        )


# ── Picker contract ──────────────────────────────────────────────────────────

class TestCommandCodeFetchModelsPickerContract:
    """``fetch_models`` must accept the kwargs the model picker passes.

    Regression: the generic live-fetch path in ``hermes_cli/models.py``
    (``provider_model_ids``) calls ``profile.fetch_models(api_key=...,
    base_url=...)``. The original CommandCode overrides only accepted
    ``api_key``/``timeout``, so every picker open raised TypeError, which
    was swallowed, leaving the provider with zero models.
    """

    @pytest.mark.parametrize("profile_name", ["commandcode", "commandcode-anthropic"])
    def test_accepts_base_url_kwarg(self, profile_name):
        import inspect

        import model_tools  # noqa: F401 — triggers discovery
        import providers

        profile = providers.get_provider_profile(profile_name)
        assert profile is not None
        assert "base_url" in inspect.signature(profile.fetch_models).parameters

    def test_resolve_provider_full(self):
        """Both profiles must resolve through the model-switch path.

        Regression: ``resolve_provider_full`` only knew models.dev + overlay
        providers, so plugin-only providers (commandcode) failed with
        "Unknown provider" on /model switches even though the picker listed
        them.
        """
        from hermes_cli.providers import resolve_provider_full

        chat = resolve_provider_full("commandcode", {}, [])
        assert chat is not None and chat.id == "commandcode"
        assert chat.transport == "openai_chat"
        assert "COMMANDCODE_API_KEY" in chat.api_key_env_vars

        anth = resolve_provider_full("commandcode-anthropic", {}, [])
        assert anth is not None and anth.id == "commandcode-anthropic"
        assert anth.transport == "anthropic_messages"


# ── base_url endpoint override ───────────────────────────────────────────────

class TestCommandCodeBaseUrlOverride:
    """A custom base_url must redirect the catalog fetch; the default must not.

    The picker passes ``base_url`` unconditionally (profile default when the
    user configured nothing), so only a value differing from the default
    ``_COMMANDCODE_BASE`` counts as a customised endpoint.
    """

    def _serve(self, models):
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": models}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), H)
        Thread(target=server.serve_forever, daemon=True).start()
        return server, server.server_address[1]

    def test_custom_base_url_redirects_fetch(self, commandcode_profile):
        server, port = self._serve([{"id": "proxied/model-x"}])
        try:
            result = commandcode_profile.fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}"
            )
            assert result == ["proxied/model-x"]
        finally:
            server.shutdown()

    def test_custom_base_url_redirects_anthropic_fetch(
        self, commandcode_anthropic_profile
    ):
        server, port = self._serve(
            [{"id": "claude-sonnet-4-6"}, {"id": "deepseek/deepseek-v4-pro"}]
        )
        try:
            result = commandcode_anthropic_profile.fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}"
            )
            assert result == ["claude-sonnet-4-6"]  # claude-* filter still applies
        finally:
            server.shutdown()

    def test_default_base_url_hits_default_endpoint(self, commandcode_profile):
        """Echoing the profile default back must NOT count as an override."""
        import sys
        from unittest.mock import patch as mock_patch

        # The bundled plugin module is registered at discovery time under
        # ``plugins.model_providers.commandcode`` — resolve via the profile's
        # own __module__ so the test doesn't depend on discovery mechanics.
        cc_mod = sys.modules[type(commandcode_profile).__module__]

        captured = {}

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"data": [{"id": "m1"}]}'

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            return _FakeResp()

        with mock_patch.object(
            cc_mod.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            result = commandcode_profile.fetch_models(
                api_key="k", base_url=cc_mod._COMMANDCODE_BASE + "/"
            )
        assert result == ["m1"]
        assert captured["url"] == cc_mod._COMMANDCODE_MODELS_URL

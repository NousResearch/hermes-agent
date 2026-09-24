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



class TestCommandCodeReasoningWireControls:
    """DeepSeek V4+ defaults to thinking when ``thinking`` is omitted, so the profile
    must put the user's setting on the wire (#95232); other families stay a no-op."""

    def test_deepseek_disabled_reasoning_sends_thinking_disabled(self, commandcode_profile):
        extra_body, top_level = commandcode_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="deepseek/deepseek-v4-flash",
        )
        assert extra_body.get("thinking") == {"type": "disabled"}
        assert top_level == {}

    def test_deepseek_effort_matches_native_profile_and_others_noop(self, commandcode_profile):
        from plugins.model_providers.deepseek import deepseek

        rc = {"enabled": True, "effort": "low"}
        expected = deepseek.build_api_kwargs_extras(reasoning_config=rc, model="deepseek-v4.1-flash")
        assert expected[1].get("reasoning_effort") == "low"  # equality below must not be ({}, {}) == ({}, {})
        assert commandcode_profile.build_api_kwargs_extras(
            reasoning_config=rc, model="deepseek/deepseek-v4.1-flash"
        ) == expected
        assert commandcode_profile.build_api_kwargs_extras(
            reasoning_config=rc, model="Qwen/Qwen3.7-Max"
        ) == ({}, {})


class TestCommandCodeAnthropicProfileIdentity:
    """Anthropic-compatible profile metadata."""






    def test_fallback_models_are_claude_family(self, commandcode_anthropic_profile):
        for model in commandcode_anthropic_profile.fallback_models:
            assert model.startswith("claude-"), (
                f"All anthropic fallback models should be claude-*: got {model}"
            )





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



# ── Model list filtering ──────────────────────────────────────────────────────



# ── Picker contract ──────────────────────────────────────────────────────────

class TestCommandCodeFetchModelsPickerContract:
    """``fetch_models`` must accept the kwargs the model picker passes.

    Regression: the generic live-fetch path in ``hermes_cli/models.py``
    (``provider_model_ids``) calls ``profile.fetch_models(api_key=...,
    base_url=...)``. The original CommandCode overrides only accepted
    ``api_key``/``timeout``, so every picker open raised TypeError, which
    was swallowed, leaving the provider with zero models.
    """


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

        with mock_patch.object(cc_mod, "open_credentialed_url", side_effect=fake_urlopen):
            result = commandcode_profile.fetch_models(
                api_key="k", base_url=cc_mod._COMMANDCODE_BASE + "/"
            )
        assert result == ["m1"]
        assert captured["url"] == cc_mod._COMMANDCODE_MODELS_URL

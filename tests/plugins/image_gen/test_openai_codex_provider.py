"""Tests for the bundled ``openai-codex`` image_gen plugin.

Mirrors ``test_openai_provider.py`` but targets the standalone
Codex/ChatGPT-OAuth-backed provider that uses the Responses
``image_generation`` tool path instead of the ``images.generate`` REST
endpoint.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory.
codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    codex_plugin._cached_command_token_source.cache_clear()
    yield tmp_path
    codex_plugin._cached_command_token_source.cache_clear()


@pytest.fixture
def provider(monkeypatch):
    # Codex plugin is API-key-independent; clear it to make the test honest.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return codex_plugin.OpenAICodexImageGenProvider()


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai-codex"

    def test_display_name(self, provider):
        assert provider.display_name == "OpenAI (Codex auth)"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_setup_schema_has_no_required_env_vars(self, provider):
        schema = provider.get_setup_schema()
        assert schema["env_vars"] == []
        assert schema["badge"] == "free"


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_unavailable_without_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False

    def test_available_with_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True

    def test_openai_api_key_alone_is_not_enough(self, monkeypatch):
        # Codex plugin is intentionally orthogonal to the API-key plugin —
        # the API key alone must NOT make it appear available.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False


# ── Generate ────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_auth_error_without_codex_token(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"


    def test_generate_uses_codex_stream_path(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: {"b64": _b64_png(), "source": "final"})

        result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["provider"] == "openai-codex"
        assert result["quality"] == "medium"
        assert result.get("image_source") == "final"
        assert result.get("pixel_size") == "1x1"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        # Filename prefix differs from the API-key plugin so cache audits can
        # tell the two backends apart.
        assert saved.name.startswith("openai_codex_")

    def test_codex_stream_request_shape(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        captured = {}

        def _collect(
            token,
            *,
            prompt,
            size,
            quality,
            input_images=None,
            base_url=None,
            extra_headers=None,
        ):
            captured.update(codex_plugin._build_responses_payload(
                prompt=prompt,
                size=size,
                quality=quality,
                input_images=input_images,
            ))
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat", aspect_ratio="portrait")
        assert result["success"] is True

        assert captured["model"] == "gpt-5.5"
        assert captured["store"] is False
        assert captured["input"][0]["type"] == "message"
        assert captured["input"][0]["role"] == "user"
        assert captured["input"][0]["content"][0]["type"] == "input_text"
        # Regression for #19505: the Codex backend 400s on every tool_choice
        # shape we have for the hosted ``image_generation`` tool, so the
        # provider must omit tool_choice entirely and rely on instructions.
        assert "tool_choice" not in captured

        tool = captured["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "gpt-image-2"
        assert tool["quality"] == "medium"
        assert tool["size"] == "1024x1536"
        assert tool["output_format"] == "png"
        assert tool["background"] == "opaque"
        # Progressive previews disabled: partial frames were being saved as
        # finals and presented as smeared/unfinished images.
        assert tool["partial_images"] == 0

    def test_generate_routes_through_named_auth_provider(
        self, provider, monkeypatch
    ):
        monkeypatch.setattr(
            codex_plugin,
            "_load_image_gen_config",
            lambda: {"openai-codex": {"auth_provider": "codex-passive"}},
        )
        monkeypatch.setattr(
            codex_plugin,
            "_resolve_named_provider_auth",
            lambda name: codex_plugin._CodexImageAuth(
                "passive-token",
                "https://chatgpt.example/backend-api/codex",
                {"ChatGPT-Account-ID": "acct-test"},
            ),
        )
        captured = {}

        def _collect(token, **kwargs):
            captured["token"] = token
            captured.update(kwargs)
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat")

        assert result["success"] is True
        assert captured["token"] == "passive-token"
        assert captured["base_url"] == "https://chatgpt.example/backend-api/codex"
        assert captured["extra_headers"] == {"ChatGPT-Account-ID": "acct-test"}

    def test_capabilities_advertise_image_inputs(self, provider):
        caps = provider.capabilities()
        assert caps["modalities"] == ["text", "image"]
        assert caps["max_reference_images"] == 16


    def test_rejects_non_image_local_source(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        text_path = tmp_path / "not-image.txt"
        text_path.write_text("hello")

        result = provider.generate("edit this", image_url=str(text_path))

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_input"
        assert "not a supported image" in result["error"]


    def test_partial_image_event_used_when_done_missing(self):
        """Extractor may surface partial b64 when no final exists (fallback only)."""
        payload = {
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": _b64_png(),
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()
        result, partial = codex_plugin._extract_image_candidates(payload)
        assert result is None
        assert partial == _b64_png()

    def test_final_result_wins_over_coexisting_partial_in_same_payload(self):
        """Blind spot that shipped the smear bug: both fields in one payload.

        partial_image_b64 must never overwrite image_generation_call.result
        when they coexist in the same event tree.
        """
        final = _b64_png()
        # Distinct non-empty stand-in so equality proves which field won.
        partial = "cGFydGlhbC1vbmx5LW5vdC1hLXJlYWwtZmluYWw="
        payload = {
            "type": "response.output_item.done",
            "item": {
                "type": "image_generation_call",
                "status": "completed",
                "result": final,
                "partial_image_b64": partial,
            },
        }
        assert codex_plugin._extract_image_b64(payload) == final
        result, got_partial = codex_plugin._extract_image_candidates(payload)
        assert result == final
        assert got_partial == partial

    def test_nested_final_wins_over_sibling_partial(self):
        payload = {
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": _b64_png(),
                }],
            },
            "partial_image_b64": "cGFydGlhbC1zaWJsaW5n",
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()

    def test_sse_parser_handles_event_and_data_lines(self):
        class _Response:
            def iter_lines(self):
                return iter([
                    "event: response.output_item.done",
                    'data: {"item": {"type": "image_generation_call", "result": "abc"}}',
                    "",
                ])

        events = list(codex_plugin._iter_sse_json(_Response()))
        assert events == [{
            "type": "response.output_item.done",
            "item": {"type": "image_generation_call", "result": "abc"},
        }]

    def test_final_response_sweep_recovers_image(self):
        """Completed response output is found by recursive payload scanning."""
        payload = {
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig_final",
                    "result": _b64_png(),
                }],
            },
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()

    def test_partial_only_stream_fails_closed_after_retry(self, provider, monkeypatch):
        """Partial-only streams must not return success:true with a smear frame."""
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _partial_only(*args, **kwargs):
            calls["n"] += 1
            return {"b64": _b64_png(), "source": "partial"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _partial_only)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "incomplete_image"
        assert "partial" in result["error"].lower()
        # One initial attempt + one content-agnostic retry.
        assert calls["n"] == codex_plugin._NONFINAL_RETRIES + 1

    def test_empty_stream_retries_then_fails(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _empty(*args, **kwargs):
            calls["n"] += 1
            return None

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _empty)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"
        assert calls["n"] == codex_plugin._NONFINAL_RETRIES + 1

    def test_partial_then_final_on_retry_succeeds(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _then_final(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"b64": _b64_png(), "source": "partial"}
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _then_final)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result.get("image_source") == "final"
        assert calls["n"] == 2

    def test_empty_then_final_on_retry_succeeds(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _then_final(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _then_final)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result.get("image_source") == "final"
        assert calls["n"] == 2

    def test_empty_response_returns_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_NONFINAL_RETRIES", 0)
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: None)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_stream_exception_returns_api_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        def _boom(*args, **kwargs):
            raise RuntimeError("cloudflare 403")

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _boom)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "cloudflare 403" in result["error"]

    def test_tool_choice_400_surfaces_verbatim_not_as_capability_error(
        self, provider, monkeypatch
    ):
        """The tool_choice 400 must NOT be reported as an account limitation.

        Regression for #19505 / #49008 / #31335: a previous version classified
        this exact request-shape rejection as "Image generation is not enabled
        for the current Codex account", telling every affected user to abandon
        Codex over a bug in our own payload. The wire error must reach the user
        unedited so it stays diagnosable.

        Drives the REAL httpx boundary (not a mocked ``_collect_image_b64``) so
        the classification path is actually exercised — mocking the collector
        would skip the code under test entirely.
        """
        import httpx

        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        body = json.dumps({
            "error": {
                "message": "Tool choice 'image_generation' not found in 'tools' parameter.",
                "type": "invalid_request_error",
                "param": "tool_choice",
            }
        })

        def _handler(request):
            return httpx.Response(400, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "HTTP 400" in result["error"]
        assert "tools' parameter" in result["error"]
        # The account-entitlement misdiagnosis must not come back.
        assert "not enabled for the current Codex account" not in result["error"]
        assert result["error_type"] != "capability_unsupported"


class TestRequestShape:
    def test_named_provider_key_cmd_source_is_reused(self, monkeypatch):
        entry = {
            "name": "codex-passive",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "key_cmd": "token-helper print",
            "extra_headers": {"ChatGPT-Account-ID": "acct-test"},
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda requested: entry,
        )
        token_source = Mock(return_value="passive-token")
        builder = Mock(return_value=token_source)
        monkeypatch.setattr(
            "agent.command_token_source.build_command_token_provider", builder
        )

        first = codex_plugin._resolve_named_provider_auth("codex-passive")
        second = codex_plugin._resolve_named_provider_auth("codex-passive")

        assert first == second == codex_plugin._CodexImageAuth(
            "passive-token",
            "https://chatgpt.com/backend-api/codex",
            {"ChatGPT-Account-ID": "acct-test"},
        )
        builder.assert_called_once_with("token-helper print", "codex-passive")
        assert token_source.call_count == 2

    def test_is_available_does_not_execute_key_cmd(self, monkeypatch):
        """Availability is a hot-path probe; it must not run the auth helper."""
        monkeypatch.setattr(
            codex_plugin,
            "_load_image_gen_config",
            lambda: {"openai-codex": {"auth_provider": "codex-passive"}},
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda requested: {
                "name": "codex-passive",
                "base_url": "https://gw.example/codex",
                "key_cmd": "token-helper print",
            },
        )
        builder = Mock(side_effect=AssertionError("key_cmd ran during is_available()"))
        monkeypatch.setattr(
            "agent.command_token_source.build_command_token_provider", builder
        )

        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True
        builder.assert_not_called()

    def test_missing_named_provider_never_falls_back_to_codex_oauth(
        self, provider, monkeypatch
    ):
        """A typo'd/disabled auth_provider fails closed, it does not downgrade."""
        monkeypatch.setattr(
            codex_plugin,
            "_load_image_gen_config",
            lambda: {"openai-codex": {"auth_provider": "typo"}},
        )
        # _get_named_custom_provider returns None for absent AND disabled entries.
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda requested: None,
        )
        # Legacy OAuth IS present — so a pass here would prove silent fallback.
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )

        assert provider.is_available() is False
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_named_provider_refuses_builtin_provider_fallback(self, monkeypatch):
        """resolve_runtime_provider() must not reroute us to a built-in."""
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda requested: {
                "name": "codex-passive",
                "base_url": "https://gw.example/codex",
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "unrelated-builtin-key",
            },
        )

        with pytest.raises(RuntimeError, match="refusing to route"):
            codex_plugin._resolve_named_provider_auth("codex-passive")

    def test_static_named_provider_uses_runtime_credential(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda requested: {
                "name": "codex-passive",
                "base_url": "https://gw.example/codex",
                "extra_headers": {
                    "ChatGPT-Account-ID": "from-entry",
                    "X-Entry-Route": "preserved",
                },
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "custom",
                "base_url": "https://gw.example/codex/",
                "api_key": "static-key",
                "extra_headers": {"ChatGPT-Account-ID": "from-runtime"},
            },
        )

        assert codex_plugin._resolve_named_provider_auth(
            "codex-passive"
        ) == codex_plugin._CodexImageAuth(
            "static-key",
            "https://gw.example/codex",
            {
                "X-Entry-Route": "preserved",
                "ChatGPT-Account-ID": "from-runtime",
            },
        )

    def test_open_endpoint_sentinel_is_not_treated_as_a_credential(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda requested: {
                "name": "codex-passive",
                "base_url": "https://gw.example/codex",
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kwargs: {
                "provider": "custom",
                "base_url": "https://gw.example/codex",
                "api_key": "no-key-required",
            },
        )

        with pytest.raises(RuntimeError, match="produced no credential"):
            codex_plugin._resolve_named_provider_auth("codex-passive")

    def test_omitted_auth_provider_leaves_codex_oauth_path_untouched(
        self, monkeypatch
    ):
        monkeypatch.setattr(codex_plugin, "_load_image_gen_config", lambda: {})
        lookup = Mock(side_effect=AssertionError("named-provider lookup ran"))
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider", lookup
        )
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )

        assert codex_plugin._resolve_codex_image_auth() == codex_plugin._CodexImageAuth(
            "codex-token", codex_plugin._CODEX_BASE_URL, {}
        )
        lookup.assert_not_called()

    def test_named_provider_headers_and_route_reach_http_boundary(self, monkeypatch):
        import httpx

        seen = {}

        def _handler(request):
            seen["request"] = request
            body = (
                'event: response.output_item.done\n'
                "data: "
                + json.dumps({
                    "item": {
                        "type": "image_generation_call",
                        "result": _b64_png(),
                    }
                })
                + "\n\n"
            )
            return httpx.Response(200, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        result = codex_plugin._collect_image_b64(
            "passive-token",
            prompt="a cat",
            size="1024x1024",
            quality="low",
            base_url="https://chatgpt.example/backend-api/codex",
            extra_headers={
                "ChatGPT-Account-ID": "acct-test",
                "Authorization": "Bearer stale-token",
                # Header names are case-insensitive on the wire but dict keys
                # are not, and httpx does not de-duplicate: an unfiltered
                # lowercase spelling would be sent ALONGSIDE the real bearer
                # and win at upstreams that honour the first Authorization.
                "authorization": "Bearer stale-token-lowercase",
                "ACCEPT": "application/json",
                "content-type": "text/plain",
            },
        )

        assert result == {"b64": _b64_png(), "source": "final"}
        request = seen["request"]
        assert str(request.url) == "https://chatgpt.example/backend-api/codex/responses"
        assert request.headers["ChatGPT-Account-ID"] == "acct-test"
        # get_list() exposes every value sent under the name, so a smuggled
        # duplicate cannot hide behind a single-value lookup.
        assert request.headers.get_list("authorization") == ["Bearer passive-token"]
        assert request.headers.get_list("accept") == ["text/event-stream"]
        assert request.headers.get_list("content-type") == ["application/json"]

    def test_config_headers_cannot_smuggle_a_second_authorization(self):
        """Reserved headers are dropped regardless of the casing config uses."""
        merged = codex_plugin._merge_request_headers(
            {"User-Agent": "codex_cli_rs/0.0.0", "ChatGPT-Account-ID": "from-jwt"},
            {
                "Authorization": "Bearer a",
                "authorization": "Bearer b",
                "AUTHORIZATION": "Bearer c",
                "Accept": "application/json",
                "Content-Type": "text/plain",
            },
        )
        assert not [k for k in merged if k.lower() in {"authorization", "accept", "content-type"}]
        assert merged["User-Agent"] == "codex_cli_rs/0.0.0"

    def test_config_headers_override_route_metadata_in_place(self):
        """A case-variant override replaces the key instead of duplicating it."""
        merged = codex_plugin._merge_request_headers(
            {"User-Agent": "codex_cli_rs/0.0.0", "ChatGPT-Account-ID": "from-jwt"},
            {"chatgpt-account-id": "from-config", "  ": "dropped"},
        )
        account_keys = [k for k in merged if k.lower() == "chatgpt-account-id"]
        assert account_keys == ["ChatGPT-Account-ID"]
        assert merged["ChatGPT-Account-ID"] == "from-config"
        assert "  " not in merged

    def test_payload_omits_tool_choice(self):
        """Codex rejects every tool_choice shape for hosted image_generation."""
        payload = codex_plugin._build_responses_payload(
            prompt="a red circle",
            size="1024x1024",
            quality="low",
        )
        assert "tool_choice" not in payload
        # The hosted tool itself is still requested, and instructions do the steering.
        assert payload["tools"][0]["type"] == "image_generation"
        assert payload["instructions"]

    def test_http_error_body_is_truncated_but_preserved(self, monkeypatch):
        """A large error body is capped at 500 chars and still surfaced."""
        import httpx

        body = json.dumps({
            "metadata": "x" * 600,
            "error": {
                "message": "Tool choice 'image_generation' not found in 'tools' parameter."
            },
        })

        def _handler(request):
            return httpx.Response(400, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        with pytest.raises(RuntimeError, match="HTTP 400") as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token",
                prompt="a cat",
                size="1024x1024",
                quality="low",
            )

        message = str(excinfo.value)
        # Body is capped, but the actionable wire message still reaches the user.
        assert "tools' parameter" in message
        assert len(message) < len(body)


# ── Plugin entry point ──────────────────────────────────────────────────────


class TestRegistration:
    def test_register_calls_register_image_gen_provider(self):
        registered = []

        class _Ctx:
            def register_image_gen_provider(self, prov):
                registered.append(prov)

        codex_plugin.register(_Ctx())
        assert len(registered) == 1
        assert registered[0].name == "openai-codex"

"""Tests for the google-gemini-cli OAuth + Code Assist inference provider.

Covers:
- agent/google_oauth.py — PKCE, credential I/O with packed refresh format,
  token refresh dedup, invalid_grant handling, headless paste fallback
- agent/google_code_assist.py — project discovery, VPC-SC fallback, onboarding
  with LRO polling, quota retrieval
- agent/gemini_cloudcode_adapter.py — OpenAI↔Gemini translation, request
  envelope wrapping, response unwrapping, tool calls bidirectional, streaming
- Provider registration — registry entry, aliases, runtime dispatch, auth
  status, _OAUTH_CAPABLE_PROVIDERS regression guard
"""
from __future__ import annotations

import base64
import hashlib
import json
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (
        "HERMES_GEMINI_CLIENT_ID",
        "HERMES_GEMINI_CLIENT_SECRET",
        "HERMES_GEMINI_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_PROJECT_ID",
        "SSH_CONNECTION",
        "SSH_CLIENT",
        "SSH_TTY",
        "HERMES_HEADLESS",
    ):
        monkeypatch.delenv(key, raising=False)
    return home


# =============================================================================
# google_oauth.py — PKCE + packed refresh format
# =============================================================================

class TestPkce:
    def test_verifier_and_challenge_s256_roundtrip(self):
        from agent.google_oauth import _generate_pkce_pair

        verifier, challenge = _generate_pkce_pair()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        assert challenge == expected
        assert 43 <= len(verifier) <= 128


class TestRefreshParts:
    def test_parse_bare_token(self):
        from agent.google_oauth import RefreshParts

        p = RefreshParts.parse("abc-token")
        assert p.refresh_token == "abc-token"
        assert p.project_id == ""
        assert p.managed_project_id == ""

    def test_parse_packed(self):
        from agent.google_oauth import RefreshParts

        p = RefreshParts.parse("rt|proj-123|mgr-456")
        assert p.refresh_token == "rt"
        assert p.project_id == "proj-123"
        assert p.managed_project_id == "mgr-456"

    def test_format_bare_token(self):
        from agent.google_oauth import RefreshParts

        assert RefreshParts(refresh_token="rt").format() == "rt"

    def test_format_with_project(self):
        from agent.google_oauth import RefreshParts

        packed = RefreshParts(
            refresh_token="rt", project_id="p1", managed_project_id="m1",
        ).format()
        assert packed == "rt|p1|m1"
        # Roundtrip
        parsed = RefreshParts.parse(packed)
        assert parsed.refresh_token == "rt"
        assert parsed.project_id == "p1"
        assert parsed.managed_project_id == "m1"

    def test_format_empty_refresh_token_returns_empty(self):
        from agent.google_oauth import RefreshParts

        assert RefreshParts(refresh_token="").format() == ""


class TestClientCredResolution:
    def test_env_override(self, monkeypatch):
        from agent.google_oauth import _get_client_id

        monkeypatch.setenv("HERMES_GEMINI_CLIENT_ID", "custom-id.apps.googleusercontent.com")
        assert _get_client_id() == "custom-id.apps.googleusercontent.com"

    def test_shipped_default_used_when_no_env(self):
        """Out of the box, the public gemini-cli desktop client is used."""
        from agent.google_oauth import _get_client_id, _DEFAULT_CLIENT_ID

        # Confirmed PUBLIC: baked into Google's open-source gemini-cli
        assert _DEFAULT_CLIENT_ID.endswith(".apps.googleusercontent.com")
        assert _DEFAULT_CLIENT_ID.startswith("681255809395-")
        assert _get_client_id() == _DEFAULT_CLIENT_ID

    def test_shipped_default_secret_present(self):
        from agent.google_oauth import _DEFAULT_CLIENT_SECRET, _get_client_secret

        assert _DEFAULT_CLIENT_SECRET.startswith("GOCSPX-")
        assert len(_DEFAULT_CLIENT_SECRET) >= 20
        assert _get_client_secret() == _DEFAULT_CLIENT_SECRET

    def test_falls_back_to_scrape_when_defaults_wiped(self, tmp_path, monkeypatch):
        """Forks that wipe the shipped defaults should still work with gemini-cli."""
        from agent import google_oauth

        monkeypatch.setattr(google_oauth, "_DEFAULT_CLIENT_ID", "")
        monkeypatch.setattr(google_oauth, "_DEFAULT_CLIENT_SECRET", "")

        fake_bin = tmp_path / "bin" / "gemini"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.write_text("#!/bin/sh\n")
        oauth_dir = tmp_path / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "src" / "code_assist"
        oauth_dir.mkdir(parents=True)
        (oauth_dir / "oauth2.js").write_text(
            'const OAUTH_CLIENT_ID = "99999-fakescrapedxyz.apps.googleusercontent.com";\n'
            'const OAUTH_CLIENT_SECRET = "GOCSPX-scraped-test-value-placeholder";\n'
        )

        monkeypatch.setattr("shutil.which", lambda _: str(fake_bin))
        google_oauth._scraped_creds_cache.clear()

        assert google_oauth._get_client_id().startswith("99999-")

    def test_missing_everything_raises_with_install_hint(self, monkeypatch):
        """When env + defaults + scrape all fail, raise with install instructions."""
        from agent import google_oauth

        monkeypatch.setattr(google_oauth, "_DEFAULT_CLIENT_ID", "")
        monkeypatch.setattr(google_oauth, "_DEFAULT_CLIENT_SECRET", "")
        google_oauth._scraped_creds_cache.clear()
        monkeypatch.setattr("shutil.which", lambda _: None)

        with pytest.raises(google_oauth.GoogleOAuthError) as exc_info:
            google_oauth._require_client_id()
        assert exc_info.value.code == "google_oauth_client_id_missing"

    def test_locate_gemini_cli_oauth_js_when_absent(self, monkeypatch):
        from agent import google_oauth

        monkeypatch.setattr("shutil.which", lambda _: None)
        assert google_oauth._locate_gemini_cli_oauth_js() is None

    def test_scrape_client_credentials_parses_id_and_secret(self, tmp_path, monkeypatch):
        from agent import google_oauth

        # Create a fake gemini binary and oauth2.js
        fake_gemini_bin = tmp_path / "bin" / "gemini"
        fake_gemini_bin.parent.mkdir(parents=True)
        fake_gemini_bin.write_text("#!/bin/sh\necho gemini\n")

        oauth_js_dir = tmp_path / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "src" / "code_assist"
        oauth_js_dir.mkdir(parents=True)
        oauth_js = oauth_js_dir / "oauth2.js"
        # Synthesize a harmless test fingerprint (valid shape, obvious test values)
        oauth_js.write_text(
            'const OAUTH_CLIENT_ID = "12345678-testfakenotrealxyz.apps.googleusercontent.com";\n'
            'const OAUTH_CLIENT_SECRET = "GOCSPX-aaaaaaaaaaaaaaaaaaaaaaaa";\n'
        )

        monkeypatch.setattr("shutil.which", lambda _: str(fake_gemini_bin))
        google_oauth._scraped_creds_cache.clear()

        cid, cs = google_oauth._scrape_client_credentials()
        assert cid == "12345678-testfakenotrealxyz.apps.googleusercontent.com"
        assert cs.startswith("GOCSPX-")


class TestCredentialIo:
    def _make(self):
        from agent.google_oauth import GoogleCredentials

        return GoogleCredentials(
            access_token="at-1",
            refresh_token="rt-1",
            expires_ms=int((time.time() + 3600) * 1000),
            email="user@example.com",
            project_id="proj-abc",
        )

    def test_save_and_load_packed_refresh(self):
        from agent.google_oauth import load_credentials, save_credentials

        creds = self._make()
        save_credentials(creds)
        loaded = load_credentials()
        assert loaded is not None
        assert loaded.refresh_token == "rt-1"
        assert loaded.project_id == "proj-abc"

    def test_save_uses_0600_permissions(self):
        from agent.google_oauth import _credentials_path, save_credentials

        save_credentials(self._make())
        mode = stat.S_IMODE(_credentials_path().stat().st_mode)
        assert mode == 0o600

    def test_disk_format_is_packed(self):
        from agent.google_oauth import _credentials_path, save_credentials

        save_credentials(self._make())
        data = json.loads(_credentials_path().read_text())
        # The refresh field on disk is the packed string, not a dict
        assert data["refresh"] == "rt-1|proj-abc|"

    def test_update_project_ids(self):
        from agent.google_oauth import (
            load_credentials, save_credentials, update_project_ids,
        )
        from agent.google_oauth import GoogleCredentials

        save_credentials(GoogleCredentials(
            access_token="at", refresh_token="rt",
            expires_ms=int((time.time() + 3600) * 1000),
        ))
        update_project_ids(project_id="new-proj", managed_project_id="mgr-xyz")

        loaded = load_credentials()
        assert loaded.project_id == "new-proj"
        assert loaded.managed_project_id == "mgr-xyz"


class TestAccessTokenExpired:
    def test_fresh_token_not_expired(self):
        from agent.google_oauth import GoogleCredentials

        creds = GoogleCredentials(
            access_token="at", refresh_token="rt",
            expires_ms=int((time.time() + 3600) * 1000),
        )
        assert creds.access_token_expired() is False

    def test_near_expiry_considered_expired(self):
        """60s skew — a token with 30s left is considered expired."""
        from agent.google_oauth import GoogleCredentials

        creds = GoogleCredentials(
            access_token="at", refresh_token="rt",
            expires_ms=int((time.time() + 30) * 1000),
        )
        assert creds.access_token_expired() is True

    def test_no_token_is_expired(self):
        from agent.google_oauth import GoogleCredentials

        creds = GoogleCredentials(
            access_token="", refresh_token="rt", expires_ms=999999999,
        )
        assert creds.access_token_expired() is True


class TestGetValidAccessToken:
    def _save(self, **over):
        from agent.google_oauth import GoogleCredentials, save_credentials

        defaults = {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_ms": int((time.time() + 3600) * 1000),
        }
        defaults.update(over)
        save_credentials(GoogleCredentials(**defaults))

    def test_returns_cached_when_fresh(self):
        from agent.google_oauth import get_valid_access_token

        self._save(access_token="cached-token")
        assert get_valid_access_token() == "cached-token"

    def test_refreshes_when_near_expiry(self, monkeypatch):
        from agent import google_oauth

        self._save(expires_ms=int((time.time() + 30) * 1000))
        monkeypatch.setattr(
            google_oauth, "_post_form",
            lambda *a, **kw: {"access_token": "refreshed", "expires_in": 3600},
        )
        assert google_oauth.get_valid_access_token() == "refreshed"

    def test_invalid_grant_clears_credentials(self, monkeypatch):
        from agent import google_oauth

        self._save(expires_ms=int((time.time() - 10) * 1000))

        def boom(*a, **kw):
            raise google_oauth.GoogleOAuthError(
                "invalid_grant", code="google_oauth_invalid_grant",
            )

        monkeypatch.setattr(google_oauth, "_post_form", boom)

        with pytest.raises(google_oauth.GoogleOAuthError) as exc_info:
            google_oauth.get_valid_access_token()
        assert exc_info.value.code == "google_oauth_invalid_grant"
        # Credentials should be wiped
        assert google_oauth.load_credentials() is None

    def test_preserves_refresh_when_google_omits(self, monkeypatch):
        from agent import google_oauth

        self._save(expires_ms=int((time.time() + 30) * 1000), refresh_token="original-rt")
        monkeypatch.setattr(
            google_oauth, "_post_form",
            lambda *a, **kw: {"access_token": "new", "expires_in": 3600},
        )
        google_oauth.get_valid_access_token()
        assert google_oauth.load_credentials().refresh_token == "original-rt"


class TestProjectIdResolution:
    @pytest.mark.parametrize("env_var", [
        "HERMES_GEMINI_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_PROJECT_ID",
    ])
    def test_env_vars_checked(self, monkeypatch, env_var):
        from agent.google_oauth import resolve_project_id_from_env

        monkeypatch.setenv(env_var, "test-proj")
        assert resolve_project_id_from_env() == "test-proj"

    def test_priority_order(self, monkeypatch):
        from agent.google_oauth import resolve_project_id_from_env

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "lower-priority")
        monkeypatch.setenv("HERMES_GEMINI_PROJECT_ID", "higher-priority")
        assert resolve_project_id_from_env() == "higher-priority"

    def test_no_env_returns_empty(self):
        from agent.google_oauth import resolve_project_id_from_env

        assert resolve_project_id_from_env() == ""


class TestHeadlessDetection:
    def test_detects_ssh(self, monkeypatch):
        from agent.google_oauth import _is_headless

        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 9876")
        assert _is_headless() is True

    def test_detects_hermes_headless(self, monkeypatch):
        from agent.google_oauth import _is_headless

        monkeypatch.setenv("HERMES_HEADLESS", "1")
        assert _is_headless() is True

    def test_default_not_headless(self):
        from agent.google_oauth import _is_headless

        assert _is_headless() is False


# =============================================================================
# google_code_assist.py — project discovery, onboarding, quota, VPC-SC
# =============================================================================

class TestCodeAssistVpcScDetection:
    def test_detects_vpc_sc_in_json(self):
        from agent.google_code_assist import _is_vpc_sc_violation

        body = json.dumps({
            "error": {
                "details": [{"reason": "SECURITY_POLICY_VIOLATED"}],
                "message": "blocked by policy",
            }
        })
        assert _is_vpc_sc_violation(body) is True

    def test_detects_vpc_sc_in_message(self):
        from agent.google_code_assist import _is_vpc_sc_violation

        body = '{"error": {"message": "SECURITY_POLICY_VIOLATED"}}'
        assert _is_vpc_sc_violation(body) is True

    def test_non_vpc_sc_returns_false(self):
        from agent.google_code_assist import _is_vpc_sc_violation

        assert _is_vpc_sc_violation('{"error": {"message": "not found"}}') is False
        assert _is_vpc_sc_violation("") is False


class TestLoadCodeAssist:
    def test_parses_response(self, monkeypatch):
        from agent import google_code_assist

        fake = {
            "currentTier": {"id": "free-tier"},
            "cloudaicompanionProject": "proj-123",
            "allowedTiers": [{"id": "free-tier"}, {"id": "standard-tier"}],
        }
        monkeypatch.setattr(google_code_assist, "_post_json", lambda *a, **kw: fake)

        info = google_code_assist.load_code_assist("access-token")
        assert info.current_tier_id == "free-tier"
        assert info.cloudaicompanion_project == "proj-123"
        assert "free-tier" in info.allowed_tiers
        assert "standard-tier" in info.allowed_tiers

    def test_vpc_sc_forces_standard_tier(self, monkeypatch):
        from agent import google_code_assist

        def boom(*a, **kw):
            raise google_code_assist.CodeAssistError(
                "VPC-SC policy violation", code="code_assist_vpc_sc",
            )

        monkeypatch.setattr(google_code_assist, "_post_json", boom)

        info = google_code_assist.load_code_assist("access-token", project_id="corp-proj")
        assert info.current_tier_id == "standard-tier"
        assert info.cloudaicompanion_project == "corp-proj"


class TestOnboardUser:
    def test_paid_tier_requires_project_id(self):
        from agent import google_code_assist

        with pytest.raises(google_code_assist.ProjectIdRequiredError):
            google_code_assist.onboard_user(
                "at", tier_id="standard-tier", project_id="",
            )

    def test_free_tier_no_project_required(self, monkeypatch):
        from agent import google_code_assist

        monkeypatch.setattr(
            google_code_assist, "_post_json",
            lambda *a, **kw: {"done": True, "response": {"cloudaicompanionProject": "gen-123"}},
        )
        resp = google_code_assist.onboard_user("at", tier_id="free-tier")
        assert resp["done"] is True

    def test_lro_polling(self, monkeypatch):
        """Simulate a long-running operation that completes on the second poll."""
        from agent import google_code_assist

        call_count = {"n": 0}

        def fake_post(url, body, token, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"name": "operations/op-abc", "done": False}
            return {"name": "operations/op-abc", "done": True, "response": {}}

        monkeypatch.setattr(google_code_assist, "_post_json", fake_post)
        monkeypatch.setattr(google_code_assist.time, "sleep", lambda *_: None)

        resp = google_code_assist.onboard_user(
            "at", tier_id="free-tier",
        )
        assert resp["done"] is True
        assert call_count["n"] >= 2


class TestRetrieveUserQuota:
    def test_parses_buckets(self, monkeypatch):
        from agent import google_code_assist

        fake = {
            "buckets": [
                {
                    "modelId": "gemini-2.5-pro",
                    "tokenType": "input",
                    "remainingFraction": 0.75,
                    "resetTime": "2026-04-17T00:00:00Z",
                },
                {
                    "modelId": "gemini-2.5-flash",
                    "remainingFraction": 0.9,
                },
            ]
        }
        monkeypatch.setattr(google_code_assist, "_post_json", lambda *a, **kw: fake)

        buckets = google_code_assist.retrieve_user_quota("at", project_id="p1")
        assert len(buckets) == 2
        assert buckets[0].model_id == "gemini-2.5-pro"
        assert buckets[0].remaining_fraction == 0.75
        assert buckets[1].remaining_fraction == 0.9


class TestResolveProjectContext:
    def test_configured_shortcircuits(self, monkeypatch):
        from agent.google_code_assist import resolve_project_context

        # Should NOT call loadCodeAssist when configured_project_id is set
        def should_not_be_called(*a, **kw):
            raise AssertionError("should short-circuit")

        monkeypatch.setattr(
            "agent.google_code_assist._post_json", should_not_be_called,
        )
        ctx = resolve_project_context("at", configured_project_id="proj-abc")
        assert ctx.project_id == "proj-abc"
        assert ctx.source == "config"

    def test_env_shortcircuits(self, monkeypatch):
        from agent.google_code_assist import resolve_project_context

        monkeypatch.setattr(
            "agent.google_code_assist._post_json",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("nope")),
        )
        ctx = resolve_project_context("at", env_project_id="env-proj")
        assert ctx.project_id == "env-proj"
        assert ctx.source == "env"

    def test_discovers_via_load_code_assist(self, monkeypatch):
        from agent import google_code_assist

        monkeypatch.setattr(
            google_code_assist, "_post_json",
            lambda *a, **kw: {
                "currentTier": {"id": "free-tier"},
                "cloudaicompanionProject": "discovered-proj",
            },
        )
        ctx = google_code_assist.resolve_project_context("at")
        assert ctx.project_id == "discovered-proj"
        assert ctx.tier_id == "free-tier"
        assert ctx.source == "discovered"


# =============================================================================
# gemini_cloudcode_adapter.py — request/response translation
# =============================================================================

class TestBuildGeminiRequest:
    def test_user_assistant_messages(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert req["contents"][0] == {
            "role": "user", "parts": [{"text": "hi"}],
        }
        assert req["contents"][1] == {
            "role": "model", "parts": [{"text": "hello"}],
        }

    def test_system_instruction_separated(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ])
        assert req["systemInstruction"]["parts"][0]["text"] == "You are helpful"
        # System should NOT appear in contents
        assert all(c["role"] != "system" for c in req["contents"])

    def test_multiple_system_messages_joined(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(messages=[
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "hi"},
        ])
        assert "A\nB" in req["systemInstruction"]["parts"][0]["text"]

    def test_tool_call_translation(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(messages=[
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }],
            },
        ])
        # Assistant turn should have a functionCall part
        model_turn = req["contents"][1]
        assert model_turn["role"] == "model"
        fc_part = next(p for p in model_turn["parts"] if "functionCall" in p)
        assert fc_part["functionCall"]["name"] == "get_weather"
        assert fc_part["functionCall"]["args"] == {"city": "SF"}

    def test_tool_result_translation(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }]},
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "c1",
                "content": '{"temp": 72}',
            },
        ])
        # Last content turn should carry functionResponse
        last = req["contents"][-1]
        fr_part = next(p for p in last["parts"] if "functionResponse" in p)
        assert fr_part["functionResponse"]["name"] == "get_weather"
        assert fr_part["functionResponse"]["response"] == {"temp": 72}

    def test_tools_translated_to_function_declarations(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"type": "function", "function": {
                    "name": "fn1", "description": "foo",
                    "parameters": {"type": "object"},
                }},
            ],
        )
        decls = req["tools"][0]["functionDeclarations"]
        assert decls[0]["name"] == "fn1"
        assert decls[0]["description"] == "foo"
        assert decls[0]["parameters"] == {"type": "object"}

    def test_tools_strip_json_schema_only_fields_from_parameters(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"type": "function", "function": {
                    "name": "fn1",
                    "description": "foo",
                    "parameters": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "city": {
                                "type": "string",
                                "$schema": "ignored",
                                "description": "City name",
                                "additionalProperties": False,
                            }
                        },
                        "required": ["city"],
                    },
                }},
            ],
        )
        params = req["tools"][0]["functionDeclarations"][0]["parameters"]
        assert "$schema" not in params
        assert "additionalProperties" not in params
        assert params["type"] == "object"
        assert params["required"] == ["city"]
        assert params["properties"]["city"] == {
            "type": "string",
            "description": "City name",
        }

    def test_tool_choice_auto(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="auto",
        )
        assert req["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"

    def test_tool_choice_required(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="required",
        )
        assert req["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"

    def test_tool_choice_specific_function(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "function", "function": {"name": "my_fn"}},
        )
        cfg = req["toolConfig"]["functionCallingConfig"]
        assert cfg["mode"] == "ANY"
        assert cfg["allowedFunctionNames"] == ["my_fn"]

    def test_generation_config_params(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            stop=["###", "END"],
        )
        gc = req["generationConfig"]
        assert gc["temperature"] == 0.7
        assert gc["maxOutputTokens"] == 512
        assert gc["topP"] == 0.9
        assert gc["stopSequences"] == ["###", "END"]

    def test_thinking_config_normalization(self):
        from agent.gemini_cloudcode_adapter import build_gemini_request

        req = build_gemini_request(
            messages=[{"role": "user", "content": "hi"}],
            thinking_config={"thinking_budget": 1024, "include_thoughts": True},
        )
        tc = req["generationConfig"]["thinkingConfig"]
        assert tc["thinkingBudget"] == 1024
        assert tc["includeThoughts"] is True


class TestWrapCodeAssistRequest:
    def test_envelope_shape(self):
        from agent.gemini_cloudcode_adapter import wrap_code_assist_request

        inner = {"contents": [], "generationConfig": {}}
        wrapped = wrap_code_assist_request(
            project_id="p1", model="gemini-2.5-pro", inner_request=inner,
        )
        assert wrapped["project"] == "p1"
        assert wrapped["model"] == "gemini-2.5-pro"
        assert wrapped["request"] is inner
        assert "user_prompt_id" in wrapped
        assert len(wrapped["user_prompt_id"]) > 10


class TestTranslateGeminiResponse:
    def test_text_response(self):
        from agent.gemini_cloudcode_adapter import _translate_gemini_response

        resp = {
            "response": {
                "candidates": [{
                    "content": {"parts": [{"text": "hello world"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        }
        result = _translate_gemini_response(resp, model="gemini-2.5-flash")
        assert result.choices[0].message.content == "hello world"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_function_call_response(self):
        from agent.gemini_cloudcode_adapter import _translate_gemini_response

        resp = {
            "response": {
                "candidates": [{
                    "content": {"parts": [{
                        "functionCall": {"name": "lookup", "args": {"q": "weather"}},
                    }]},
                    "finishReason": "STOP",
                }],
            }
        }
        result = _translate_gemini_response(resp, model="gemini-2.5-flash")
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "lookup"
        assert json.loads(tc.function.arguments) == {"q": "weather"}
        assert result.choices[0].finish_reason == "tool_calls"

    def test_thought_parts_go_to_reasoning(self):
        from agent.gemini_cloudcode_adapter import _translate_gemini_response

        resp = {
            "response": {
                "candidates": [{
                    "content": {"parts": [
                        {"thought": True, "text": "let me think"},
                        {"text": "final answer"},
                    ]},
                }],
            }
        }
        result = _translate_gemini_response(resp, model="gemini-2.5-flash")
        assert result.choices[0].message.content == "final answer"
        assert result.choices[0].message.reasoning == "let me think"

    def test_unwraps_direct_format(self):
        """If response is already at top level (no 'response' wrapper), still parse."""
        from agent.gemini_cloudcode_adapter import _translate_gemini_response

        resp = {
            "candidates": [{
                "content": {"parts": [{"text": "hi"}]},
                "finishReason": "STOP",
            }],
        }
        result = _translate_gemini_response(resp, model="gemini-2.5-flash")
        assert result.choices[0].message.content == "hi"

    def test_empty_candidates(self):
        from agent.gemini_cloudcode_adapter import _translate_gemini_response

        result = _translate_gemini_response({"response": {"candidates": []}}, model="gemini-2.5-flash")
        assert result.choices[0].message.content == ""
        assert result.choices[0].finish_reason == "stop"

    def test_finish_reason_mapping(self):
        from agent.gemini_cloudcode_adapter import _map_gemini_finish_reason

        assert _map_gemini_finish_reason("STOP") == "stop"
        assert _map_gemini_finish_reason("MAX_TOKENS") == "length"
        assert _map_gemini_finish_reason("SAFETY") == "content_filter"
        assert _map_gemini_finish_reason("RECITATION") == "content_filter"


class TestTranslateStreamEvent:
    def test_parallel_calls_to_same_tool_get_unique_indices(self):
        """Gemini may emit several functionCall parts with the same name in a
        single turn (e.g. parallel file reads). Each must get its own OpenAI
        ``index`` — otherwise downstream aggregators collapse them into one.
        """
        from agent.gemini_cloudcode_adapter import _translate_stream_event

        event = {
            "response": {
                "candidates": [{
                    "content": {"parts": [
                        {"functionCall": {"name": "read_file", "args": {"path": "a"}}},
                        {"functionCall": {"name": "read_file", "args": {"path": "b"}}},
                        {"functionCall": {"name": "read_file", "args": {"path": "c"}}},
                    ]},
                }],
            }
        }
        counter = [0]
        chunks = _translate_stream_event(event, model="gemini-2.5-flash",
                                         tool_call_counter=counter)
        indices = [c.choices[0].delta.tool_calls[0].index for c in chunks]
        assert indices == [0, 1, 2]
        assert counter[0] == 3

    def test_counter_persists_across_events(self):
        """Index assignment must continue across SSE events in the same stream."""
        from agent.gemini_cloudcode_adapter import _translate_stream_event

        def _event(name):
            return {"response": {"candidates": [{
                "content": {"parts": [{"functionCall": {"name": name, "args": {}}}]},
            }]}}

        counter = [0]
        chunks_a = _translate_stream_event(_event("foo"), model="m", tool_call_counter=counter)
        chunks_b = _translate_stream_event(_event("bar"), model="m", tool_call_counter=counter)
        chunks_c = _translate_stream_event(_event("foo"), model="m", tool_call_counter=counter)

        assert chunks_a[0].choices[0].delta.tool_calls[0].index == 0
        assert chunks_b[0].choices[0].delta.tool_calls[0].index == 1
        assert chunks_c[0].choices[0].delta.tool_calls[0].index == 2

    def test_finish_reason_switches_to_tool_calls_when_any_seen(self):
        from agent.gemini_cloudcode_adapter import _translate_stream_event

        counter = [0]
        # First event emits one tool call.
        _translate_stream_event(
            {"response": {"candidates": [{
                "content": {"parts": [{"functionCall": {"name": "x", "args": {}}}]},
            }]}},
            model="m", tool_call_counter=counter,
        )
        # Second event carries only the terminal finishReason.
        chunks = _translate_stream_event(
            {"response": {"candidates": [{"finishReason": "STOP"}]}},
            model="m", tool_call_counter=counter,
        )
        assert chunks[-1].choices[0].finish_reason == "tool_calls"


class TestMakeStreamChunk:
    def test_reasoning_only_chunk_has_content_none(self):
        from agent.gemini_cloudcode_adapter import _make_stream_chunk

        chunk = _make_stream_chunk(model="m", reasoning="think")
        delta = chunk.choices[0].delta
        assert delta.content is None
        assert delta.reasoning == "think"

    def test_content_only_chunk_has_reasoning_none(self):
        from agent.gemini_cloudcode_adapter import _make_stream_chunk

        chunk = _make_stream_chunk(model="m", content="hello")
        delta = chunk.choices[0].delta
        assert delta.content == "hello"
        assert delta.reasoning is None
        assert delta.tool_calls is None

    def test_finish_only_chunk_has_all_fields_none(self):
        from agent.gemini_cloudcode_adapter import _make_stream_chunk

        chunk = _make_stream_chunk(model="m", finish_reason="stop")
        delta = chunk.choices[0].delta
        assert delta.content is None
        assert delta.reasoning is None
        assert delta.tool_calls is None
        assert chunk.choices[0].finish_reason == "stop"


class TestGeminiCloudCodeClient:
    def test_client_exposes_openai_interface(self):
        from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

        client = GeminiCloudCodeClient(api_key="dummy")
        try:
            assert hasattr(client, "chat")
            assert hasattr(client.chat, "completions")
            assert callable(client.chat.completions.create)
        finally:
            client.close()


class TestGeminiHttpErrorParsing:
    """Regression coverage for _gemini_http_error Google-envelope parsing.

    These are the paths that users actually hit during Google-side throttling
    (April 2026: gemini-2.5-pro MODEL_CAPACITY_EXHAUSTED, gemma-4-26b-it
    returning 404).  The error needs to carry status_code + response so the
    main loop's error_classifier and Retry-After logic work.
    """

    @staticmethod
    def _fake_response(status: int, body: dict | str = "", headers=None):
        """Minimal httpx.Response stand-in (duck-typed for _gemini_http_error)."""
        class _FakeResponse:
            def __init__(self):
                self.status_code = status
                if isinstance(body, dict):
                    self.text = json.dumps(body)
                else:
                    self.text = body
                self.headers = headers or {}
        return _FakeResponse()

    def test_model_capacity_exhausted_produces_friendly_message(self):
        from agent.gemini_cloudcode_adapter import _gemini_http_error

        body = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "MODEL_CAPACITY_EXHAUSTED",
                        "domain": "googleapis.com",
                        "metadata": {"model": "gemini-2.5-pro"},
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "30s",
                    },
                ],
            }
        }
        err = _gemini_http_error(self._fake_response(429, body))
        assert err.status_code == 429
        assert err.code == "code_assist_capacity_exhausted"
        assert err.retry_after == 30.0
        assert err.details["reason"] == "MODEL_CAPACITY_EXHAUSTED"
        # Message must be user-friendly, not a raw JSON dump.
        message = str(err)
        assert "gemini-2.5-pro" in message
        assert "capacity exhausted" in message.lower()
        assert "30s" in message
        # response attr is preserved for run_agent's Retry-After header path.
        assert err.response is not None

    def test_resource_exhausted_without_reason(self):
        from agent.gemini_cloudcode_adapter import _gemini_http_error

        body = {
            "error": {
                "code": 429,
                "message": "Quota exceeded for requests per minute.",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
        err = _gemini_http_error(self._fake_response(429, body))
        assert err.status_code == 429
        assert err.code == "code_assist_rate_limited"
        message = str(err)
        assert "quota" in message.lower()

    def test_404_model_not_found_produces_model_retired_message(self):
        from agent.gemini_cloudcode_adapter import _gemini_http_error

        body = {
            "error": {
                "code": 404,
                "message": "models/gemma-4-26b-it is not found for API version v1internal",
                "status": "NOT_FOUND",
            }
        }
        err = _gemini_http_error(self._fake_response(404, body))
        assert err.status_code == 404
        message = str(err)
        assert "not available" in message.lower() or "retired" in message.lower()
        # Error message should reference the actual model text from Google.
        assert "gemma-4-26b-it" in message

    def test_unauthorized_preserves_status_code(self):
        from agent.gemini_cloudcode_adapter import _gemini_http_error

        err = _gemini_http_error(self._fake_response(
            401, {"error": {"code": 401, "message": "Invalid token", "status": "UNAUTHENTICATED"}},
        ))
        assert err.status_code == 401
        assert err.code == "code_assist_unauthorized"

    def test_retry_after_header_fallback(self):
        """If the body has no RetryInfo detail, fall back to Retry-After header."""
        from agent.gemini_cloudcode_adapter import _gemini_http_error

        resp = self._fake_response(
            429,
            {"error": {"code": 429, "message": "Rate limited", "status": "RESOURCE_EXHAUSTED"}},
            headers={"Retry-After": "45"},
        )
        err = _gemini_http_error(resp)
        assert err.retry_after == 45.0

    def test_malformed_body_still_produces_structured_error(self):
        """Non-JSON body must not swallow status_code — we still want the classifier path."""
        from agent.gemini_cloudcode_adapter import _gemini_http_error

        err = _gemini_http_error(self._fake_response(500, "<html>internal error</html>"))
        assert err.status_code == 500
        # Raw body snippet must still be there for debugging.
        assert "500" in str(err)

    def test_status_code_flows_through_error_classifier(self):
        """End-to-end: CodeAssistError from a 429 must classify as rate_limit.

        This is the whole point of adding status_code to CodeAssistError —
        _extract_status_code must see it and FailoverReason.rate_limit must
        fire, so the main loop triggers fallback_providers.
        """
        from agent.gemini_cloudcode_adapter import _gemini_http_error
        from agent.error_classifier import classify_api_error, FailoverReason

        body = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "MODEL_CAPACITY_EXHAUSTED",
                        "metadata": {"model": "gemini-2.5-pro"},
                    }
                ],
            }
        }
        err = _gemini_http_error(self._fake_response(429, body))

        classified = classify_api_error(
            err, provider="google-gemini-cli", model="gemini-2.5-pro",
        )
        assert classified.status_code == 429
        assert classified.reason == FailoverReason.rate_limit


# =============================================================================
# Provider registration
# =============================================================================

class TestProviderRegistration:
    def test_registry_entry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "google-gemini-cli" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["google-gemini-cli"].auth_type == "oauth_external"

    def test_google_gemini_alias_still_goes_to_api_key_gemini(self):
        """Regression guard: don't shadow the existing google-gemini → gemini alias."""
        from hermes_cli.auth import resolve_provider

        assert resolve_provider("google-gemini") == "gemini"

    def test_runtime_provider_raises_when_not_logged_in(self):
        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with pytest.raises(AuthError) as exc_info:
            resolve_runtime_provider(requested="google-gemini-cli")
        assert exc_info.value.code == "google_oauth_not_logged_in"

    def test_runtime_provider_returns_correct_shape_when_logged_in(self):
        from agent.google_oauth import GoogleCredentials, save_credentials
        from hermes_cli.runtime_provider import resolve_runtime_provider

        save_credentials(GoogleCredentials(
            access_token="live-tok",
            refresh_token="rt",
            expires_ms=int((time.time() + 3600) * 1000),
            project_id="my-proj",
            email="t@e.com",
        ))

        result = resolve_runtime_provider(requested="google-gemini-cli")
        assert result["provider"] == "google-gemini-cli"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "live-tok"
        assert result["base_url"] == "cloudcode-pa://google"
        assert result["project_id"] == "my-proj"
        assert result["email"] == "t@e.com"

    def test_determine_api_mode(self):
        from hermes_cli.providers import determine_api_mode

        assert determine_api_mode("google-gemini-cli", "cloudcode-pa://google") == "chat_completions"

    def test_oauth_capable_set_preserves_existing(self):
        from hermes_cli.auth_commands import _OAUTH_CAPABLE_PROVIDERS

        for required in ("anthropic", "nous", "openai-codex", "qwen-oauth", "google-gemini-cli"):
            assert required in _OAUTH_CAPABLE_PROVIDERS

    def test_config_env_vars_registered(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for key in (
            "HERMES_GEMINI_CLIENT_ID",
            "HERMES_GEMINI_CLIENT_SECRET",
            "HERMES_GEMINI_PROJECT_ID",
        ):
            assert key in OPTIONAL_ENV_VARS


class TestAuthStatus:
    def test_not_logged_in(self):
        from hermes_cli.auth import get_auth_status

        s = get_auth_status("google-gemini-cli")
        assert s["logged_in"] is False

    def test_logged_in_reports_email_and_project(self):
        from agent.google_oauth import GoogleCredentials, save_credentials
        from hermes_cli.auth import get_auth_status

        save_credentials(GoogleCredentials(
            access_token="tok", refresh_token="rt",
            expires_ms=int((time.time() + 3600) * 1000),
            email="tek@nous.ai",
            project_id="tek-proj",
        ))

        s = get_auth_status("google-gemini-cli")
        assert s["logged_in"] is True
        assert s["email"] == "tek@nous.ai"
        assert s["project_id"] == "tek-proj"


class TestGquotaCommand:
    def test_gquota_registered(self):
        from hermes_cli.commands import COMMANDS

        assert "/gquota" in COMMANDS


class TestRunGeminiOauthLoginPure:
    def test_returns_pool_compatible_dict(self, monkeypatch):
        from agent import google_oauth

        def fake_start(**kw):
            return google_oauth.GoogleCredentials(
                access_token="at", refresh_token="rt",
                expires_ms=int((time.time() + 3600) * 1000),
                email="u@e.com", project_id="p",
            )

        monkeypatch.setattr(google_oauth, "start_oauth_flow", fake_start)

        result = google_oauth.run_gemini_oauth_login_pure()
        assert result["access_token"] == "at"
        assert result["refresh_token"] == "rt"
        assert result["email"] == "u@e.com"
        assert result["project_id"] == "p"
        assert isinstance(result["expires_at_ms"], int)

"""Tests for the google-antigravity (Antigravity subscription) provider.

The provider is an opt-in, OAuth-backed path to Google's Code Assist
endpoint. It must never change `gemini` (API-key) behavior.
"""

import time

import pytest


# ── Slice 1: provider profile registration & aliases ──

class TestAntigravityProfile:
    def test_profile_is_registered(self):
        from providers import get_provider_profile

        profile = get_provider_profile("google-antigravity")
        assert profile is not None
        assert profile.name == "google-antigravity"

    @pytest.mark.parametrize("alias", ["antigravity", "agy"])
    def test_aliases_resolve(self, alias):
        from providers import get_provider_profile

        profile = get_provider_profile(alias)
        assert profile is not None
        assert profile.name == "google-antigravity"

    def test_uses_oauth_not_api_key(self):
        from providers import get_provider_profile

        profile = get_provider_profile("google-antigravity")
        assert profile.auth_type == "oauth_external"

    def test_targets_code_assist_endpoint(self):
        from providers import get_provider_profile

        profile = get_provider_profile("google-antigravity")
        assert profile.base_url == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"

    def test_hermes_overlay_targets_daily_code_assist_endpoint(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["google-antigravity"]
        assert (
            overlay.base_url_override
            == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"
        )

    def test_curated_gemini_fallback_models(self):
        """Fallback list must be Gemini-only (no Claude/GPT) and non-empty."""
        from providers import get_provider_profile

        profile = get_provider_profile("google-antigravity")
        assert profile.fallback_models
        assert all(m.startswith("gemini-") for m in profile.fallback_models)

    def test_does_not_disturb_gemini_profile(self):
        from providers import get_provider_profile

        gemini = get_provider_profile("gemini")
        assert gemini.name == "gemini"
        assert gemini.auth_type == "api_key"
        assert gemini.base_url == "https://generativelanguage.googleapis.com/v1beta"


# ── Slice 2: client identity resolution (env override + agy discovery) ──

class TestClientIdentity:
    def test_env_overrides_win(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_ID", "cid-from-env")
        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_SECRET", "csec-from-env")
        ident = ag.resolve_client_identity()
        assert ident["client_id"] == "cid-from-env"
        assert ident["client_secret"] == "csec-from-env"
        assert ident["source"] == "env"

    def test_extracts_adjacent_compiled_client_secrets(self):
        from hermes_cli import antigravity_auth as ag

        client_id = "123456789012-" + ("a" * 32) + ".apps.googleusercontent.com"
        first_secret = "GOCSPX-" + ("A" * 28)
        second_secret = "GOCSPX-" + ("B" * 28)
        blob = client_id + "\n" + first_secret + second_secret

        identity = ag._extract_client_identity_from_strings(blob)

        assert identity == {
            "client_id": client_id,
            "client_secret": first_secret,
            "source": "agy",
        }

    def test_prefers_cli_client_id_near_cloud_code_marker(self):
        from hermes_cli import antigravity_auth as ag

        desktop_id = "111111111111-" + ("d" * 32) + ".apps.googleusercontent.com"
        cli_id = "222222222222-" + ("c" * 33) + ".apps.googleusercontent.com"
        desktop_secret = "GOCSPX-" + ("A" * 28)
        cli_secret = "GOCSPX-" + ("B" * 28)
        blob = (
            desktop_id
            + desktop_secret
            + " unrelated desktop auth data "
            + "Overriding CloudCodeServerURL via CLOUD_CODE_URL "
            + cli_id
            + cli_secret
        )

        identity = ag._extract_client_identity_from_strings(blob)

        assert identity["client_id"] == cli_id
        assert identity["client_secret"] == cli_secret

    def test_pairs_marked_cli_client_by_candidate_ordinal_in_common_layout(self):
        from hermes_cli import antigravity_auth as ag

        desktop_id = "111111111111-" + ("d" * 32) + ".apps.googleusercontent.com"
        cli_id = "222222222222-" + ("c" * 33) + ".apps.googleusercontent.com"
        desktop_secret = "GOCSPX-" + ("D" * 28)
        cli_secret = "GOCSPX-" + ("C" * 28)
        blob = (
            desktop_id
            + desktop_secret
            + " unrelated desktop auth data "
            + "Overriding CloudCodeServerURL via CLOUD_CODE_URL "
            + cli_id
            + cli_secret
        )

        identity = ag._extract_client_identity_from_strings(blob)

        assert identity == {
            "client_id": cli_id,
            "client_secret": cli_secret,
            "source": "agy",
        }

    def test_pairs_marked_cli_client_by_candidate_ordinal(self):
        from hermes_cli import antigravity_auth as ag

        desktop_id = "111111111111-" + ("d" * 32) + ".apps.googleusercontent.com"
        cli_id = "222222222222-" + ("c" * 33) + ".apps.googleusercontent.com"
        desktop_secret = "GOCSPX-" + ("D" * 28)
        cli_secret = "GOCSPX-" + ("C" * 28)
        blob = (
            desktop_id
            + "Overriding CloudCodeServerURL via CLOUD_CODE_URL "
            + cli_id
            + desktop_secret
            + " unrelated filler "
            + cli_secret
        )

        identity = ag._extract_client_identity_from_strings(blob)

        assert identity == {
            "client_id": cli_id,
            "client_secret": cli_secret,
            "source": "agy",
        }

    def test_fails_clearly_when_no_identity_available(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.delenv("HERMES_ANTIGRAVITY_CLIENT_ID", raising=False)
        monkeypatch.delenv("HERMES_ANTIGRAVITY_CLIENT_SECRET", raising=False)
        monkeypatch.setattr(ag, "_discover_client_identity_from_agy", lambda: None)
        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.resolve_client_identity()
        assert "HERMES_ANTIGRAVITY_CLIENT_ID" in str(exc.value)

    def test_error_never_echoes_a_secret(self, monkeypatch):
        """A failure must not leak whatever partial secret we did have."""
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_ID", "cid-x")
        monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_SECRET", "GOCSPX-supersecret")
        monkeypatch.delenv("HERMES_ANTIGRAVITY_CLIENT_ID", raising=False)
        monkeypatch.setattr(ag, "_discover_client_identity_from_agy", lambda: None)
        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.resolve_client_identity()
        assert "GOCSPX-supersecret" not in str(exc.value)


# ── Slice 2b: credential redaction ──

class TestRedaction:
    def test_status_payload_has_no_tokens(self):
        from hermes_cli import antigravity_auth as ag

        state = {
            "access_token": "ya29.SECRET-ACCESS",
            "refresh_token": "1//SECRET-REFRESH",
            "id_token": "eyJhbGciOi.SECRET",
            "client_secret": "GOCSPX-SECRET",
            "expires_at": 1234567890,
            "email": "user@example.com",
            "project_id": "proj-123",
        }
        summary = ag.redact_state(state)
        blob = repr(summary)
        for secret in ("ya29.SECRET-ACCESS", "1//SECRET-REFRESH", "eyJhbGciOi.SECRET", "GOCSPX-SECRET"):
            assert secret not in blob
        # Non-sensitive fields survive so status output stays useful.
        assert summary["email"] == "user@example.com"
        assert summary["project_id"] == "proj-123"

    def test_auth_error_str_is_redacted(self):
        from hermes_cli import antigravity_auth as ag

        exc = ag.AntigravityAuthError(
            "token exchange failed: refresh_token=1//LEAKED access_token=ya29.LEAKED"
        )
        assert "1//LEAKED" not in str(exc)
        assert "ya29.LEAKED" not in str(exc)
        assert "[redacted]" in str(exc)


# ── Slice 3: Code Assist request/response translation ──

class TestCodeAssistTranslation:
    def test_request_wraps_gemini_body_with_project_and_model(self):
        from agent.gemini_cloudcode_adapter import build_code_assist_request

        wrapped = build_code_assist_request(
            model="gemini-3-pro-preview",
            gemini_request={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            project="proj-123",
        )
        assert wrapped["model"] == "gemini-3-pro-preview"
        assert wrapped["project"] == "proj-123"
        assert wrapped["request"]["contents"][0]["parts"][0]["text"] == "hi"

    def test_response_unwraps_and_preserves_tool_call_ids(self):
        """Code Assist nests the Gemini response under `response`."""
        from agent.gemini_cloudcode_adapter import translate_code_assist_response

        raw = {
            "response": {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {"name": "read_file", "args": {"path": "a.py"}},
                            "thoughtSignature": "sig-abc",
                        }],
                    },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 16,
                },
            }
        }
        resp = translate_code_assist_response(raw, "gemini-3-pro-preview")
        tool_calls = resp.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        call = tool_calls[0]
        assert call.function.name == "read_file"
        assert call.id  # stable, non-empty id Hermes can match a tool result to
        assert resp.usage.prompt_tokens == 11
        assert resp.usage.completion_tokens == 5

    def test_stream_events_are_unwrapped_before_gemini_translation(self):
        from agent.gemini_cloudcode_adapter import unwrap_code_assist_stream_event

        assert unwrap_code_assist_stream_event(
            {"response": {"candidates": [{"content": {"parts": [{"text": "yo"}]}}]}}
        ) == {"candidates": [{"content": {"parts": [{"text": "yo"}]}}]}
        # Already-bare events pass through untouched.
        assert unwrap_code_assist_stream_event({"candidates": []}) == {"candidates": []}


# ── Slice 4: entitled model discovery + curated fallback ──

class TestEntitledModels:
    def test_discovery_returns_entitled_gemini_models(self):
        from agent.gemini_cloudcode_adapter import parse_entitled_models

        payload = {
            "currentTier": {"id": "standard-tier"},
            "allowedModels": [
                {"modelId": "gemini-3-pro-preview"},
                {"modelId": "gemini-3-flash-preview"},
            ],
        }
        assert parse_entitled_models(payload) == [
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
        ]

    def test_non_gemini_models_are_filtered_out(self):
        """This provider surfaces Gemini only."""
        from agent.gemini_cloudcode_adapter import parse_entitled_models

        payload = {"allowedModels": [
            {"modelId": "claude-sonnet-4.5"},
            {"modelId": "gpt-5"},
            {"modelId": "gemini-3-pro-preview"},
        ]}
        assert parse_entitled_models(payload) == ["gemini-3-pro-preview"]

    def test_live_contract_models_mapping_uses_gemini_keys(self):
        from agent.gemini_cloudcode_adapter import parse_entitled_models

        payload = {
            "defaultAgentModelId": "gemini-3.1-pro-high",
            "models": {
                "claude-sonnet-4-6": {"displayName": "other vendor"},
                "gemini-3.1-pro-high": {"displayName": "Gemini Pro"},
                "gemini-3-flash-agent": {"displayName": "Gemini Flash"},
            },
            "agentModelSorts": [
                {
                    "displayName": "Recommended",
                    "groups": [
                        {"modelIds": ["gemini-3.1-pro-high", "claude-sonnet-4-6"]}
                    ],
                }
            ],
        }

        assert parse_entitled_models(payload) == [
            "gemini-3.1-pro-high",
            "gemini-3-flash-agent",
        ]

    def test_empty_or_malformed_discovery_returns_empty(self):
        from agent.gemini_cloudcode_adapter import parse_entitled_models

        assert parse_entitled_models({}) == []
        assert parse_entitled_models({"allowedModels": "nope"}) == []
        assert parse_entitled_models(None) == []

    def test_curated_fallback_used_when_discovery_fails(self, monkeypatch):
        from providers import get_provider_profile

        profile = get_provider_profile("google-antigravity")
        monkeypatch.setattr(
            "agent.gemini_cloudcode_adapter.discover_entitled_models",
            lambda **kw: None,
        )
        models = profile.fetch_models()
        # None → caller falls back to the curated list on the profile.
        assert models is None or models == list(profile.fallback_models)
        assert all(m.startswith("gemini-") for m in profile.fallback_models)


# ── Slice 5: token refresh, runtime credentials, auth registry wiring ──

class TestRefreshAndRuntime:
    def _state(self, **over):
        base = {
            "access_token": "ya29.OLD",
            "refresh_token": "1//KEEPME",
            "expires_at": 0,  # already expired
            "project_id": "proj-1",
            "email": "u@example.com",
        }
        base.update(over)
        return base

    def test_expired_token_is_refreshed(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(
            ag, "resolve_client_identity",
            lambda: {"client_id": "cid", "client_secret": "sec", "source": "env"},
        )
        calls = {}

        def fake_exchange(*, refresh_token, client_id, client_secret, timeout=30.0):
            calls["refresh_token"] = refresh_token
            return {"access_token": "ya29.NEW", "expires_in": 3600}

        monkeypatch.setattr(ag, "_refresh_access_token", fake_exchange)
        fresh = ag.ensure_fresh_access_token(self._state())
        assert fresh["access_token"] == "ya29.NEW"
        # Google omits refresh_token on refresh; the old one must survive.
        assert fresh["refresh_token"] == "1//KEEPME"
        assert fresh["expires_at"] > time.time()
        assert calls["refresh_token"] == "1//KEEPME"

    def test_unexpired_token_is_not_refreshed(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        def explode(**kw):
            raise AssertionError("must not refresh a live token")

        monkeypatch.setattr(ag, "_refresh_access_token", explode)
        state = self._state(expires_at=time.time() + 9999, access_token="ya29.LIVE")
        assert ag.ensure_fresh_access_token(state)["access_token"] == "ya29.LIVE"

    def test_missing_refresh_token_fails_clearly(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.ensure_fresh_access_token(self._state(refresh_token=""))
        assert "hermes auth add google-antigravity" in str(exc.value)

    def test_runtime_credentials_shape(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        state = self._state(expires_at=time.time() + 9999)
        monkeypatch.setattr(ag, "load_state_with_source", lambda: (state, None))
        creds = ag.resolve_antigravity_runtime_credentials()
        assert creds["provider"] == "google-antigravity"
        assert creds["api_key"] == "ya29.OLD"
        assert creds["base_url"] == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"
        assert creds["project_id"] == "proj-1"

    def test_runtime_credentials_require_login(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(ag, "load_state", lambda: None)
        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.resolve_antigravity_runtime_credentials()
        assert "hermes auth add google-antigravity" in str(exc.value)

    def test_status_is_redacted_and_reports_logged_out(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(ag, "load_state", lambda: None)
        status = ag.get_antigravity_auth_status()
        assert status["authenticated"] is False
        monkeypatch.setattr(ag, "load_state", lambda: self._state())
        status = ag.get_antigravity_auth_status()
        assert status["authenticated"] is True
        assert "ya29.OLD" not in repr(status)
        assert "1//KEEPME" not in repr(status)


class TestAuthRegistryWiring:
    def test_provider_in_auth_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "google-antigravity" in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["google-antigravity"]
        assert cfg.auth_type == "oauth_external"
        assert cfg.inference_base_url == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"

    def test_no_client_secret_is_checked_in(self):
        """Scan repo source for a concrete-looking checked-in secret.

        A real Google desktop-client secret is ``GOCSPX-`` followed by ~28
        base62 chars with no separators. Regex literals that merely *match*
        that shape (``GOCSPX-[A-Za-z0-9_-]{20,40}``) contain a character
        class — ``[``  — right after the prefix, so they're distinguishable
        from a literal secret without hand-listing file paths to exempt.
        """
        import re
        import subprocess

        literal_secret = re.compile(r"GOCSPX-(?!\[)[A-Za-z0-9_\-]{15,}")
        tracked = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()

        offenders = []
        for rel in tracked:
            if not rel.endswith((".py", ".yaml", ".yml", ".json", ".md", ".txt")):
                continue
            try:
                text = open(rel, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if literal_secret.search(text):
                offenders.append(rel)
        assert offenders == []


# ── Slice 6: full localhost PKCE login flow ──

class TestPKCELogin:
    def test_authorize_url_has_pkce_and_state(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(
            ag, "resolve_client_identity",
            lambda: {"client_id": "cid", "client_secret": "sec", "source": "env"},
        )
        url = ag.build_authorize_url(
            redirect_uri="http://127.0.0.1:8765/callback",
            code_challenge="chal123",
            state="state123",
        )
        assert url.startswith(ag.GOOGLE_AUTH_URL)
        assert "code_challenge=chal123" in url
        assert "code_challenge_method=S256" in url
        assert "state=state123" in url
        assert "client_id=cid" in url
        assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8765%2Fcallback" in url

    def test_redirect_uri_must_be_loopback(self):
        from hermes_cli import antigravity_auth as ag

        with pytest.raises(ag.AntigravityAuthError):
            ag._validate_antigravity_redirect_uri("http://evil.example.com/callback")
        with pytest.raises(ag.AntigravityAuthError):
            ag._validate_antigravity_redirect_uri("https://127.0.0.1:8765/callback")
        host, port, path = ag._validate_antigravity_redirect_uri(
            "http://127.0.0.1:8765/callback"
        )
        assert (host, port, path) == ("127.0.0.1", 8765, "/callback")

    def test_browser_login_does_not_print_authorization_url(
        self, monkeypatch, capsys
    ):
        from hermes_cli import antigravity_auth as ag

        identity = {
            "client_id": "123456789012-" + ("a" * 32) + ".apps.googleusercontent.com",
            "client_secret": "GOCSPX-" + ("S" * 28),
        }
        monkeypatch.setattr(ag, "resolve_client_identity", lambda: identity)

        def fake_wait(redirect_uri, *, timeout_seconds=180.0, on_ready=None):
            assert on_ready is not None
            on_ready()
            return {"code": "AUTH-CODE", "state": "EXPECTED-STATE", "error": None}

        opened = []
        monkeypatch.setattr(ag, "_wait_for_antigravity_callback", fake_wait)
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
        monkeypatch.setattr(
            ag,
            "_exchange_code_for_tokens",
            lambda **kw: {
                "access_token": "ya29.NEW",
                "refresh_token": "1//NEW",
                "expires_in": 3600,
            },
        )
        monkeypatch.setattr(ag, "_fetch_userinfo", lambda access_token: {})
        monkeypatch.setattr(ag, "save_state", lambda state, **kw: None)

        ag.run_pkce_login(
            open_browser=True,
            _state_override="EXPECTED-STATE",
            _code_verifier_override="verifier",
        )

        stdout = capsys.readouterr().out
        assert len(opened) == 1
        assert identity["client_id"] in opened[0]
        assert "accounts.google.com" not in stdout
        assert identity["client_id"] not in stdout
        assert "EXPECTED-STATE" not in stdout

    def test_callback_state_mismatch_is_rejected(self, monkeypatch):
        """A callback carrying a different state than we generated must be
        treated as a failed login, not accepted (CSRF/session-fixation guard).
        """
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(
            ag, "_wait_for_antigravity_callback",
            lambda redirect_uri, *, timeout_seconds=180.0, on_ready=None: {
                "code": "abc", "state": "WRONG-STATE", "error": None,
            },
        )
        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.run_pkce_login(
                open_browser=False,
                redirect_uri="http://127.0.0.1:8765/callback",
                timeout_seconds=5,
                _state_override="EXPECTED-STATE",
                _code_verifier_override="verifier",
                _client_identity_override={"client_id": "cid", "client_secret": "sec"},
            )
        assert "state" in str(exc.value).lower()

    def test_callback_error_is_surfaced_without_leaking_body(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(
            ag, "_wait_for_antigravity_callback",
            lambda redirect_uri, *, timeout_seconds=180.0, on_ready=None: {
                "code": None, "state": "EXPECTED-STATE",
                "error": "access_denied", "error_description": "user said no",
            },
        )
        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.run_pkce_login(
                open_browser=False,
                redirect_uri="http://127.0.0.1:8765/callback",
                timeout_seconds=5,
                _state_override="EXPECTED-STATE",
                _code_verifier_override="verifier",
                _client_identity_override={"client_id": "cid", "client_secret": "sec"},
            )
        assert "access_denied" in str(exc.value)

    def test_callback_timeout_raises_clearly(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        def _timeout(redirect_uri, *, timeout_seconds=180.0, on_ready=None):
            raise ag.AntigravityAuthError(
                "Antigravity authorization timed out waiting for the local callback.",
                code="antigravity_callback_timeout",
            )

        monkeypatch.setattr(ag, "_wait_for_antigravity_callback", _timeout)
        with pytest.raises(ag.AntigravityAuthError) as exc:
            ag.run_pkce_login(
                open_browser=False,
                redirect_uri="http://127.0.0.1:8765/callback",
                timeout_seconds=5,
                _state_override="EXPECTED-STATE",
                _code_verifier_override="verifier",
                _client_identity_override={"client_id": "cid", "client_secret": "sec"},
            )
        assert exc.value.code == "antigravity_callback_timeout"

    def test_successful_login_exchanges_code_and_persists_state(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(
            ag, "_wait_for_antigravity_callback",
            lambda redirect_uri, *, timeout_seconds=180.0, on_ready=None: {
                "code": "auth-code-xyz", "state": "EXPECTED-STATE", "error": None,
            },
        )

        exchange_calls = {}

        def fake_exchange(*, client_id, client_secret, code, redirect_uri, code_verifier, timeout=30.0):
            exchange_calls["code"] = code
            exchange_calls["code_verifier"] = code_verifier
            return {
                "access_token": "ya29.BRANDNEW",
                "refresh_token": "1//BRANDNEW",
                "expires_in": 3600,
                "token_type": "Bearer",
            }

        monkeypatch.setattr(ag, "_exchange_code_for_tokens", fake_exchange)
        monkeypatch.setattr(ag, "_fetch_userinfo", lambda access_token: {"email": "u@x.com"})

        saved = {}
        monkeypatch.setattr(ag, "save_state", lambda state, **kw: saved.update(state) or saved.update(kw=kw))

        result = ag.run_pkce_login(
            open_browser=False,
            redirect_uri="http://127.0.0.1:8765/callback",
            timeout_seconds=5,
            _state_override="EXPECTED-STATE",
            _code_verifier_override="verifier-xyz",
            _client_identity_override={"client_id": "cid", "client_secret": "sec"},
        )
        assert exchange_calls["code"] == "auth-code-xyz"
        assert exchange_calls["code_verifier"] == "verifier-xyz"
        assert result["access_token"] == "ya29.BRANDNEW"
        assert saved["access_token"] == "ya29.BRANDNEW"
        assert saved["refresh_token"] == "1//BRANDNEW"
        assert saved["email"] == "u@x.com"

    def test_login_never_logs_the_authorization_code_or_tokens(self, monkeypatch, caplog):
        from hermes_cli import antigravity_auth as ag

        monkeypatch.setattr(
            ag, "_wait_for_antigravity_callback",
            lambda redirect_uri, *, timeout_seconds=180.0, on_ready=None: {
                "code": "SUPER-SECRET-CODE", "state": "EXPECTED-STATE", "error": None,
            },
        )
        monkeypatch.setattr(
            ag, "_exchange_code_for_tokens",
            lambda **kw: {"access_token": "ya29.ABC", "refresh_token": "1//DEF", "expires_in": 3600},
        )
        monkeypatch.setattr(ag, "_fetch_userinfo", lambda access_token: {})
        monkeypatch.setattr(ag, "save_state", lambda state, **kw: None)

        with caplog.at_level("DEBUG"):
            ag.run_pkce_login(
                open_browser=False,
                redirect_uri="http://127.0.0.1:8765/callback",
                timeout_seconds=5,
                _state_override="EXPECTED-STATE",
                _code_verifier_override="verifier",
                _client_identity_override={"client_id": "cid", "client_secret": "sec"},
            )
        blob = caplog.text
        assert "SUPER-SECRET-CODE" not in blob
        assert "ya29.ABC" not in blob
        assert "1//DEF" not in blob


class TestModelPickerFlow:
    def test_antigravity_flow_persists_live_model(self, monkeypatch):
        from hermes_cli import model_setup_flows as flows

        monkeypatch.setattr(
            "hermes_cli.antigravity_auth.get_antigravity_auth_status",
            lambda: {"logged_in": True},
        )
        monkeypatch.setattr(
            "providers.get_provider_profile",
            lambda provider: type(
                "Profile",
                (),
                {
                    "base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
                    "fallback_models": ("gemini-fallback",),
                    "fetch_models": lambda self, **kw: ["gemini-live-model"],
                },
            )(),
        )
        monkeypatch.setattr(
            "hermes_cli.auth._prompt_model_selection",
            lambda models, **kw: models[0],
        )
        saved = {}
        monkeypatch.setattr(
            "hermes_cli.auth._save_model_choice",
            lambda model: saved.update(model=model),
        )
        monkeypatch.setattr(
            "hermes_cli.auth._update_config_for_provider",
            lambda provider, base_url: saved.update(
                provider=provider, base_url=base_url
            ),
        )

        flows._model_flow_antigravity({}, current_model="")

        assert saved == {
            "model": "gemini-live-model",
            "provider": "google-antigravity",
            "base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
        }

    def test_main_dispatches_google_submenu_to_antigravity_flow(self, monkeypatch):
        from hermes_cli import main

        def choose(choices, *, default=0, title="Select provider:"):
            needle = "Google Antigravity" if "Google Gemini" in title else "Google Gemini"
            return next(i for i, choice in enumerate(choices) if needle in choice)

        monkeypatch.setattr(main, "_prompt_provider_choice", choose)
        called = []
        monkeypatch.setattr(
            main,
            "_model_flow_antigravity",
            lambda config, current_model="", args=None: called.append(True),
        )
        monkeypatch.setattr(main, "_clear_stale_openai_base_url", lambda: None)

        main.select_provider_and_model()

        assert called == [True]


class TestAuthCommandsWiring:
    def test_auth_status_dispatches_to_antigravity(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag
        from hermes_cli import auth as auth_mod

        monkeypatch.setattr(
            ag, "get_antigravity_auth_status",
            lambda: {"logged_in": True, "provider": "google-antigravity"},
        )
        status = auth_mod.get_auth_status("google-antigravity")
        assert status["logged_in"] is True

    def test_auth_add_calls_pkce_login_and_stores_pool_entry(self, monkeypatch, tmp_path):
        """`hermes auth add google-antigravity` must run the real PKCE login,
        not silently no-op, and must land a credential the pool can see.
        """
        import json
        from argparse import Namespace

        from hermes_cli import auth as auth_mod
        from hermes_cli import auth_commands

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        monkeypatch.setattr(
            "hermes_cli.antigravity_auth.run_pkce_login",
            lambda **kw: {
                "access_token": "ya29.POOLED",
                "refresh_token": "1//POOLED",
                "expires_at": 9999999999.0,
                "email": "pooled@example.com",
            },
        )

        args = Namespace(
            provider="google-antigravity", type=None, label=None,
            client_id=None, scope=None, no_browser=True, timeout=5,
            insecure=False, ca_bundle=None,
        )
        auth_commands.auth_add_command(args)

        auth_file = tmp_path / "hermes" / "auth.json"
        assert auth_file.exists()
        stored = json.loads(auth_file.read_text())
        import os as _os

        assert (_os.stat(auth_file).st_mode & 0o777) == 0o600
        providers = stored.get("providers", {})
        pool = stored.get("credential_pool", {})
        assert "google-antigravity" in providers
        assert "google-antigravity" not in pool

    def test_auth_logout_uses_antigravity_revocation_path(self, monkeypatch):
        from argparse import Namespace

        from hermes_cli import auth_commands

        called = []
        monkeypatch.setattr(
            "hermes_cli.antigravity_auth.revoke_and_logout",
            lambda: called.append(True) or True,
        )
        auth_commands.auth_logout_command(Namespace(provider="google-antigravity"))
        assert called == [True]


class TestProfileSourceAwareRefresh:
    def test_refresh_persists_back_to_source_store(self, monkeypatch, tmp_path):
        from hermes_cli import antigravity_auth as ag

        source = tmp_path / "global" / "auth.json"
        initial = {
            "access_token": "ya29.OLD",
            "refresh_token": "1//OLD",
            "expires_at": 0,
        }
        monkeypatch.setattr(ag, "load_state_with_source", lambda: (dict(initial), source))
        monkeypatch.setattr(
            ag,
            "ensure_fresh_access_token",
            lambda state: {
                **state,
                "access_token": "ya29.NEW",
                "refresh_token": "1//ROTATED",
                "expires_at": 9999999999,
            },
        )
        saved = {}
        monkeypatch.setattr(
            ag,
            "save_state_to_source",
            lambda state, source_path: saved.update(
                state=dict(state), source_path=source_path
            ),
        )

        result = ag.resolve_antigravity_runtime_credentials()

        assert saved["source_path"] == source
        assert saved["state"]["refresh_token"] == "1//ROTATED"
        assert result["api_key"] == "ya29.NEW"


# ── Slice 7: project_id plumbing end-to-end ──

class TestProjectIdPlumbing:
    def test_runtime_provider_result_carries_project_id(self, monkeypatch):
        from hermes_cli import runtime_provider

        monkeypatch.setattr(
            "hermes_cli.antigravity_auth.resolve_antigravity_runtime_credentials",
            lambda: {
                "provider": "google-antigravity",
                "api_mode": "chat_completions",
                "base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
                "api_key": "ya29.TOK",
                "project_id": "proj-abc",
                "source": "hermes-auth-store",
            },
        )
        result = runtime_provider.resolve_runtime_provider(requested="google-antigravity")
        assert result["project_id"] == "proj-abc"

    def test_code_assist_client_accepts_and_uses_project(self):
        from agent.gemini_cloudcode_adapter import CodeAssistClient

        client = CodeAssistClient(api_key="ya29.TOK", project="proj-xyz")
        assert client._project == "proj-xyz"

    def test_create_openai_client_passes_project_to_code_assist_client(self, monkeypatch):
        """End-to-end: runtime creds carry project_id -> agent client construction
        must thread it into CodeAssistClient, not silently drop it.
        """
        from agent import agent_runtime_helpers as helpers

        captured = {}

        class _FakeCodeAssistClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "agent.gemini_cloudcode_adapter.CodeAssistClient", _FakeCodeAssistClient
        )

        class _FakeAgent:
            provider = "google-antigravity"
            _client_kwargs = {
                "api_key": "ya29.TOK",
                "base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
                "project_id": "proj-e2e",
            }

            def _build_keepalive_http_client(self, base_url, verify=None):
                return None

            def _client_log_context(self):
                return ""

        agent = _FakeAgent()
        helpers.create_openai_client(
            agent, dict(agent._client_kwargs), reason="test", shared=False
        )
        assert captured.get("project") == "proj-e2e"


# ── Slice 8: Code Assist transport contract (loadCodeAssist / fetchAvailableModels) ──

class TestResolveProjectContext:
    def test_configured_project_wins_without_a_network_call(self):
        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, resolve_project_context

        def handler(request):
            raise AssertionError("must not call loadCodeAssist when a project is configured")

        import httpx

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            project = resolve_project_context(
                access_token="tok-abc",
                configured_project="proj-configured",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
            )
        assert project == "proj-configured"

    def test_discovers_project_via_load_code_assist_exact_url(self):
        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, resolve_project_context

        import httpx

        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(200, json={"cloudaicompanionProject": "proj-discovered"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            project = resolve_project_context(
                access_token="tok-abc",
                configured_project=None,
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
            )
        assert captured["method"] == "POST"
        assert captured["url"] == f"{CODE_ASSIST_BASE_URL}:loadCodeAssist"
        assert project == "proj-discovered"

    def test_callback_ready_hook_runs_after_server_bind(self, monkeypatch):
        from hermes_cli import antigravity_auth as ag

        observed = []

        class _FakeServer:
            allow_reuse_address = True

            def __init__(self, address, handler):
                observed.append(("bound", address))

            def serve_forever(self, poll_interval=0.1):
                return None

            def shutdown(self):
                return None

            def server_close(self):
                return None

        monkeypatch.setattr(ag, "HTTPServer", _FakeServer)
        monkeypatch.setattr(ag.time, "sleep", lambda _: None)
        ticks = iter([0.0, 10.0])
        monkeypatch.setattr(ag.time, "monotonic", lambda: next(ticks))

        with pytest.raises(ag.AntigravityAuthError):
            ag._wait_for_antigravity_callback(
                ag.DEFAULT_ANTIGRAVITY_REDIRECT_URI,
                timeout_seconds=5,
                on_ready=lambda: observed.append(("ready", None)),
            )

        assert observed[0][0] == "bound"
        assert observed[1][0] == "ready"


class TestFetchAvailableModelsContract:
    def test_uses_fetch_available_models_and_filters_to_gemini(self):
        import httpx

        from agent.gemini_cloudcode_adapter import (
            CODE_ASSIST_BASE_URL,
            discover_entitled_models,
        )

        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = request.read().decode()
            return httpx.Response(
                200,
                json={
                    "agentModelSorts": [
                        {
                            "displayName": "Recommended",
                            "modelIds": [
                                "gemini-3.1-pro-high",
                                "claude-opus-4-6-thinking",
                                "gemini-3.1-pro-high",
                                {"modelId": "gemini-3.6-flash-high"},
                            ],
                        }
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            models = discover_entitled_models(
                access_token="tok-abc",
                project="proj-1",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=client,
            )

        assert captured["url"] == f"{CODE_ASSIST_BASE_URL}:fetchAvailableModels"
        assert '"project":"proj-1"' in captured["body"].replace(" ", "")
        assert models == ["gemini-3.1-pro-high", "gemini-3.6-flash-high"]

    def test_missing_project_is_resolved_before_fetching_models(self):
        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, discover_entitled_models

        urls = []

        def handler(request):
            urls.append(str(request.url))
            if str(request.url).endswith(":loadCodeAssist"):
                return httpx.Response(
                    200, json={"cloudaicompanionProject": "proj-discovered"}
                )
            return httpx.Response(200, json={"models": ["gemini-3.1-pro-high"]})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            models = discover_entitled_models(
                access_token="tok-abc",
                project=None,
                base_url=CODE_ASSIST_BASE_URL,
                http_client=client,
            )

        assert urls == [
            f"{CODE_ASSIST_BASE_URL}:loadCodeAssist",
            f"{CODE_ASSIST_BASE_URL}:fetchAvailableModels",
        ]
        assert models == ["gemini-3.1-pro-high"]


class TestAntigravityHeaders:
    def test_headers_include_required_metadata_and_unique_request_id(self):
        import json

        from agent.gemini_cloudcode_adapter import build_antigravity_headers

        first = build_antigravity_headers("ya29.TEST")
        second = build_antigravity_headers("ya29.TEST")

        assert first["Authorization"] == "Bearer ya29.TEST"
        assert first["User-Agent"].startswith("antigravity/")
        assert first["X-Goog-Api-Client"]
        assert json.loads(first["Client-Metadata"])["ideType"] == "ANTIGRAVITY"
        assert first["x-activity-request-id"] != second["x-activity-request-id"]


class TestCodeAssistProjectDiscovery:
    def test_first_completion_discovers_and_reuses_project(self):
        import json

        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, CodeAssistClient

        requests = []

        def handler(request):
            requests.append((str(request.url), json.loads(request.read())))
            if str(request.url).endswith(":loadCodeAssist"):
                return httpx.Response(
                    200, json={"cloudaicompanionProject": "proj-managed"}
                )
            return httpx.Response(
                200,
                json={
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "ok"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    }
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = CodeAssistClient(
                api_key="ya29.TEST",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
            )
            response = client._create_chat_completion(
                model="gemini-3.1-pro-high",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert response.choices[0].message.content == "ok"
        assert requests[0][0] == f"{CODE_ASSIST_BASE_URL}:loadCodeAssist"
        assert requests[1][0] == f"{CODE_ASSIST_BASE_URL}:generateContent"
        assert requests[1][1]["project"] == "proj-managed"

    def test_gemini3_completion_preserves_tool_call_ids_in_request(self):
        import json

        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, CodeAssistClient

        captured = {}

        def handler(request):
            captured.update(json.loads(request.read()))
            return httpx.Response(
                200,
                json={
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "ok"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    }
                },
            )

        messages = [
            {"role": "user", "content": "Read a.txt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"a.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "AAA"},
        ]

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = CodeAssistClient(
                api_key="ya29.TEST",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
                project="proj-managed",
            )
            client._create_chat_completion(
                model="gemini-3.1-pro-high", messages=messages
            )

        parts = [
            part
            for content in captured["request"]["contents"]
            for part in content["parts"]
        ]
        assert [
            part["functionCall"]["id"]
            for part in parts
            if "functionCall" in part
        ] == ["call_1"]
        assert [
            part["functionResponse"]["id"]
            for part in parts
            if "functionResponse" in part
        ] == ["call_1"]


class TestCodeAssistErrorRedaction:
    @pytest.mark.parametrize(
        "status,body",
        [
            (500, "upstream echoed access_token=ya29.SUPERSECRET"),
            (429, "refresh_token=1//SUPERSECRET code=AUTHCODESECRET"),
        ],
    )
    def test_http_errors_do_not_retain_raw_response_or_secrets(self, status, body):
        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, CodeAssistClient
        from agent.gemini_native_adapter import GeminiAPIError

        def handler(request):
            return httpx.Response(status, text=body)

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = CodeAssistClient(
                api_key="ya29.CLIENTSECRET",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
                project="proj-1",
            )
            with pytest.raises(GeminiAPIError) as exc:
                client._create_chat_completion(
                    model="gemini-3.1-pro-high",
                    messages=[{"role": "user", "content": "hello"}],
                )

        rendered = f"{exc.value} {exc.value.details!r}"
        assert "SUPERSECRET" not in rendered
        assert "CLIENTSECRET" not in rendered
        assert exc.value.response is None

    def test_stream_error_does_not_expose_body_or_response(self):
        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, CodeAssistClient
        from agent.gemini_native_adapter import GeminiAPIError

        def handler(request):
            return httpx.Response(429, text="access_token=ya29.STREAMSECRET")

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = CodeAssistClient(
                api_key="ya29.CLIENTSECRET",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
                project="proj-1",
            )
            stream = client._create_chat_completion(
                model="gemini-3.1-pro-high",
                stream=True,
                messages=[{"role": "user", "content": "hello"}],
            )
            with pytest.raises(GeminiAPIError) as exc:
                list(stream)

        assert "STREAMSECRET" not in str(exc.value)
        assert "CLIENTSECRET" not in str(exc.value)
        assert exc.value.response is None

    @pytest.mark.parametrize("error_type", ["read", "timeout"])
    def test_stream_transport_error_retains_no_httpx_request(self, error_type):
        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, CodeAssistClient
        from agent.gemini_native_adapter import GeminiAPIError

        def handler(request):
            if error_type == "timeout":
                raise httpx.ReadTimeout("secret timeout", request=request)
            raise httpx.ReadError("secret read", request=request)

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = CodeAssistClient(
                api_key="ya29.CLIENTSECRET",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
                project="proj-1",
            )
            stream = client._create_chat_completion(
                model="gemini-3.1-pro-high",
                stream=True,
                messages=[{"role": "user", "content": "hello"}],
            )
            with pytest.raises(GeminiAPIError) as exc:
                list(stream)

        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None
        assert "CLIENTSECRET" not in str(exc.value)


class TestCodeAssistStreamingClient:
    def test_stream_preserves_text_reasoning_usage_and_stable_tool_id(self):
        import json

        import httpx

        from agent.gemini_cloudcode_adapter import CODE_ASSIST_BASE_URL, CodeAssistClient

        events = [
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "reason"},
                                    {"text": "hello"},
                                    {
                                        "functionCall": {
                                            "name": "read_file",
                                            "args": {"path": "README.md"},
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {},
                                    {},
                                    {
                                        "functionCall": {
                                            "name": "read_file",
                                            "args": {"path": "README.md"},
                                        }
                                    },
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 10,
                    },
                }
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

        def handler(request):
            return httpx.Response(200, content=body.encode())

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = CodeAssistClient(
                api_key="ya29.TEST",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=http_client,
                project="proj-1",
            )
            chunks = list(
                client._create_chat_completion(
                    model="gemini-3.1-pro-high",
                    stream=True,
                    messages=[{"role": "user", "content": "hello"}],
                )
            )

        assert any(c.choices[0].delta.content == "hello" for c in chunks)
        assert any(c.choices[0].delta.reasoning == "reason" for c in chunks)
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]
        assert len(tool_chunks) == 2
        assert tool_chunks[0].choices[0].delta.tool_calls[0].id == tool_chunks[1].choices[0].delta.tool_calls[0].id
        finish = next(c for c in chunks if c.choices[0].finish_reason)
        assert finish.choices[0].finish_reason == "tool_calls"
        assert finish.usage.prompt_tokens == 7


class TestQuotaSummaryParsing:
    """Normalization of ``:retrieveUserQuotaSummary`` payloads (PR2).

    Every assertion here is on the *normalized* output: the parser must never
    carry account identity, description text, or raw payload fragments through.
    """

    def test_parses_grouped_weekly_and_five_hour_windows(self):
        from agent.gemini_cloudcode_adapter import parse_quota_summary

        payload = {
            "groups": [
                {
                    "displayName": "user@example.com must never escape",
                    "description": "private description",
                    "buckets": [
                        {
                            "bucketId": "gemini-weekly",
                            "displayName": "account@example.com weekly",
                            "remainingFraction": 0.62,
                            "resetTime": "2026-08-25T09:00:00Z",
                            "window": "weekly",
                        },
                        {
                            "bucketId": "gemini-5h",
                            "remainingFraction": 0.25,
                            "resetTime": "2026-08-22T14:30:00Z",
                            "window": "5h",
                        },
                    ],
                },
            ],
            # Top-level entries can contain both grouped duplicates and
            # distinct pools.  Combine both sources and deduplicate by ID.
            "buckets": [
                {
                    "bucketId": "gemini-weekly",
                    "remainingFraction": 0.62,
                    "window": "weekly",
                },
                {
                    "bucketId": "3p-weekly",
                    "displayName": "arbitrary provider text",
                    "remainingFraction": 0.9,
                    "resetTime": "2026-08-29T09:00:00Z",
                    "window": "weekly",
                },
                {
                    "bucketId": "3p-5h",
                    "remainingFraction": 0.4,
                    "resetTime": "2026-08-22T15:00:00Z",
                    "window": "5h",
                },
            ],
        }

        windows = parse_quota_summary(payload)

        assert [(w.label, w.used_percent) for w in windows] == [
            ("Gemini Models (weekly)", pytest.approx(38.0)),
            ("Gemini Models (5h)", pytest.approx(75.0)),
            ("Claude and GPT models (weekly)", pytest.approx(10.0)),
            ("Claude and GPT models (5h)", pytest.approx(60.0)),
        ]
        blob = " ".join(f"{w.label} {w.detail or ''}" for w in windows)
        assert "example.com" not in blob
        assert "private description" not in blob

    def test_clamps_fractions_and_skips_malformed_buckets(self):
        from agent.gemini_cloudcode_adapter import parse_quota_summary

        payload = {
            "buckets": [
                {"bucketId": "gemini-weekly", "remainingFraction": 1.4},
                {"bucketId": "gemini-5h", "remainingFraction": -0.5},
                {"bucketId": "3p-weekly", "remainingFraction": float("nan")},
                {"bucketId": "3p-weekly", "remainingFraction": 10**1000},
                {"bucketId": "3p-weekly", "remainingFraction": 0.5},
                {"bucketId": "3p-5h", "remainingFraction": "0.5"},
                {"bucketId": "3p-5h", "remainingFraction": True},
                {"bucketId": "3p-5h", "remainingAmount": "500"},
                {"bucketId": "future-daily", "remainingFraction": 0.5},
                {
                    "bucketId": "3p-5h",
                    "remainingFraction": 0.5,
                    "disabled": True,
                },
                "not-a-bucket",
                {"remainingFraction": 0.5},
            ]
        }

        windows = parse_quota_summary(payload)

        assert [(w.label, w.used_percent) for w in windows] == [
            ("Gemini Models (weekly)", 0.0),
            ("Gemini Models (5h)", 100.0),
            ("Claude and GPT models (weekly)", 50.0),
        ]

    def test_preserves_reset_timestamps_without_identity_leakage(self):
        from agent.gemini_cloudcode_adapter import parse_quota_summary

        payload = {
            "groups": [
                {
                    "displayName": "user@example.com",
                    "buckets": [
                        {
                            "bucketId": "gemini-weekly",
                            "remainingFraction": 0.5,
                            "resetTime": 10**100,
                            "userEmail": "user@example.com",
                            "description": "secret description",
                        },
                        {
                            "bucketId": "gemini-5h",
                            "remainingFraction": 0.4,
                            "resetTime": "2026-08-25T09:00:00Z",
                        },
                    ],
                }
            ]
        }

        windows = parse_quota_summary(payload)

        assert len(windows) == 2
        assert windows[0].reset_at is None
        assert windows[1].reset_at is not None
        assert "example.com" not in repr(windows)
        assert "secret description" not in repr(windows)

    def test_unsupported_payloads_return_no_windows(self):
        from agent.gemini_cloudcode_adapter import parse_quota_summary

        assert parse_quota_summary(None) == []
        assert parse_quota_summary({}) == []
        assert parse_quota_summary({"quotaBuckets": "nope"}) == []
        assert parse_quota_summary([1, 2, 3]) == []


class TestQuotaSummaryFetch:
    """The ``:retrieveUserQuotaSummary`` HTTP contract (PR2)."""

    def test_posts_to_quota_summary_endpoint_with_project_only(self):
        import json as _json

        import httpx

        from agent.gemini_cloudcode_adapter import (
            CODE_ASSIST_BASE_URL,
            fetch_quota_summary,
        )

        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = _json.loads(request.read().decode())
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={
                    "buckets": [
                        {
                            "bucketId": "gemini-weekly",
                            "remainingFraction": 0.8,
                            "window": "weekly",
                            "resetTime": "2026-08-25T09:00:00Z",
                        }
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            windows = fetch_quota_summary(
                access_token="tok-abc",
                project="proj-1",
                base_url=CODE_ASSIST_BASE_URL,
                http_client=client,
            )

        assert captured["method"] == "POST"
        assert captured["url"] == f"{CODE_ASSIST_BASE_URL}:retrieveUserQuotaSummary"
        # Request carries the already-resolved project ONLY — no identity.
        assert captured["body"] == {"project": "proj-1"}
        assert captured["auth"] == "Bearer tok-abc"
        # ...and only normalized quota data comes back.
        assert [(w.label, w.used_percent) for w in windows] == [
            ("Gemini Models (weekly)", pytest.approx(20.0))
        ]

    def test_http_error_fails_closed_without_body_leakage(self, caplog):
        import logging

        import httpx

        from agent.gemini_cloudcode_adapter import (
            CODE_ASSIST_BASE_URL,
            fetch_quota_summary,
        )

        def handler(request):
            return httpx.Response(403, json={"error": "user@example.com is not entitled"})

        with caplog.at_level(logging.DEBUG):
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                windows = fetch_quota_summary(
                    access_token="tok-abc",
                    project="proj-1",
                    base_url=CODE_ASSIST_BASE_URL,
                    http_client=client,
                )

        assert windows is None
        assert "example.com" not in caplog.text
        assert "not entitled" not in caplog.text

    def test_malformed_json_fails_closed(self, caplog):
        import logging

        import httpx

        from agent.gemini_cloudcode_adapter import (
            CODE_ASSIST_BASE_URL,
            fetch_quota_summary,
        )

        def handler(request):
            return httpx.Response(200, content=b"user@example.com <<not json>>")

        with caplog.at_level(logging.DEBUG):
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                windows = fetch_quota_summary(
                    access_token="tok-abc",
                    project="proj-1",
                    base_url=CODE_ASSIST_BASE_URL,
                    http_client=client,
                )

        assert windows is None
        assert "example.com" not in caplog.text

    def test_missing_token_or_project_makes_no_request(self):
        import httpx

        from agent.gemini_cloudcode_adapter import (
            CODE_ASSIST_BASE_URL,
            fetch_quota_summary,
        )

        def handler(request):
            raise AssertionError("must not call the control plane without credentials")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            assert (
                fetch_quota_summary(
                    access_token="",
                    project="proj-1",
                    base_url=CODE_ASSIST_BASE_URL,
                    http_client=client,
                )
                is None
            )
            assert (
                fetch_quota_summary(
                    access_token="tok-abc",
                    project=None,
                    base_url=CODE_ASSIST_BASE_URL,
                    http_client=client,
                )
                is None
            )

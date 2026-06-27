"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import pytest
from unittest.mock import patch


class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False
        assert "Classic Personal Access Tokens" in msg
        assert "ghp_" in msg

    def test_oauth_token_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("gho_abcdefghijklmnop1234")
        assert valid is True

    def test_fine_grained_pat_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("github_pat_abcdefghijklmnop1234")
        assert valid is True

    def test_github_app_token_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghu_abcdefghijklmnop1234")
        assert valid is True

    def test_empty_token_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("")
        assert valid is False



class TestResolveToken:
    """Token resolution with env var priority."""

    def test_copilot_github_token_first_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot_first")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_copilot_first"
        assert source == "COPILOT_GITHUB_TOKEN"

    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"

    def test_github_token_third_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_github_third"
        assert source == "GITHUB_TOKEN"

    def test_classic_pat_in_env_skipped(self, monkeypatch):
        """Classic PATs in env vars should be skipped, not returned."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_classic_pat_nope")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_valid_oauth")
        token, source = resolve_copilot_token()
        # Should skip the ghp_ token and find the gho_ one
        assert token == "gho_valid_oauth"
        assert source == "GITHUB_TOKEN"

    def test_gh_cli_fallback(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_from_cli"):
            token, source = resolve_copilot_token()
        assert token == "gho_from_cli"
        assert source == "gh auth token"

    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"):
            with pytest.raises(ValueError, match="classic PAT"):
                resolve_copilot_token()

    def test_no_token_returns_empty(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value=None):
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""


class TestSourceSuppression:
    """Suppressed Copilot sources must be respected so removals actually stick.

    On Windows, repeated `gh auth token` invocations from dashboard / status
    probes can spawn short-lived `gh.exe` / `tzutil.exe` console windows that
    flash and steal focus.  The fix: gate `resolve_copilot_token()` on the
    persisted suppression marker so an explicit `hermes auth remove copilot`
    suppresses the underlying reader, not just the pool entry.
    """

    def test_suppressed_env_var_is_skipped(self, monkeypatch, tmp_path):
        """A suppressed env:GH_TOKEN must not be returned, even when set."""
        from hermes_cli import copilot_auth

        # Hermetic HERMES_HOME so is_source_suppressed reads from tmp_path
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("copilot", "env:GH_TOKEN")

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot_first")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_should_be_ignored")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")

        token, source = copilot_auth.resolve_copilot_token()
        # Suppressed GH_TOKEN is skipped; COPILOT_GITHUB_TOKEN still wins.
        assert token == "gho_copilot_first"
        assert source == "COPILOT_GITHUB_TOKEN"

    def test_suppressed_copilot_github_token_falls_through_to_unset_gh_token(self, monkeypatch, tmp_path):
        """Suppressing COPILOT_GITHUB_TOKEN must not return it; next env var wins."""
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("copilot", "env:COPILOT_GITHUB_TOKEN")

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_should_be_ignored")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")

        token, source = copilot_auth.resolve_copilot_token()
        assert token == "gho_github_third"
        assert source == "GITHUB_TOKEN"

    def test_suppressed_gh_cli_skips_subprocess_invocation(self, monkeypatch, tmp_path):
        """When gh_cli is suppressed, _try_gh_cli_token must NOT be called.

        This is the load-bearing assertion: if a future change re-introduces
        the unconditional `gh auth token` call, the subprocess will fire
        even though the user removed the source.  Mocking with a
        RuntimeError makes a stray invocation loudly fail the test.
        """
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("copilot", "gh_cli")

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        def _must_not_run(*args, **kwargs):
            raise AssertionError(
                "_try_gh_cli_token must not be invoked when gh_cli is suppressed"
            )

        with patch.object(copilot_auth, "_try_gh_cli_token", side_effect=_must_not_run):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == ""
        assert source == ""

    def test_unsuppressed_gh_cli_still_falls_back(self, monkeypatch, tmp_path):
        """When gh_cli is NOT suppressed, the fallback still works."""
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch.object(copilot_auth, "_try_gh_cli_token", return_value="gho_from_cli"):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == "gho_from_cli"
        assert source == "gh auth token"

    def test_all_sources_suppressed_returns_empty(self, monkeypatch, tmp_path):
        """Suppressing every Copilot source returns empty, no errors."""
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        from hermes_cli.auth import suppress_credential_source
        for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            suppress_credential_source("copilot", f"env:{env_var}")
        suppress_credential_source("copilot", "gh_cli")

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_should_be_ignored")
        monkeypatch.setenv("GH_TOKEN", "gho_should_be_ignored")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_should_be_ignored")

        def _must_not_run(*args, **kwargs):
            raise AssertionError("_try_gh_cli_token must not be invoked when suppressed")

        with patch.object(copilot_auth, "_try_gh_cli_token", side_effect=_must_not_run):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == ""
        assert source == ""


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-edits"
        assert headers["User-Agent"] == "HermesAgent/1.0"
        assert "Editor-Version" in headers

    def test_agent_turn_sets_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers(is_agent_turn=True)
        assert headers["x-initiator"] == "agent"

    def test_user_turn_sets_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers(is_agent_turn=False)
        assert headers["x-initiator"] == "user"

    def test_vision_header(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers(is_vision=True)
        assert headers["Copilot-Vision-Request"] == "true"

    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""

    def test_includes_openai_intent(self):
        from hermes_cli.models import copilot_default_headers
        headers = copilot_default_headers()
        assert "Openai-Intent" in headers
        assert headers["Openai-Intent"] == "conversation-edits"

    def test_includes_x_initiator(self):
        from hermes_cli.models import copilot_default_headers
        headers = copilot_default_headers()
        assert "x-initiator" in headers


class TestApiModeSelection:
    """API mode selection matching opencode's shouldUseCopilotResponsesApi."""

    def test_gpt5_uses_responses(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5.4") is True
        assert _should_use_copilot_responses_api("gpt-5.4-mini") is True
        assert _should_use_copilot_responses_api("gpt-5.3-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2") is True
        assert _should_use_copilot_responses_api("gpt-5.1-codex-max") is True

    def test_gpt5_mini_excluded(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5-mini") is False

    def test_gpt4_uses_chat(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-4.1") is False
        assert _should_use_copilot_responses_api("gpt-4o") is False
        assert _should_use_copilot_responses_api("gpt-4o-mini") is False

    def test_non_gpt_uses_chat(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("claude-sonnet-4.6") is False
        assert _should_use_copilot_responses_api("claude-opus-4.6") is False
        assert _should_use_copilot_responses_api("gemini-2.5-pro") is False
        assert _should_use_copilot_responses_api("grok-code-fast-1") is False


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"

    def test_copilot_env_vars_order_matches_docs(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert copilot.api_key_env_vars == (
            "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"
        )

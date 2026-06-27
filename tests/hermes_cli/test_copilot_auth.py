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


class TestSuppressedSources:
    """Regression tests for issue #53781: resolve_copilot_token() must
    respect explicit source suppression from `hermes auth remove copilot`,
    so dashboard/status-probe paths don't silently re-discover a removed
    credential source (e.g. repeatedly invoking `gh auth token`, which can
    spawn flashing console windows on Windows).

    These use REAL suppress_credential_source() writes against a hermetic
    HERMES_HOME (tmp_path), not mocked is_source_suppressed() returns --
    this exercises the actual persisted-marker round-trip a real
    `hermes auth remove copilot` invocation would produce, not just the
    gating logic in isolation.
    """

    def test_suppressed_gh_cli_skips_subprocess_invocation(self, monkeypatch, tmp_path):
        """When gh_cli is suppressed, _try_gh_cli_token must NOT be called.

        Load-bearing assertion: if a future change re-introduces the
        unconditional `gh auth token` call, the mock raises immediately --
        a stray invocation loudly fails the test instead of silently
        returning a token.
        """
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("copilot", "gh_cli")

        def _fail_if_called():
            raise RuntimeError("_try_gh_cli_token must not be called when gh_cli is suppressed")

        with patch.object(copilot_auth, "_try_gh_cli_token", side_effect=_fail_if_called):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == ""
        assert source == ""

    def test_unsuppressed_gh_cli_source_is_used(self, monkeypatch, tmp_path):
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch.object(copilot_auth, "_try_gh_cli_token", return_value="gho_from_gh_cli"):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == "gho_from_gh_cli"
        assert source == "gh auth token"

    def test_suppressed_env_var_falls_through_to_next(self, monkeypatch, tmp_path):
        """Suppressing COPILOT_GITHUB_TOKEN must skip it; GITHUB_TOKEN wins
        instead of the suppressed value being returned."""
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

    def test_suppressed_env_var_is_skipped_other_unset(self, monkeypatch, tmp_path):
        """A suppressed env:GH_TOKEN must not be returned, even when set,
        while an unsuppressed higher-priority var still wins."""
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("copilot", "env:GH_TOKEN")

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot_first")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_should_be_ignored")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")

        token, source = copilot_auth.resolve_copilot_token()
        assert token == "gho_copilot_first"
        assert source == "COPILOT_GITHUB_TOKEN"

    def test_all_sources_suppressed_returns_empty_without_error(self, monkeypatch, tmp_path):
        """Suppressing every Copilot source must return ("", "") cleanly,
        with no exception even though no token is available."""
        from hermes_cli import copilot_auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        from hermes_cli.auth import suppress_credential_source
        for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            suppress_credential_source("copilot", f"env:{env_var}")
        suppress_credential_source("copilot", "gh_cli")

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_a")
        monkeypatch.setenv("GH_TOKEN", "gho_b")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_c")

        with patch.object(copilot_auth, "_try_gh_cli_token", side_effect=RuntimeError("must not be called")):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == ""
        assert source == ""

    def test_suppression_check_failure_does_not_block_resolution(self, monkeypatch):
        """If is_source_suppressed() itself raises, resolution must still
        proceed (fail open on the suppression check, not on token discovery)."""
        from hermes_cli import copilot_auth

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_env_token")

        with patch(
            "hermes_cli.auth.is_source_suppressed",
            side_effect=RuntimeError("auth store unavailable"),
        ):
            token, source = copilot_auth.resolve_copilot_token()

        assert token == "gho_env_token"
        assert source == "COPILOT_GITHUB_TOKEN"

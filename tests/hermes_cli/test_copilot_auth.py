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



class TestIdentityAudit:
    """Structured Copilot identity resolution audit."""

    def test_identity_precedence_records_skipped_classic_pat(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_classic_pat_nope")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        audit = resolve_copilot_identity_audit()

        assert audit.token == "gho_gh_second"
        assert audit.source == "GH_TOKEN"
        assert audit.source_kind == "env"
        assert len(audit.skipped_sources) == 1
        assert audit.skipped_sources[0].source == "COPILOT_GITHUB_TOKEN"
        assert "Classic Personal Access Tokens" in audit.skipped_sources[0].reason

    def test_pool_audit_records_skipped_invalid_entries_and_gh_fallback(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                "not-a-dict",
                {"label": "no-token-here"},
                {"access_token": ""},
                {"access_token": "ghp_classic_pat"},
            ],
        ), patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ):
            audit = resolve_copilot_identity_audit(include_credential_pool=True)

        assert audit.token == "gho_from_cli"
        assert audit.source == "gh auth token"
        assert audit.source_kind == "gh_auth"
        assert [skip.source for skip in audit.skipped_sources] == [
            "credential_pool:copilot[0]",
            "credential_pool:copilot[1]",
            "credential_pool:copilot[2]",
            "credential_pool:copilot[3]",
        ]
        assert any(
            "Non-dict credential pool entry" in skip.reason
            for skip in audit.skipped_sources
        )
        assert any("Missing access_token" in skip.reason for skip in audit.skipped_sources)
        assert any("Classic Personal Access Tokens" in skip.reason for skip in audit.skipped_sources)

    def test_pool_token_wins_before_gh_auth(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_pool_token"}],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            return_value=("tid_from_pool", 1234567890.0),
        ), patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ):
            audit = resolve_copilot_identity_audit(
                include_credential_pool=True,
                exchange_pool_tokens=True,
            )

        assert audit.token == "tid_from_pool"
        assert audit.source == "credential_pool:copilot[0]"
        assert audit.source_kind == "credential_pool"


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

    def test_default_headers_include_openai_intent(self, monkeypatch):
        from hermes_cli.copilot_auth import copilot_request_headers
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._latest_copilot_cli_version",
            lambda: "1.0.63",
        )
        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-panel"
        # Presents as the @github/copilot CLI: UA is copilot/<ver> (short form
        # or full "copilot/<ver> (<platform> <node>) term/<term>" when node is
        # resolvable). The Editor-* VS Code Chat headers are NOT sent; the CLI
        # sends Runtime-Client-Version instead.
        assert headers["User-Agent"].startswith("copilot/1.0.63")
        assert "Editor-Version" not in headers
        assert "Editor-Plugin-Version" not in headers
        assert headers["Runtime-Client-Version"] == "1.0.63"

    def test_user_agent_full_cli_form_when_node_present(self, monkeypatch):
        """When a Node runtime + TERM_PROGRAM are resolvable, the UA matches the
        real CLI ``FG()`` builder: copilot/<ver> (<platform> <node>) term/<term>.
        """
        from hermes_cli import copilot_auth
        monkeypatch.setattr(copilot_auth, "_latest_copilot_cli_version", lambda: "1.0.63")
        monkeypatch.setattr(copilot_auth, "_copilot_node_version", lambda: "v22.22.3")
        monkeypatch.setattr(copilot_auth.sys, "platform", "linux")
        monkeypatch.setenv("HERMES_COPILOT_TERM_PROGRAM", "vscode")
        ua = copilot_auth._copilot_user_agent()
        assert ua == "copilot/1.0.63 (linux v22.22.3) term/vscode"

    def test_user_agent_short_form_when_no_node(self, monkeypatch):
        """No resolvable Node runtime → honest short core, no fabricated runtime."""
        from hermes_cli import copilot_auth
        monkeypatch.setattr(copilot_auth, "_latest_copilot_cli_version", lambda: "1.0.63")
        monkeypatch.setattr(copilot_auth, "_copilot_node_version", lambda: "")
        ua = copilot_auth._copilot_user_agent()
        assert ua == "copilot/1.0.63"

    def test_term_program_defaults_to_vscode_not_unknown(self, monkeypatch):
        """Unset TERM_PROGRAM resolves to a valid default (vscode), never the
        bot-signalling literal ``unknown`` the raw CLI builder would emit."""
        from hermes_cli import copilot_auth
        monkeypatch.delenv("HERMES_COPILOT_TERM_PROGRAM", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert copilot_auth._copilot_term_program() == "vscode"

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

    def test_includes_openai_intent(self, monkeypatch):
        from hermes_cli.models import copilot_default_headers
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._latest_copilot_cli_version",
            lambda: "1.0.63",
        )
        headers = copilot_default_headers()
        assert "Openai-Intent" in headers
        assert headers["Openai-Intent"] == "conversation-panel"
        assert headers["User-Agent"].startswith("copilot/1.0.63")

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

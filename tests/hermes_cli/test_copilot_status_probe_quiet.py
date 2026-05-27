"""Regression coverage for quiet Copilot status probes.

Generic GitHub tokens (GH_TOKEN / GITHUB_TOKEN) are often classic PATs that are
valid for normal GitHub workflows but unusable for Copilot. Provider discovery
must not warn about them unless Copilot is explicitly selected.
"""
from __future__ import annotations

import logging
from unittest.mock import patch


def test_resolve_copilot_token_quiet_probe_skips_classic_pat_without_warning(monkeypatch, caplog):
    from hermes_cli.copilot_auth import resolve_copilot_token

    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_classic_pat_for_regular_github")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_another_regular_github_pat")

    with patch("hermes_cli.copilot_auth._try_gh_cli_token") as gh_cli, caplog.at_level(logging.WARNING):
        token, source = resolve_copilot_token(warn_invalid=False, include_gh_cli=False)

    assert token == ""
    assert source == ""
    gh_cli.assert_not_called()
    assert "Token from GH_TOKEN is not supported" not in caplog.text
    assert "Token from GITHUB_TOKEN is not supported" not in caplog.text


def test_copilot_provider_status_probe_does_not_warn_on_ambient_classic_pats(monkeypatch, caplog):
    from hermes_cli.auth import get_api_key_provider_status

    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_classic_pat_for_regular_github")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_another_regular_github_pat")

    with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_cli_classic_pat") as gh_cli, caplog.at_level(logging.WARNING):
        status = get_api_key_provider_status("copilot")

    assert status["configured"] is False
    assert status["logged_in"] is False
    gh_cli.assert_called_once()
    assert "Copilot token validation failed" not in caplog.text
    assert "Classic Personal Access Tokens" not in caplog.text

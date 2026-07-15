"""Tests for hermes_cli.git_credentials_boot — boot-time provisioning of
per-profile git credentials from the profile's own configured GitHub token.

The module runs in-process at container boot, where the profile's .env is
readable on the mounted volume and HERMES_HOME / get_subprocess_home() are
known directly — unlike the tool subprocess, which cannot see the token
(GITHUB_TOKEN is blocklisted from the subprocess env by design).

Tests run against a fake $HERMES_HOME under tmp_path. No container, no git.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from hermes_cli.git_credentials_boot import (
    build_git_config_content,
    build_git_credentials_content,
    provision_all,
    provision_git_credentials,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_env(home: Path, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(body)


def _cred_path(home: Path) -> Path:
    return home / "home" / ".git-credentials"


def _cfg_path(home: Path) -> Path:
    return home / "home" / ".gitconfig"


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


# ---------------------------------------------------------------------------
# Pure content builders
# ---------------------------------------------------------------------------


def test_credentials_content_uses_x_access_token_https_line():
    assert build_git_credentials_content("ghp_abc") == (
        "https://x-access-token:ghp_abc@github.com\n"
    )


def test_config_content_has_store_helper_identity_and_insteadof():
    cfg = build_git_config_content(name="agent-a", email="agent-a@users.noreply.github.com")
    assert "helper = store" in cfg
    assert "name = agent-a" in cfg
    assert "email = agent-a@users.noreply.github.com" in cfg
    # ssh-style remotes get rewritten through HTTPS so the agent doesn't die on
    # "Host key verification failed" when it reaches for an SSH URL.
    assert 'insteadOf = git@github.com:' in cfg
    assert 'insteadOf = ssh://git@github.com/' in cfg


# ---------------------------------------------------------------------------
# provision_git_credentials — single home
# ---------------------------------------------------------------------------


def test_writes_credentials_and_config_from_env_token(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is True
    assert result.source == "GITHUB_PAT"
    assert _cred_path(tmp_path).read_text() == "https://x-access-token:ghp_secret@github.com\n"
    assert "helper = store" in _cfg_path(tmp_path).read_text()


def test_credentials_file_is_chmod_600(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    provision_git_credentials(tmp_path)
    assert _mode(_cred_path(tmp_path)) == 0o600


def test_config_file_is_chmod_644(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_secret\n")
    provision_git_credentials(tmp_path)
    assert _mode(_cfg_path(tmp_path)) == 0o644


def test_token_precedence_pat_beats_github_token_beats_gh_token(tmp_path):
    _write_env(tmp_path, "GH_TOKEN=gh_v\nGITHUB_TOKEN=ght_v\nGITHUB_PAT=pat_v\n")
    result = provision_git_credentials(tmp_path)
    assert result.source == "GITHUB_PAT"
    assert "pat_v" in _cred_path(tmp_path).read_text()


def test_falls_back_to_gh_token_when_only_one_present(tmp_path):
    _write_env(tmp_path, "GH_TOKEN=gh_only\n")
    result = provision_git_credentials(tmp_path)
    assert result.source == "GH_TOKEN"
    assert "gh_only" in _cred_path(tmp_path).read_text()


def test_strips_quotes_and_whitespace_from_token(tmp_path):
    _write_env(tmp_path, 'GITHUB_PAT="ghp_quoted"  \n')
    provision_git_credentials(tmp_path)
    assert _cred_path(tmp_path).read_text() == "https://x-access-token:ghp_quoted@github.com\n"


def test_no_token_is_inert(tmp_path):
    _write_env(tmp_path, "OPENAI_API_KEY=sk-whatever\n")
    result = provision_git_credentials(tmp_path)
    assert result.provisioned is False
    assert result.reason == "no GitHub token configured"
    assert not _cred_path(tmp_path).exists()
    assert not (tmp_path / "home").exists()


def test_missing_env_is_inert(tmp_path):
    result = provision_git_credentials(tmp_path)
    assert result.provisioned is False
    assert not _cred_path(tmp_path).exists()


def test_empty_token_value_is_inert(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=\n")
    result = provision_git_credentials(tmp_path)
    assert result.provisioned is False


# ---------------------------------------------------------------------------
# Apply-if-absent data-loss guard
# ---------------------------------------------------------------------------


def test_apply_if_absent_does_not_clobber_existing_credentials(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".git-credentials").write_text("PRE-EXISTING\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is False
    assert "already present" in (result.reason or "")
    assert (home / ".git-credentials").read_text() == "PRE-EXISTING\n"


def test_apply_if_absent_does_not_clobber_existing_gitconfig(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[user]\n\tname = human\n")

    result = provision_git_credentials(tmp_path)

    assert result.provisioned is False
    assert (home / ".gitconfig").read_text() == "[user]\n\tname = human\n"


def test_force_overwrites_existing(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=ghp_new\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".git-credentials").write_text("PRE-EXISTING\n")

    result = provision_git_credentials(tmp_path, force=True)

    assert result.provisioned is True
    assert "ghp_new" in (home / ".git-credentials").read_text()


# ---------------------------------------------------------------------------
# provision_all — HERMES_HOME root (default profile) + named profiles
# ---------------------------------------------------------------------------


def test_provision_all_does_root_and_each_named_profile(tmp_path):
    # Default profile = HERMES_HOME root.
    _write_env(tmp_path, "GITHUB_PAT=root_tok\n")
    # Named profiles live under profiles/<name>/, each with its own .env.
    _write_env(tmp_path / "profiles" / "alpha", "GITHUB_PAT=alpha_tok\n")
    _write_env(tmp_path / "profiles" / "beta", "GH_TOKEN=beta_tok\n")

    results = provision_all(tmp_path)

    provisioned = {r.home: r for r in results if r.provisioned}
    assert tmp_path in provisioned
    assert (tmp_path / "profiles" / "alpha") in provisioned
    assert (tmp_path / "profiles" / "beta") in provisioned
    assert "root_tok" in _cred_path(tmp_path).read_text()
    assert "alpha_tok" in _cred_path(tmp_path / "profiles" / "alpha").read_text()
    assert "beta_tok" in _cred_path(tmp_path / "profiles" / "beta").read_text()


def test_provision_all_skips_profile_without_token(tmp_path):
    _write_env(tmp_path / "profiles" / "notoken", "OPENAI_API_KEY=sk-x\n")
    results = provision_all(tmp_path)
    notoken = tmp_path / "profiles" / "notoken"
    assert all(not r.provisioned for r in results if r.home == notoken)
    assert not _cred_path(notoken).exists()


def test_provision_all_handles_no_profiles_dir(tmp_path):
    _write_env(tmp_path, "GITHUB_PAT=root_tok\n")
    results = provision_all(tmp_path)
    assert any(r.provisioned and r.home == tmp_path for r in results)


def test_provision_all_uses_hermes_agent_name_for_root_identity(tmp_path, monkeypatch):
    # The default profile (HERMES_HOME root) basename is "data" in production
    # (/opt/data); HERMES_AGENT_NAME carries the real agent name, so commits
    # are authored as the agent, not "data".
    monkeypatch.setenv("HERMES_AGENT_NAME", "gamma")
    _write_env(tmp_path, "GITHUB_PAT=root_tok\n")

    provision_all(tmp_path)

    cfg = _cfg_path(tmp_path).read_text()
    assert "name = gamma" in cfg
    assert "email = gamma@users.noreply.github.com" in cfg


def test_provision_all_profile_identity_uses_profile_name_not_env(tmp_path, monkeypatch):
    # A named profile is its own agent — its identity is the profile name,
    # never the container-level HERMES_AGENT_NAME.
    monkeypatch.setenv("HERMES_AGENT_NAME", "rootname")
    _write_env(tmp_path / "profiles" / "beta", "GITHUB_PAT=tok\n")

    provision_all(tmp_path)

    cfg = _cfg_path(tmp_path / "profiles" / "beta").read_text()
    assert "name = beta" in cfg
    assert "name = rootname" not in cfg

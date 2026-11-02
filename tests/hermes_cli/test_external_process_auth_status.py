"""Tests for external-process provider auth status and Accounts-tab wiring.

Covers the external-process (ACP) fix class, driven through a fake profile:
  * ``get_auth_status()`` dispatches on ``auth_type == "external_process"``
    (not a hardcoded slug), so future ACP-style providers inherit the
    behaviour automatically.
  * ``auth_verified``/``auth_source`` carry positive credential evidence
    (the profile's ``external_auth_evidence()`` answer) while remaining
    honest — no evidence means unknown, never "signed out".
  * The Accounts-tab sign-in ``cli_command`` reflects the executable the
    user actually configured (the profile's ``process_command_env_vars``),
    and its default is the profile's own CLI invocation.
"""

import os

import pytest

from hermes_cli.auth import (
    get_auth_status,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
)
from providers.base import ProviderProfile


class _AcmeAcpProfile(ProviderProfile):
    """An ACP CLI whose login evidence is whatever the test sets on ``evidence``."""

    evidence = None

    def external_auth_evidence(self):
        return self.evidence


@pytest.fixture()
def acme_acp(tmp_path, monkeypatch):
    """Register a fake external-process profile and neutralize host state so tests pin behaviour."""
    import providers
    from pathlib import Path

    from hermes_cli import auth

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for var in ("HERMES_ACME_ACP_COMMAND", "HERMES_ACME_ACP_ARGS", "ACME_ACP_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    profile = _AcmeAcpProfile(name="acme-acp", auth_type="external_process", base_url="acp://acme",
                              process_command="acme", process_args=("--acp", "--stdio"),
                              process_command_env_vars=("HERMES_ACME_ACP_COMMAND",),
                              process_args_env_var="HERMES_ACME_ACP_ARGS",
                              base_url_env_var="ACME_ACP_BASE_URL")
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    auth.PROVIDER_REGISTRY[profile.name] = auth.ProviderConfig(
        profile.name, profile.name, profile.auth_type, inference_base_url=profile.base_url,
        base_url_env_var=profile.base_url_env_var)
    return profile


# --- get_auth_status dispatches on auth_type, not slug ----------------------


def test_get_auth_status_dispatches_external_process_by_auth_type(
    tmp_path, monkeypatch, acme_acp
):
    fake = tmp_path / ("acme.exe" if os.name == "nt" else "acme")
    fake.write_text("", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_ACME_ACP_COMMAND", str(fake))
    # Point HOME somewhere empty so on-disk credential stores don't leak in.
    monkeypatch.setenv("HOME", str(tmp_path))

    status = get_auth_status("acme-acp")

    # The external_process status shape, not the {"logged_in": False}
    # fallthrough — proves the dispatcher reached the right branch.
    assert status.get("provider") == "acme-acp"
    assert status.get("configured") is True
    assert status.get("resolved_command") == str(fake)
    assert "auth_verified" in status


def test_external_process_status_finds_user_local_cli_outside_backend_path(
    tmp_path, monkeypatch, acme_acp
):
    """Desktop SSH inventory must discover a CLI installed in the user's local bin."""
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    cli = local_bin / "acme"
    cli.write_text("", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "backend-bin"))

    status = get_external_process_provider_status("acme-acp")

    assert status["configured"] is True
    assert status["resolved_command"] == str(cli)
    assert resolve_external_process_provider_credentials("acme-acp")["command"] == str(cli)


@pytest.mark.platforms("windows")
def test_external_process_status_finds_user_profile_local_cli_as_native_path(
    tmp_path, monkeypatch, acme_acp
):
    """The Windows user-local hit is one native path, not a mixed-separator join."""
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    cli = local_bin / "acme.exe"
    cli.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path / "backend-bin"))

    status = get_external_process_provider_status("acme-acp")
    resolved = status["resolved_command"]

    assert status["configured"] is True
    assert "/" not in resolved and os.path.samefile(resolved, cli)
    assert resolve_external_process_provider_credentials("acme-acp")["command"] == resolved


def test_external_process_status_resolves_relative_configured_path_from_cwd(
    tmp_path, monkeypatch, acme_acp
):
    """An operator's relative command path keeps resolving against the launch cwd."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli = bin_dir / ("acme.exe" if os.name == "nt" else "acme")
    cli.write_text("", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "backend-bin"))
    monkeypatch.setenv("HERMES_ACME_ACP_COMMAND", os.path.join(".", "bin", cli.name))

    status = get_external_process_provider_status("acme-acp")

    assert status["configured"] is True
    assert status["resolved_command"] == str(cli)
    assert resolve_external_process_provider_credentials("acme-acp")["command"] == str(cli)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("preferred", ["claude-prefix", "user-local", "path", "explicit"])
def test_external_process_claude_resolution_preserves_precedence(
    tmp_path, monkeypatch, acme_acp, preferred
):
    """Generic discovery must retain Claude's extra prefixes and command precedence."""
    candidates = {}
    for name, directory in (
        ("claude-prefix", tmp_path / ".claude" / "local"),
        ("user-local", tmp_path / ".local" / "bin"),
        ("path", tmp_path / "backend-bin"),
        ("explicit", tmp_path / "configured-bin"),
    ):
        directory.mkdir(parents=True)
        executable = directory / "claude"
        executable.write_text("", encoding="utf-8")
        executable.chmod(0o755)
        candidates[name] = executable
        if name == preferred:
            break
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "backend-bin"))
    monkeypatch.setenv(
        "HERMES_ACME_ACP_COMMAND",
        str(candidates[preferred]) if preferred == "explicit" else "claude",
    )

    status = get_external_process_provider_status("acme-acp")

    assert status["configured"] is True
    assert status["resolved_command"] == str(candidates[preferred])
    assert resolve_external_process_provider_credentials("acme-acp")["command"] == str(candidates[preferred])


def test_external_process_status_rejects_wrong_auth_type():
    # A provider that exists but is not external_process must be refused —
    # the generic dispatcher relies on this guard.
    assert get_external_process_provider_status("openrouter") == {"configured": False}
    assert get_external_process_provider_status("no-such-provider") == {"configured": False}


# --- auth_verified: positive evidence only ----------------------------------


def test_auth_verified_false_without_evidence(tmp_path, monkeypatch, acme_acp):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "backend-bin"))  # no binary, and the profile offers no evidence
    status = get_external_process_provider_status("acme-acp")
    assert status["auth_verified"] is False
    assert status["auth_source"] is None


# --- desktop picker explicit-only filter ------------------------------------


def test_explicit_filter_keeps_signed_in_external_process_row(tmp_path, monkeypatch, acme_acp):
    # A verified CLI login leaves no trace in active_provider/config/env —
    # the explicit-only desktop filter must treat it like the Anthropic OAuth
    # carve-out and keep the row.
    from hermes_cli.inventory import _filter_explicit_provider_rows

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(acme_acp, "evidence", (True, "~/.acme/credentials.json"), raising=False)

    class _Ctx:
        current_provider = "nous"

    rows = [{"slug": "acme-acp", "models": ["gpt-5.4"]}]
    kept = _filter_explicit_provider_rows(rows, _Ctx())

    assert any(r["slug"] == "acme-acp" for r in kept), \
        "signed-in acme-acp must survive the explicit-only picker filter"


def test_explicit_filter_drops_unverified_external_process_row(tmp_path, monkeypatch, acme_acp):
    # Merely having the executable on PATH is ambient discovery, not an
    # explicit configuration — the desktop filter keeps its narrower contract
    # for a profile whose CLI exposes readable token stores (none here).
    from hermes_cli.inventory import _filter_explicit_provider_rows

    monkeypatch.setenv("HOME", str(tmp_path))  # no credential stores
    monkeypatch.setattr(acme_acp, "evidence", (False, None), raising=False)

    class _Ctx:
        current_provider = "nous"

    rows = [{"slug": "acme-acp", "models": ["gpt-5.4"]}]
    kept = _filter_explicit_provider_rows(rows, _Ctx())

    assert all(r["slug"] != "acme-acp" for r in kept)


# --- Accounts-tab cli_command ------------------------------------------------


def test_cli_command_reflects_configured_executable(tmp_path, monkeypatch, acme_acp):
    from hermes_cli.web_server_oauth import _external_process_cli_command

    fake = tmp_path / ("acme.exe" if os.name == "nt" else "acme")
    fake.write_text("", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_ACME_ACP_COMMAND", str(fake))

    rendered = _external_process_cli_command("acme-acp", "acme login")

    assert rendered == f"{fake} login"


def test_cli_command_untouched_for_non_external_providers(acme_acp):
    from hermes_cli.web_server_oauth import _external_process_cli_command

    assert _external_process_cli_command("nous", "hermes auth add nous") == "hermes auth add nous"


def test_cli_command_default_when_no_override(monkeypatch, acme_acp):
    from hermes_cli.web_server_oauth import _external_process_cli_command

    assert _external_process_cli_command("acme-acp", "acme login") == "acme login"

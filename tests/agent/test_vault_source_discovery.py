"""Invariants for login-source DISCOVERY: which host owns a manager, what state it is in, and
that discovery never touches a secret value.

Three contracts:
1. Capability is a property of the session, not of the gateway process's PATH. A manager CLI
   installed in a prefix the launcher's PATH omits is still detected, and when the session's
   terminal backend is a remote host, that host is what the status names — not the gateway box.
2. The four states are distinct: not_installed / disconnected / auth_required / available.
   Every non-available state carries a reason naming the affected host.
3. Discovery is metadata only: no vault item, token or credential value is read, and nothing
   resolvable appears in a probe result.

Regression for the "1Password shows 'Not detected' with no explanation" class.
"""

from __future__ import annotations

import stat

import pytest

from agent.secret_sources import base as sources_base
from agent.vault_backends import base as vb


@pytest.fixture
def vault_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _fake_binary(directory, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


@pytest.fixture
def no_ambient_op(monkeypatch):
    """A PATH with no manager CLI on it — the service-process condition this bug class lives in."""
    monkeypatch.setenv("PATH", "/nonexistent-for-tests")
    monkeypatch.setattr(sources_base, "_COMMON_CLI_BIN_DIRS", ())


# ── capability is a property of the host, not of the launcher's PATH ──────────────────────────


def test_probe_detects_a_binary_outside_the_service_path(vault_home, monkeypatch, no_ambient_op):
    """A gateway started by launchd/systemd inherits a minimal PATH. `op` installed in a common
    prefix must still be detected, or the Desktop reports "not installed" for a host that has it."""
    prefix = vault_home.parent / "opt" / "homebrew" / "bin"
    _fake_binary(prefix, "op")
    monkeypatch.setattr(sources_base, "_COMMON_CLI_BIN_DIRS", (str(prefix),))

    assert vb.find_manager_binary("onepassword") is not None
    assert vb.probe("onepassword").installed is True


def test_probe_without_the_binary_anywhere_is_not_installed(vault_home, no_ambient_op):
    result = vb.probe("onepassword")
    assert result.installed is False
    assert result.status is vb.SourceStatus.not_installed
    assert result.host and result.host in result.reason


def test_probe_names_the_remote_host_that_owns_the_manager(vault_home, monkeypatch):
    """With an SSH terminal backend the manager CLI runs on the REMOTE machine; naming the gateway
    host would send the user to the wrong box to authenticate (root rule: capability is a property
    of the SESSION, resolved from the session's own policy, never the process env)."""
    from tools import terminal_scope

    token = terminal_scope.set_terminal_scope({"TERMINAL_ENV": "ssh", "TERMINAL_SSH_HOST": "greenmini-jonathon"})
    try:
        assert vb.owning_host("onepassword") == "greenmini-jonathon"
        assert vb.probe("onepassword").host == "greenmini-jonathon"
    finally:
        terminal_scope.reset_terminal_scope(token)


def test_local_backend_names_this_host_not_a_remote_one(vault_home, monkeypatch):
    from tools import terminal_scope

    token = terminal_scope.set_terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_SSH_HOST": "unused-host"})
    try:
        assert vb.owning_host("onepassword") != "unused-host"
    finally:
        terminal_scope.reset_terminal_scope(token)


# ── the four states stay distinct ─────────────────────────────────────────────────────────────


def test_probe_status_available_when_installed_and_unlocked(vault_home, monkeypatch):
    exe = _fake_binary(vault_home.parent / "bin", "op")
    monkeypatch.setattr(vb, "_cfg", lambda: {"onepassword": {"binary_path": str(exe)}})
    monkeypatch.setattr(
        "agent.vault_backends.onepassword.OnePasswordLoginBackend.is_unlocked", lambda self: True)

    result = vb.probe("onepassword")
    assert result.status is vb.SourceStatus.available
    assert result.installed is True
    assert result.reason == ""


def test_probe_status_auth_required_points_at_the_owning_host(vault_home, monkeypatch):
    """Installed but locked is NOT "not detected": the user must be told to authenticate, and
    where. This is the state the old six-boolean shape collapsed away."""
    exe = _fake_binary(vault_home.parent / "bin", "op")
    monkeypatch.setattr(vb, "_cfg", lambda: {"onepassword": {"binary_path": str(exe)}})
    monkeypatch.setattr(
        "agent.vault_backends.onepassword.OnePasswordLoginBackend.is_unlocked", lambda self: False)

    result = vb.probe("onepassword")
    assert result.status is vb.SourceStatus.auth_required
    assert result.installed is True
    assert result.host in result.reason


def test_probe_status_disconnected_when_the_backend_cannot_be_interrogated(vault_home, monkeypatch):
    exe = _fake_binary(vault_home.parent / "bin", "op")
    monkeypatch.setattr(vb, "_cfg", lambda: {"onepassword": {"binary_path": str(exe)}})

    def boom(self):
        raise RuntimeError("ssh: connect to host greenmini-jonathon port 22: Operation timed out")

    monkeypatch.setattr("agent.vault_backends.onepassword.OnePasswordLoginBackend.is_unlocked", boom)

    result = vb.probe("onepassword")
    assert result.status is vb.SourceStatus.disconnected
    assert result.installed is True, "an unreachable host is not the same as an uninstalled CLI"
    assert result.host in result.reason


def test_pinned_but_missing_binary_is_not_installed_and_says_so(vault_home, monkeypatch, no_ambient_op):
    """A pinned binary_path is never silently replaced by a PATH lookup — the user pinned it for
    a reason, and a wrong pin must be visible rather than masked."""
    monkeypatch.setattr(vb, "_cfg", lambda: {"onepassword": {"binary_path": "/definitely/not/here/op"}})

    result = vb.probe("onepassword")
    assert result.installed is False
    assert result.status is vb.SourceStatus.not_installed
    assert "binary_path" in result.reason


def test_is_installed_delegates_to_probe(vault_home, monkeypatch):
    """One discovery answer for every caller: the legacy boolean is a view on the probe."""
    calls = []
    real = vb.probe

    def spy(name):
        calls.append(name)
        return real(name)

    monkeypatch.setattr(vb, "probe", spy)
    vb.is_installed("onepassword")
    assert calls == ["onepassword"]


# ── discovery is metadata only ────────────────────────────────────────────────────────────────


def test_probe_never_reads_or_carries_a_secret_value(vault_home, monkeypatch):
    """Discovery must not resolve anything. The probe runs no manager command and its result
    contains no field that could carry a value."""
    exe = _fake_binary(vault_home.parent / "bin", "op")
    monkeypatch.setattr(vb, "_cfg", lambda: {"onepassword": {"binary_path": str(exe)}})
    monkeypatch.setattr(
        "agent.vault_backends.onepassword.OnePasswordLoginBackend.is_unlocked", lambda self: True)

    def forbidden(*_a, **_kw):
        raise AssertionError("discovery must not resolve, list or read anything from the manager")

    monkeypatch.setattr("agent.vault_backends.onepassword.OnePasswordLoginBackend._run", forbidden)
    monkeypatch.setattr(
        "agent.vault_backends.onepassword.OnePasswordLoginBackend.resolve_password", forbidden)

    result = vb.probe("onepassword")
    assert set(result.to_dict()) == {"installed", "status", "host", "reason"}

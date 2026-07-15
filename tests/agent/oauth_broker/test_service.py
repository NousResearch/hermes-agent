"""launchd rendering and lifecycle helpers. Rendering only — no test ever
invokes a real launchctl; runners are always fakes."""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

import pytest

import agent.oauth_broker.service as service_mod
from agent.oauth_broker.service import (
    BROKER_LAUNCHD_LABEL,
    broker_launchd_plist_path,
    install_broker_service,
    launchctl_bootout_argv,
    launchctl_bootstrap_argv,
    launchctl_kickstart_argv,
    launchctl_print_argv,
    launchd_domain,
    render_broker_launchd_plist,
    uninstall_broker_service,
    write_broker_launchd_plist,
)


def _render(tmp_path, **kwargs):
    kwargs.setdefault("python_executable", "/synthetic/venv/bin/python")
    kwargs.setdefault("hermes_home", tmp_path / "hermes-home")
    return render_broker_launchd_plist(**kwargs)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_plist_renders_label_keepalive_and_loopback_arguments(tmp_path):
    payload = plistlib.loads(_render(tmp_path))
    assert payload["Label"] == "ai.hermes.oauth-broker"
    assert BROKER_LAUNCHD_LABEL == "ai.hermes.oauth-broker"
    assert payload["RunAtLoad"] is True
    # Restart on crash, stay down on a clean exit (launchctl bootout).
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ProgramArguments"] == [
        "/synthetic/venv/bin/python",
        "-m",
        "hermes_cli.main",
        "oauth-broker",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        "17880",
    ]


def test_plist_logs_live_under_hermes_home_and_env_is_minimal(tmp_path):
    home = tmp_path / "hermes-home"
    payload = plistlib.loads(_render(tmp_path, hermes_home=home))
    assert payload["StandardOutPath"] == str(home / "logs" / "oauth-broker.log")
    assert payload["StandardErrorPath"] == str(
        home / "logs" / "oauth-broker.error.log"
    )
    # Exactly one environment variable — no keys, tokens, or secret paths.
    assert payload["EnvironmentVariables"] == {"HERMES_HOME": str(home)}


def test_plist_rejects_non_loopback_host(tmp_path):
    with pytest.raises(ValueError):
        _render(tmp_path, host="0.0.0.0")


def test_plist_throttles_crash_loop_restarts(tmp_path):
    payload = plistlib.loads(_render(tmp_path))
    assert payload["ThrottleInterval"] == 5


def test_render_creates_log_directory_owner_only(tmp_path):
    home = tmp_path / "hermes-home"
    _render(tmp_path, hermes_home=home)
    log_dir = home / "logs"
    assert log_dir.is_dir()
    assert (log_dir.stat().st_mode & 0o777) == 0o700


def test_plist_port_is_configurable(tmp_path):
    payload = plistlib.loads(_render(tmp_path, port=17999))
    assert payload["ProgramArguments"][-1] == "17999"


def test_default_plist_path_is_user_launch_agents():
    path = broker_launchd_plist_path()
    assert path.name == "ai.hermes.oauth-broker.plist"
    assert path.parent.name == "LaunchAgents"
    assert path.parent.parent.name == "Library"


# ---------------------------------------------------------------------------
# Atomic write with owner-only permissions
# ---------------------------------------------------------------------------


def test_write_is_atomic_with_0600_permissions(tmp_path):
    target = tmp_path / "LaunchAgents" / "ai.hermes.oauth-broker.plist"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-content")
    content = _render(tmp_path)
    written = write_broker_launchd_plist(plist_path=target, content=content)
    assert written == target
    assert target.read_bytes() == content
    assert (target.stat().st_mode & 0o777) == 0o600
    leftovers = [p for p in target.parent.iterdir() if p != target]
    assert leftovers == []  # temp staging file was replaced, not abandoned


def test_write_handles_short_writes_and_fsyncs_file_and_directory(
    tmp_path, monkeypatch
):
    target = tmp_path / "LaunchAgents" / "ai.hermes.oauth-broker.plist"
    target.parent.mkdir(parents=True)
    content = _render(tmp_path)
    real_write = service_mod.os.write
    fsync_calls = []

    def short_write(fd, data):
        return real_write(fd, bytes(data[:7]))

    monkeypatch.setattr(service_mod.os, "write", short_write)
    monkeypatch.setattr(service_mod.os, "fsync", lambda fd: fsync_calls.append(fd))

    write_broker_launchd_plist(plist_path=target, content=content)
    assert target.read_bytes() == content
    assert len(fsync_calls) >= 2


def test_write_does_not_follow_planted_fixed_temp_symlink(tmp_path):
    parent = tmp_path / "LaunchAgents"
    parent.mkdir(parents=True)
    target = parent / "ai.hermes.oauth-broker.plist"
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim-must-not-change")
    planted = target.with_name(target.name + ".tmp")
    planted.symlink_to(victim)
    content = _render(tmp_path)

    write_broker_launchd_plist(plist_path=target, content=content)

    assert victim.read_bytes() == b"victim-must-not-change"
    assert target.read_bytes() == content
    assert not target.is_symlink()


# ---------------------------------------------------------------------------
# launchctl argv builders — gui/$uid domain, explicit lists
# ---------------------------------------------------------------------------


def test_launchd_domain_targets_gui_uid():
    assert launchd_domain(uid=501) == "gui/501"
    assert launchd_domain() == f"gui/{os.getuid()}"


def test_launchctl_argv_lists_are_exact(tmp_path):
    plist = tmp_path / "ai.hermes.oauth-broker.plist"
    assert launchctl_bootstrap_argv(plist, uid=501) == [
        "launchctl",
        "bootstrap",
        "gui/501",
        str(plist),
    ]
    assert launchctl_bootout_argv(uid=501) == [
        "launchctl",
        "bootout",
        "gui/501/ai.hermes.oauth-broker",
    ]
    assert launchctl_kickstart_argv(uid=501) == [
        "launchctl",
        "kickstart",
        "-k",
        "gui/501/ai.hermes.oauth-broker",
    ]
    assert launchctl_print_argv(uid=501) == [
        "launchctl",
        "print",
        "gui/501/ai.hermes.oauth-broker",
    ]


# ---------------------------------------------------------------------------
# Install / uninstall lifecycle — render-only by default
# ---------------------------------------------------------------------------


def test_install_defaults_to_render_only(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    result = install_broker_service(
        plist_path=target, content=_render(tmp_path), uid=501
    )
    assert result["executed"] is False
    assert target.exists()
    assert result["bootstrap"][0:2] == ["launchctl", "bootstrap"]
    assert result["kickstart"][0:2] == ["launchctl", "kickstart"]


def test_install_apply_requires_explicit_runner(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    with pytest.raises(ValueError):
        install_broker_service(
            plist_path=target, content=_render(tmp_path), apply=True
        )
    assert not target.exists()  # aborted before any filesystem change


def test_install_apply_runs_bootstrap_then_kickstart_via_runner(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)

    result = install_broker_service(
        plist_path=target,
        content=_render(tmp_path),
        apply=True,
        runner=runner,
        uid=501,
    )
    assert result["executed"] is True
    assert calls == [
        ["launchctl", "bootstrap", "gui/501", str(target)],
        ["launchctl", "kickstart", "-k", "gui/501/ai.hermes.oauth-broker"],
    ]


def test_install_apply_kickstart_failure_boots_out_and_restores_plist(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    original = b"synthetic-original-plist"
    target.write_bytes(original)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "kickstart":
            raise RuntimeError("synthetic kickstart failure")

    with pytest.raises(RuntimeError, match="synthetic kickstart failure"):
        install_broker_service(
            plist_path=target,
            content=_render(tmp_path),
            apply=True,
            runner=runner,
            uid=501,
        )

    assert calls == [
        ["launchctl", "bootstrap", "gui/501", str(target)],
        ["launchctl", "kickstart", "-k", "gui/501/ai.hermes.oauth-broker"],
        ["launchctl", "bootout", "gui/501/ai.hermes.oauth-broker"],
    ]
    assert target.read_bytes() == original


def test_install_apply_kickstart_interrupt_boots_out_and_restores_plist(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    original = b"synthetic-original-plist"
    target.write_bytes(original)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "kickstart":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        install_broker_service(
            plist_path=target,
            content=_render(tmp_path),
            apply=True,
            runner=runner,
            uid=501,
        )

    assert calls == [
        ["launchctl", "bootstrap", "gui/501", str(target)],
        ["launchctl", "kickstart", "-k", "gui/501/ai.hermes.oauth-broker"],
        ["launchctl", "bootout", "gui/501/ai.hermes.oauth-broker"],
    ]
    assert target.read_bytes() == original


def test_uninstall_render_only_keeps_plist_and_reports_argv(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    target.write_bytes(b"synthetic")
    result = uninstall_broker_service(plist_path=target, uid=501)
    assert result["executed"] is False
    assert target.exists()
    assert result["bootout"] == [
        "launchctl",
        "bootout",
        "gui/501/ai.hermes.oauth-broker",
    ]


def test_uninstall_apply_boots_out_and_removes_plist(tmp_path):
    target = tmp_path / "ai.hermes.oauth-broker.plist"
    target.write_bytes(b"synthetic")
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)

    result = uninstall_broker_service(
        plist_path=target, apply=True, runner=runner, uid=501
    )
    assert result["executed"] is True
    assert calls == [["launchctl", "bootout", "gui/501/ai.hermes.oauth-broker"]]
    assert not target.exists()

"""The Linux launcher asks the Hermes binary itself whether Chromium's sandbox starts (#130177).

Under ``kernel.apparmor_restrict_unprivileged_userns=1`` a generic ``unshare --user`` succeeds while
Chromium, unconfined, aborts with ``No usable sandbox!``; an AppArmor profile attached to the Hermes
path flips that answer for Hermes alone. The fake packaged app below answers the probe the way the
real Electron build does: exit 0 when its sandbox starts, Chromium's FATAL when it does not.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import main_desktop, main_desktop_apparmor

pytestmark = pytest.mark.platforms("linux")

_BLOCKED = (
    "#!/bin/sh\n"
    "echo 'FATAL:zygote_host_impl_linux.cc(128)] No usable sandbox!' >&2\n"
    "kill -TRAP $$\n"
)
_ALLOWED = "#!/bin/sh\nexit 0\n"


def _packaged_app(tmp_path, script: str):
    unpacked = tmp_path / "linux-unpacked"
    # electron-builder's layout: dist/ unpacked beside the asar. The main bundle carries the probe
    # switch, so the launcher knows this build answers it.
    dist = unpacked / "resources" / "app.asar.unpacked" / "dist"
    dist.mkdir(parents=True)
    (dist / "electron-main.mjs").write_text(
        f"if (process.argv.includes('{main_desktop_apparmor.SANDBOX_PROBE_SWITCH}')) process.exit(0)\n",
        encoding="utf-8")
    exe = unpacked / "Hermes"
    exe.write_text(script, encoding="utf-8")
    exe.chmod(0o755)
    helper = unpacked / "chrome-sandbox"
    helper.write_text("", encoding="utf-8")
    helper.chmod(0o755)  # user-owned, not setuid: the reporter's layout
    return exe


@pytest.fixture
def restricted_host(tmp_path, monkeypatch):
    sysctl = tmp_path / "apparmor_restrict_unprivileged_userns"
    sysctl.write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(main_desktop_apparmor, "RESTRICT_SYSCTL", sysctl)
    # No sudo on this host, so the setuid helper cannot be configured either.
    monkeypatch.setattr(main_desktop.shutil, "which", lambda name: None)


@pytest.mark.parametrize("restricted", [False, True])
def test_no_sandbox_fallback_names_apparmor_only_when_apparmor_is_the_reason(
        tmp_path, monkeypatch, capsys, restricted_host, restricted):
    """``ELECTRON_DISABLE_SANDBOX=1`` on an unrestricted host gets no AppArmor fix-it lines."""
    main_desktop_apparmor.RESTRICT_SYSCTL.write_text("1\n" if restricted else "0\n", encoding="utf-8")
    monkeypatch.setenv("ELECTRON_DISABLE_SANDBOX", "1")
    monkeypatch.setattr(main_desktop, "_desktop_linux_userns_sandbox_available", lambda: False)
    exe = _packaged_app(tmp_path, _BLOCKED)

    assert main_desktop._packaged_desktop_launch_command(exe) == [str(exe), "--no-sandbox"]
    out = capsys.readouterr().out
    assert ("--install-apparmor-profile" in out) is restricted
    assert ("AppArmor" in out) is restricted


def test_generic_unshare_success_does_not_override_a_blocked_binary(tmp_path, monkeypatch, restricted_host):
    """The reporter's host: unshare passes, Chromium cannot sandbox -> never the userns path."""
    exe = _packaged_app(tmp_path, _BLOCKED)
    monkeypatch.setattr(main_desktop, "_desktop_linux_userns_sandbox_available", lambda: True)

    assert main_desktop._desktop_linux_sandbox_fixup(exe) is False


def test_apparmor_profile_for_the_binary_enables_the_userns_path(tmp_path, monkeypatch, restricted_host):
    """A profile granting Hermes ``userns`` is honoured although unconfined probes fail."""
    exe = _packaged_app(tmp_path, _ALLOWED)
    monkeypatch.setattr(main_desktop, "_desktop_linux_userns_sandbox_available", lambda: False)

    assert main_desktop._desktop_linux_sandbox_fixup(exe) is True
    assert main_desktop._desktop_linux_needs_disable_setuid_sandbox(exe) is True


def _fake_root_tools(tmp_path, monkeypatch, *, parser_fails_on: str = ""):
    """``sudo`` that logs and runs its argv as this user; ``apparmor_parser`` failing on one flag."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    sudo_log = tmp_path / "sudo.log"
    (bin_dir / "sudo").write_text(f'#!/bin/sh\necho "$*" >> "{sudo_log}"\nexec "$@"\n', encoding="utf-8")
    (bin_dir / "apparmor_parser").write_text(
        f'#!/bin/sh\nfor a; do [ "$a" = "{parser_fails_on}" ] && exit 1; done\nexit 0\n', encoding="utf-8")
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    profile_dir = tmp_path / "apparmor.d"
    profile_dir.mkdir()
    monkeypatch.setattr(main_desktop_apparmor, "_PROFILE_DIR", profile_dir)
    return profile_dir, sudo_log


def test_install_path_with_apparmor_glob_characters_is_refused_before_sudo(tmp_path, monkeypatch):
    """``{x}`` breaks the profile parse and ``*``/``?``/``[`` widen it to other binaries."""
    profile_dir, sudo_log = _fake_root_tools(tmp_path, monkeypatch)
    exe = _packaged_app(tmp_path / "a{x}b", _ALLOWED)

    assert main_desktop_apparmor.install_profile(exe) is False
    assert not sudo_log.exists()
    assert list(profile_dir.iterdir()) == []


@pytest.mark.parametrize("failing_step", ["--skip-kernel-load", "-r"])
def test_profile_that_fails_to_parse_or_load_leaves_nothing_in_the_profile_dir(tmp_path, monkeypatch, failing_step):
    """The user-side parse check runs before any sudo write; a failed load removes the written file."""
    profile_dir, sudo_log = _fake_root_tools(tmp_path, monkeypatch, parser_fails_on=failing_step)
    exe = _packaged_app(tmp_path, _ALLOWED)

    assert main_desktop_apparmor.install_profile(exe) is False
    assert list(profile_dir.iterdir()) == []
    sudo_calls = sudo_log.read_text(encoding="utf-8").splitlines() if sudo_log.exists() else []
    # A parse failure never reaches sudo; a load failure wrote the file, then removed it again.
    expected = ["tee", "apparmor_parser", "rm"] if failing_step == "-r" else []
    assert [Path(call.split()[0]).name for call in sudo_calls] == expected


def test_a_failed_profile_write_keeps_the_installed_profile(tmp_path, monkeypatch):
    """A reinstall whose privileged write fails (disk full, I/O error) leaves the profile that was
    already installed byte-for-byte, and no partial file in the profile directory."""
    profile_dir, sudo_log = _fake_root_tools(tmp_path, monkeypatch)
    # The write dies at its first byte, the way a full disk or a file-size limit stops ``tee``.
    (tmp_path / "fakebin" / "sudo").write_text(
        f'#!/bin/sh\necho "$*" >> "{sudo_log}"\n[ "$1" = tee ] && ulimit -f 0\nexec "$@"\n', encoding="utf-8")
    exe = _packaged_app(tmp_path, _ALLOWED)
    installed = main_desktop_apparmor.profile_path(exe)
    installed.write_text(main_desktop_apparmor.profile_text(exe), encoding="utf-8")
    before = installed.read_bytes()

    assert main_desktop_apparmor.install_profile(exe) is False
    assert installed.read_bytes() == before
    assert list(profile_dir.iterdir()) == [installed]


def test_install_stages_loads_then_replaces_the_old_profile(tmp_path, monkeypatch):
    """The privileged steps run write-staged, load-staged, move-into-place; the old profile is
    replaced, the staged file is gone, and the binary's own probe confirms the sandbox."""
    profile_dir, sudo_log = _fake_root_tools(tmp_path, monkeypatch)
    sysctl = tmp_path / "apparmor_restrict_unprivileged_userns"
    sysctl.write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(main_desktop_apparmor, "RESTRICT_SYSCTL", sysctl)
    exe = _packaged_app(tmp_path, _ALLOWED)
    target = main_desktop_apparmor.profile_path(exe)
    target.write_text("# an older Hermes profile\n", encoding="utf-8")

    assert main_desktop_apparmor.install_profile(exe) is True
    calls = [call.split() for call in sudo_log.read_text(encoding="utf-8").splitlines()]
    staged = calls[0][1]
    assert [Path(call[0]).name for call in calls] == ["tee", "apparmor_parser", "mv"]
    assert calls[1][1:] == ["-r", staged]
    assert calls[2][1:] == ["-f", staged, str(target)]
    assert list(profile_dir.iterdir()) == [target]
    assert target.read_text(encoding="utf-8") == main_desktop_apparmor.profile_text(exe)

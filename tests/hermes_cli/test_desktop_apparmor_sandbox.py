"""The Linux launcher asks the Hermes binary itself whether Chromium's sandbox starts (#130177).

Under ``kernel.apparmor_restrict_unprivileged_userns=1`` a generic ``unshare --user`` succeeds while
Chromium, unconfined, aborts with ``No usable sandbox!``; an AppArmor profile attached to the Hermes
path flips that answer for Hermes alone. The fake packaged app below answers the probe the way the
real Electron build does: exit 0 when its sandbox starts, Chromium's FATAL when it does not.
"""

from __future__ import annotations

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
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
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

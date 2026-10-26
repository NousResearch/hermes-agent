"""The Linux launcher asks the Hermes binary itself whether Chromium's sandbox starts (#130177).

Under ``kernel.apparmor_restrict_unprivileged_userns=1`` a generic ``unshare --user`` succeeds while
Chromium, unconfined, aborts with ``No usable sandbox!``; an AppArmor profile attached to the Hermes
path flips that answer for Hermes alone. The fake packaged app below answers the probe the way the
real Electron build does: exit 0 when its sandbox starts, Chromium's FATAL when it does not.
"""

from __future__ import annotations

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
    (unpacked / "resources").mkdir(parents=True)
    # The bundled entry point carries the probe switch, so the launcher knows this build answers it.
    (unpacked / "resources" / "app.asar").write_bytes(
        b"\0asar\0" + main_desktop_apparmor.SANDBOX_PROBE_SWITCH.encode() + b"\0")
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

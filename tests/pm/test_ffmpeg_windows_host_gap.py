"""A Windows host that cannot start the pinned BtbN ffmpeg reports a gap instead of failing syncs (#130064).

BtbN autobuilds import ``SetThreadDescription`` from KERNEL32; Windows Server 2016 (build 14393)
does not export it there, so every ffmpeg invocation dies with 0xC0000139 and the install's
smoke probe fails every ``hermes update``. The gap must be host-scoped, pin-scoped (a non-BtbN
Windows pin clears it) and never touch other targets.
"""
from __future__ import annotations

import pytest

from pm import packages, registry
from pm.packages import Ffmpeg

BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-x/ffmpeg-n9.0.1-win64-gpl-9.0.zip"
GYAND = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"


@pytest.fixture
def host(monkeypatch):
    """A win32-x64 host whose lock pins ``state['url']`` and whose KERNEL32 lacks ``state['missing']``."""
    state = {"url": BTBN, "missing": ["SetThreadDescription"], "probed": 0}

    class _Lock:
        def __init__(self, *args, **kwargs):
            pass

        def artifacts(self, name, target):
            assert (name, target) == ("ffmpeg", "win32-x64")
            return [{"url": state["url"]}]

    def probe(names):
        state["probed"] += 1
        return [name for name in names if name in state["missing"]]

    monkeypatch.setattr(packages.sys, "platform", "win32")
    monkeypatch.setattr(packages, "current_target", lambda: "win32-x64")
    monkeypatch.setattr("pm.lock.Lockfile", _Lock)
    monkeypatch.setattr(packages, "_kernel32_missing_exports", probe)
    return state


def test_btbn_pin_on_a_kernel32_without_the_export_is_a_gap(host):
    reason = Ffmpeg().missing_reason("win32-x64")
    assert reason is not None and "SetThreadDescription" in reason


def test_gap_keeps_ffmpeg_out_of_the_required_tool_closure(host, monkeypatch):
    monkeypatch.setattr("pm.store.current_target", lambda: "win32-x64")
    assert "ffmpeg" not in registry.source_install_packages(["ffmpeg"])
    host["missing"] = []
    assert "ffmpeg" in registry.source_install_packages(["ffmpeg"])


def test_kernel32_with_the_export_has_no_gap(host):
    host["missing"] = []
    assert Ffmpeg().missing_reason("win32-x64") is None


def test_non_btbn_windows_pin_clears_the_gap_without_probing(host):
    host["url"] = GYAND
    assert Ffmpeg().missing_reason("win32-x64") is None
    assert host["probed"] == 0


def test_other_targets_and_non_windows_hosts_are_untouched(host, monkeypatch):
    assert Ffmpeg().missing_reason("win32-arm64") is None
    assert Ffmpeg().missing_reason("linux-x64") is None
    assert host["probed"] == 0
    monkeypatch.setattr(packages.sys, "platform", "linux")
    assert packages._win_ffmpeg_host_gap() is None


def test_probe_failure_fails_open(host, monkeypatch):
    def broken(names):
        raise OSError("no kernel32")

    monkeypatch.setattr(packages, "_kernel32_missing_exports", broken)
    assert Ffmpeg().missing_reason("win32-x64") is None

"""A desktop app swap that died between its two renames is finished on the next swap, never
erased: the previous app must survive a crash at the commit point and a failing retry."""

from __future__ import annotations

import os
from pathlib import Path

from hermes_cli import main_desktop
from tests.hermes_cli.test_desktop_swap_file_lock_retry import _staged_over_live


def test_swap_after_a_crash_between_renames_keeps_the_previous_app(tmp_path, monkeypatch):
    desktop_dir, staging, live_exe, _slept = _staged_over_live(tmp_path, monkeypatch)
    live_root = live_exe.parent
    while live_root.parent.name != "release":
        live_root = live_root.parent
    # Crash state: the live app was moved aside and nothing moved in.
    previous = live_root.with_name(live_root.name + main_desktop._DESKTOP_PREVIOUS_SUFFIX)
    live_root.rename(previous)
    real_rename = os.rename

    def promotion_fails(src, dst):
        if Path(src).parent == staging:
            raise OSError(5, "disk error")
        return real_rename(src, dst)

    monkeypatch.setattr(main_desktop.os, "rename", promotion_fails)
    monkeypatch.setattr(main_desktop, "_rename_riding_out_file_lock", lambda s, d: os.rename(s, d))

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8-sig") == "old"

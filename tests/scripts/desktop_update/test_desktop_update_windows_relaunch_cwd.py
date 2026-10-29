"""The Desktop the Windows hand-off relaunches never runs from its own app folder.

Everything the relaunched Desktop spawns without an explicit cwd (WSL/ssh helpers,
IME cloud helpers that outlive it) inherits its working directory, and Windows
refuses to rename a directory that is any live process's cwd. A relaunch from
``release\\win-unpacked`` therefore let one leftover helper block every later swap
of that folder (#101789). The relaunched process here is real: the hand-off
creates it through WMI exactly as production does.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'


@pytest.mark.platforms('windows')
def test_relaunched_desktop_does_not_hold_its_app_folder(tmp_path: Path) -> None:
    home = tmp_path / 'home'
    profile = tmp_path / 'profile'
    app_dir = tmp_path / 'release' / 'win-unpacked'
    for directory in (home, profile, app_dir):
        directory.mkdir(parents=True)
    # A console exe with no arguments waits on its own console: a stand-in Desktop
    # that stays alive until the test kills it.
    relaunch = app_dir / 'Hermes.exe'
    shutil.copy2(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'cmd.exe', relaunch)

    result = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(home / 'hermes-agent'), '-NoUi', '-SelfTestMarker', '-RelaunchExe', str(relaunch)],
        cwd=home, env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(home / 'empty-store'),
                       'USERPROFILE': str(profile)},
        capture_output=True, text=True, timeout=180, check=False,
    )
    log = (home / 'logs/desktop-update-handoff.log').read_text(encoding='utf-8-sig')
    launched = re.search(r'desktop relaunched detached \(pid (\d+)\)', log)
    assert result.returncode == 0 and launched, result.stdout + result.stderr + log
    pid = int(launched.group(1))
    try:
        assert Path(psutil.Process(pid).cwd()).resolve() == profile.resolve()
        # The next update's swap renames the app folder while the relaunched process lives.
        moved = app_dir.with_name('win-unpacked.previous')
        os.rename(app_dir, moved)
        os.rename(moved, app_dir)
    finally:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], capture_output=True, check=False)

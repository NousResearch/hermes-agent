"""The Windows hand-off relays the stages `hermes update` publishes, instead of one static line.

The real ``windows.ps1`` runs a fixture ``hermes`` whose update publishes a stage through the real
``hermes_cli.update_stage.publish_stage`` and then waits to be released. On main the hand-off never
told the child where to publish, so the window (and the hand-off log) read "Updating code and
dependencies" for the whole 20-60 minute run (#139952, #135363).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher
from tests.scripts.desktop_update.windows_handoff_support import _hold_against_writers, _log, _release
from tests.waiting import wait_until

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / "scripts/desktop-update/windows.ps1"
STAGE_CLI = """
import os, sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] == ['update']:
        sys.path.insert(0, os.environ['HANDOFF_REPO'])
        # The fixture install ships its own hermes_cli package: load the tree's.
        for name in [m for m in sys.modules if m == 'hermes_cli' or m.startswith('hermes_cli.')]:
            del sys.modules[name]
        from hermes_cli.update_stage import publish_stage
        from tests.waiting import wait_until
        publish_stage('Packaging the desktop app')
        wait_until(Path(os.environ['HANDOFF_HOLD']).exists, timeout=60, what='the test to release the update')
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""


@pytest.mark.platforms("windows")
def test_handoff_shows_the_stage_the_update_publishes(tmp_path: Path) -> None:
    install = tmp_path / "checkout"
    publish_fixture_launcher(install, STAGE_CLI)
    home = tmp_path / "home"
    home.mkdir()
    hold = tmp_path / "release-update"
    script = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-InstallRoot", str(install), "-NoUi"],
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "HERMES_HOME": str(home), "HERMES_RUNTIME_DIR": str(tmp_path / "empty-store"),
             "HANDOFF_HOLD": str(hold), "HANDOFF_REPO": str(ROOT)},
    )
    try:
        wait_until(lambda: "stage: Packaging the desktop app" in _log(home) or script.poll() is not None,
                   timeout=90, what="the hand-off to relay the update's stage")
        assert "stage: Packaging the desktop app" in _log(home), (
            "the hand-off never relayed the update's own stage:\n" + _log(home)[-3000:])
    finally:
        hold.write_text("", encoding="utf-8")
        try:
            script.wait(timeout=120)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(script.pid)], capture_output=True, check=False)
            script.wait(timeout=30)



HELD_LOG_CLI = """
import os, sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] == ['update']:
        sys.path.insert(0, os.environ['HANDOFF_REPO'])
        for name in [m for m in sys.modules if m == 'hermes_cli' or m.startswith('hermes_cli.')]:
            del sys.modules[name]
        from hermes_cli.update_stage import publish_stage
        from tests.waiting import wait_until
        Path(os.environ['HANDOFF_STARTED']).write_text('', encoding='utf-8')
        wait_until(Path(os.environ['HANDOFF_RELEASED']).exists, timeout=60, what='the log to be released')
        publish_stage('Packaging the desktop app')
        wait_until(Path(os.environ['HANDOFF_HOLD']).exists, timeout=60, what='the test to release the update')
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""


@pytest.mark.platforms("windows")
def test_lines_logged_while_another_process_holds_the_log_land_once_it_lets_go(tmp_path: Path) -> None:
    install = tmp_path / "checkout"
    publish_fixture_launcher(install, HELD_LOG_CLI)
    home = tmp_path / "home"
    log = home / "logs" / "desktop-update-handoff.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"")
    started, released, hold = tmp_path / "started", tmp_path / "released", tmp_path / "release-update"
    handle = _hold_against_writers(log)
    script = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-InstallRoot", str(install), "-NoUi"],
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "HERMES_HOME": str(home), "HERMES_RUNTIME_DIR": str(tmp_path / "empty-store"),
             "HANDOFF_STARTED": str(started), "HANDOFF_RELEASED": str(released), "HANDOFF_HOLD": str(hold),
             "HANDOFF_REPO": str(ROOT)},
    )
    try:
        # Everything up to the update's own start was logged while the log was held.
        wait_until(lambda: started.exists() or script.poll() is not None, timeout=90, what="the update to start")
        _release(handle)
        handle = 0
        released.write_text("", encoding="utf-8")
        wait_until(lambda: "stage: Packaging the desktop app" in _log(home) or script.poll() is not None,
                   timeout=90, what="the hand-off to relay the update's stage")
        text = _log(home)
        assert "running: python" in text and "stage: Packaging the desktop app" in text, (
            "lines logged while the log was held were lost:\n" + text[-3000:])
    finally:
        if handle:
            _release(handle)
        hold.write_text("", encoding="utf-8")
        try:
            script.wait(timeout=120)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(script.pid)], capture_output=True, check=False)
            script.wait(timeout=30)

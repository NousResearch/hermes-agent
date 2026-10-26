"""#101789 on the real Windows hand-off script: a Desktop swap blocked by a program outside Hermes.

A real ``windows.ps1`` drives a published launcher whose ``hermes update`` is a scripted stand-in
printing exactly what ``hermes_cli.desktop_release_holders`` prints when Windows refused the
promotion rename. The result the relaunched Desktop reads must name that program (message and a
structured ``release_holders`` list), so the Desktop can offer to close exactly it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
EXE = r'D:\Program Files (x86)\SogouInput\16.7.0.4912\SogouCloud.exe'
CLI = """
import sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] != ['update']:
        return 0
    print('  Desktop app folder held by: SogouCloud.exe (pid 4242, EXE_PATH)', flush=True)
    print('  Desktop app build owed: desktop app build failed', flush=True)
    print('Update complete!', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
""".replace('EXE_PATH', EXE.replace('\\', '\\\\'))


@pytest.mark.platforms('windows')
def test_blocked_swap_result_names_the_holder_for_the_desktop(tmp_path: Path) -> None:
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n', encoding='utf-8')
    home = tmp_path / 'home'
    home.mkdir()
    proc = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(install), '-NoUi', '-NoGateway'],
        check=False, cwd=tmp_path, capture_output=True, text=True, timeout=150,
        env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
             'PYTHONIOENCODING': 'utf-8'},
    )
    result_path = home / '.hermes-update-result.json'
    log = home / 'logs' / 'desktop-update-handoff.log'
    assert result_path.is_file(), (proc.stdout, log.read_text(encoding='utf-8', errors='replace') if log.is_file() else '')
    result = json.loads(result_path.read_text(encoding='utf-8-sig'))

    assert (result['ok'], result['manual']) == (True, True), (proc.stdout, result)
    assert result['release_holders'] == [{'name': 'SogouCloud.exe', 'pid': 4242, 'exe': EXE}], result
    assert 'SogouCloud.exe (pid 4242)' in result['message'], result
    assert 'not part of Hermes' in result['message'], result

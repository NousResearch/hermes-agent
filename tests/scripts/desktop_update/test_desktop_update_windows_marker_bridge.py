"""Windows marker bridge contract on real processes."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import _creation_time
from tests.scripts.desktop_update import windows_marker_support
from tests.scripts.desktop_update.windows_marker_support import (MARKER, _result, _run)

sleeper = windows_marker_support.sleeper

@pytest.mark.platforms('windows')
@pytest.mark.parametrize('bridge', ['absent', 'someone-else'])
def test_desktop_started_handoff_only_adopts_its_bridge(
    tmp_path: Path, sleeper: subprocess.Popen, bridge: str,
) -> None:
    """A4: the Desktop gave up on a late script: it must not claim fresh, run, or leave a result."""
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    try:
        body = f'{other.pid}\n{int(time.time())}\nct:{_creation_time(other.pid)}\n'.encode()
        if bridge == 'someone-else':
            (tmp_path / MARKER).write_bytes(body)
        _, code, out = _run(tmp_path, '-DesktopPid', str(sleeper.pid))
    finally:
        other.kill(); other.wait()
    assert code == 2, out
    assert not (tmp_path / '.hermes-update-result.json').exists()
    if bridge == 'absent':
        assert not (tmp_path / MARKER).exists()
    else:
        assert (tmp_path / MARKER).read_bytes() == body


@pytest.mark.platforms('windows')
def test_live_desktop_bridge_marker_is_adopted_keeping_its_started_at(
    tmp_path: Path, sleeper: subprocess.Popen,
) -> None:
    started = int(time.time()) - 30
    (tmp_path / MARKER).write_bytes(f'{sleeper.pid}\n{started}\nct:{_creation_time(sleeper.pid)}\n'.encode())
    pid, code, out = _run(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup', '-DesktopPid', str(sleeper.pid))
    assert code == 0, out
    lines = (tmp_path / MARKER).read_bytes().decode().split('\n')
    assert lines[:2] == [str(pid), str(started)], lines
    assert lines[2].startswith('ct:'), lines
    assert _result(tmp_path)['started_at'] == started

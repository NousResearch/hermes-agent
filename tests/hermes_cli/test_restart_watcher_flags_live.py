import sys
import time
import pytest
from hermes_cli import gateway


@pytest.mark.parametrize('wait_for_exit', [True, False])
def test_real_watcher_respawns_both_flags(tmp_path, monkeypatch, wait_for_exit):
    marker = tmp_path / 'respawned'
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(gateway, 'GATEWAY_RESTART_WATCHER_TIMEOUT_S', 0.1)
    argv = [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).write_text("ok")']
    assert gateway._spawn_gateway_restart_watcher(99999999, argv, host=False, wait_for_exit=wait_for_exit)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.read_text() == 'ok'

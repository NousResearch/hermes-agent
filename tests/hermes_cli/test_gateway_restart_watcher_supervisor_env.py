"""The detached restart watcher's respawn never claims the systemd unit that armed it (#73480).

An update run from a systemd-supervised gateway (or any process below one) carries the unit's
``INVOCATION_ID``/``JOURNAL_STREAM``. A respawned ``gateway run --replace`` holding them believes
systemd launched it, skips the supervised-gateway conflict guard, and races the unit for the
profile's PID file and Telegram long-poll.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from hermes_cli import gateway

MARKERS = {"INVOCATION_ID": "unit-invocation", "JOURNAL_STREAM": "8:1234",
           "NOTIFY_SOCKET": "/run/user/1000/systemd/notify", "WATCHDOG_PID": "4242"}


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("host", [False, True], ids=["profile", "host"])
def test_respawned_gateway_does_not_inherit_systemd_markers(tmp_path, monkeypatch, host):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key, value in MARKERS.items():
        monkeypatch.setenv(key, value)
    witness = tmp_path / "respawn-env.json"
    respawn = [sys.executable, "-c",
               "import json, os, sys; open(sys.argv[1], 'w').write(json.dumps("
               f"{{k: os.environ.get(k) for k in {sorted(MARKERS)!r}}}))", str(witness)]

    assert gateway._spawn_gateway_restart_watcher(_dead_pid(), respawn, host=host)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not witness.exists():
        time.sleep(0.1)
    assert witness.exists(), "the restart watcher never respawned the gateway command"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not witness.read_text():
        time.sleep(0.05)
    assert json.loads(witness.read_text()) == dict.fromkeys(MARKERS)

"""The detached restart watcher's respawn never claims the systemd unit that armed it (#73480).

An update run from a systemd-supervised gateway (or any process below one) carries the unit's
``INVOCATION_ID``/``JOURNAL_STREAM``. A respawned ``gateway run --replace`` holding them believes
systemd launched it, skips the supervised-gateway conflict guard, and races the unit for the
profile's PID file and Telegram long-poll. It must still stay on the profile it ran on: a default
respawn has no ``--profile``, so losing the marker must not let the sticky ``active_profile`` win.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from hermes_cli import gateway
from tests.waiting import wait_until

MARKERS = {"INVOCATION_ID": "unit-invocation", "JOURNAL_STREAM": "8:1234",
           "NOTIFY_SOCKET": "/run/user/1000/systemd/notify", "WATCHDOG_PID": "4242"}

_RESPAWN = """import json, os, sys
sys.path.insert(0, sys.argv[2])
from gateway.restart import is_gateway_supervisor_process
from hermes_cli.main import _under_gateway_supervisor
open(sys.argv[1] + ".tmp", "w").write(json.dumps({
    "markers": {k: os.environ.get(k) for k in json.loads(sys.argv[3])},
    "claims_supervisor": is_gateway_supervisor_process(),
    "keeps_profile": _under_gateway_supervisor(["gateway", "run", "--replace"])}))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
"""


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("host", [False, True], ids=["profile", "host"])
def test_respawned_gateway_drops_the_unit_identity_but_keeps_its_profile(tmp_path, monkeypatch, host):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_SUPERVISED_CHILD", "HERMES_GATEWAY_EXTERNAL_SUPERVISOR", "HERMES_GATEWAY_PROFILE_PIN"):
        monkeypatch.delenv(key, raising=False)
    for key, value in MARKERS.items():
        monkeypatch.setenv(key, value)
    witness = tmp_path / "respawn-env.json"
    respawn = [sys.executable, "-c", _RESPAWN, str(witness), str(gateway.PROJECT_ROOT), json.dumps(sorted(MARKERS))]

    assert gateway._spawn_gateway_restart_watcher(_dead_pid(), respawn, host=host)

    wait_until(witness.exists, timeout=30, what="the restart watcher respawning the gateway command")
    assert json.loads(witness.read_text()) == {
        "markers": dict.fromkeys(MARKERS), "claims_supervisor": False, "keeps_profile": True}

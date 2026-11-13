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
from hermes_agent.gateway.restart import is_gateway_supervisor_process
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


_LAUNCHD = {"XPC_SERVICE_NAME": "ai.hermes.gateway", "HERMES_LAUNCHD_LABEL": "ai.hermes.gateway",
            "HERMES_SUPERVISED_CHILD": "1", "INVOCATION_ID": "unit-invocation", "JOURNAL_STREAM": "8:1"}


@pytest.mark.platforms("posix")
def test_a_respawn_on_the_launchers_own_profile_sheds_the_markers_too(tmp_path, monkeypatch):
    """The watcher inherits the launcher's env for its own profile (an exported token may be its only
    credential); that path must still drop the unit identity, or the respawn races the unit."""
    from hermes_cli.gateway_restart_env import restart_watcher_env
    root = tmp_path / ".hermes"
    api1 = root / "profiles" / "api1"
    api1.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(api1))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "exported-api1-token")
    for key, value in _LAUNCHD.items():
        monkeypatch.setenv(key, value)
    env = restart_watcher_env(["--profile", "api1", "gateway", "run", "--replace"], host=False, home=str(api1))
    assert env["TELEGRAM_BOT_TOKEN"] == "exported-api1-token"
    assert not any(key in env for key in _LAUNCHD)


@pytest.mark.platforms("posix")
def test_a_paused_gateway_replayed_from_a_supervised_update_claims_no_supervisor(tmp_path, monkeypatch):
    """``/update`` sent to a launchd gateway pauses a bare gateway and replays it later. The replay
    env used to drop only INVOCATION_ID/JOURNAL_STREAM, so it still claimed launchd."""
    from hermes_agent.gateway.restart import is_gateway_supervisor_process
    from hermes_cli.update_cmd_posix_pause import _bare_replay_env
    root = tmp_path / ".hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    for key, value in _LAUNCHD.items():
        monkeypatch.setenv(key, value)
    for home in (root / "profiles" / "work", root):
        env = _bare_replay_env(str(home))
        assert not any(key in env for key in _LAUNCHD), home
        assert not is_gateway_supervisor_process(env)

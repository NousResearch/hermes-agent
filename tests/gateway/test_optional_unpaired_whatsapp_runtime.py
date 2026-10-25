"""An unpaired optional adapter must not stop ordinary local CLI sessions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from gateway.run_startup import GatewayStartupMixin
from tests.gateway.test_normal_runtime_boot import control


class _NoConnections:
    _start_handle_no_connections = GatewayStartupMixin._start_handle_no_connections

    def __init__(self):
        self.session_authority = object()
        self._startup_parked_platforms = False
        self._startup_ownership_conflict = False
        self.fatal = []

    def _startup_fail_fatal_config(self, reason):
        self.fatal.append(reason)


@pytest.mark.parametrize("retryable, nonretryable, parked, degraded_write", [
    ([], ["whatsapp: not paired"], True, []),                         # the exit-78 case: local clients keep the runtime
    (["telegram: TimedOut"], [], False, ["degraded"]),                 # nothing serves: degraded, not a healthy run
    (["telegram: TimedOut"], ["whatsapp: not paired"], True, ["degraded"]),
])
def test_local_runtime_keeps_failed_startup_degraded(monkeypatch, retryable, nonretryable, parked, degraded_write):
    import gateway.run as run_mod
    writes = []
    monkeypatch.setattr(run_mod, "_write_runtime_status_quiet",
                        lambda **k: writes.append(k["gateway_state"]), raising=False)
    runner = _NoConnections()
    assert runner._start_handle_no_connections(0, 1, list(retryable), list(nonretryable)) is False
    assert (runner._startup_parked_platforms, writes, runner.fatal) == (parked, degraded_write, [])


def test_ownership_conflict_still_exits_with_local_runtime():
    runner = _NoConnections()
    runner._startup_ownership_conflict = True
    assert runner._start_handle_no_connections(0, 1, [], ["telegram: token held by another gateway"]) is True
    assert runner.fatal


def test_unpaired_whatsapp_keeps_local_runtime_available(tmp_path, model_peer):
    home = tmp_path / "state"
    home.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    endpoint = f"http://127.0.0.1:{model_peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "local-wire-stub", "base_url": endpoint},
        "auxiliary": {"title_generation": {"enabled": False}},
    }))
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=endpoint, WHATSAPP_ENABLED="true")
    with (tmp_path / "gateway.log").open("w+") as log:
        process = subprocess.Popen([sys.executable, "-B", "-m", "gateway.run"], cwd=root,
                                   env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        descriptor = {}
        try:
            deadline = time.monotonic() + 35
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, "identify")
                    if descriptor.get("state") == "ready":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.flush()
            log.seek(0)
            assert descriptor.get("state") == "ready", (process.poll(), descriptor, log.read())
            assert "session-authority-v1" in descriptor["capabilities"]
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=20) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

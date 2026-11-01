"""Real gateway and client signal cancellation against a loopback-only model."""
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("signal_name,expected", [("SIGINT", 130), ("SIGTERM", 143)])
def test_signal_stops_real_gateway_turn(tmp_path, signal_name, expected):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "signal-model", "base_url": url},
        "auxiliary": {"title_generation": {"enabled": False}},
    }))
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ", "TMPDIR") if key in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)
    client = None
    with (home / "gateway.log").open("w+") as log:
        daemon = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=root, env=env,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            def ready():
                try:
                    return control(home, "identify").get("state") == "ready"
                except (OSError, ValueError):
                    return False

            assert wait_until(lambda: ready() or daemon.poll() is not None, timeout=60)
            assert daemon.poll() is None and ready(), (home / "gateway.log").read_text()
            client = subprocess.Popen([
                sys.executable, "-m", "hermes_cli.main", "chat", "--oneshot", "--interrupt-on-exit",
                "--format", "stream-json", "--source", "tool", "-q", "BLOCK_STARTED",
            ], cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert wait_until(lambda: peer.blocked.is_set() or client.poll() is not None, timeout=40)
            assert peer.blocked.is_set() and client.poll() is None, client.communicate(timeout=5)
            client.send_signal(getattr(signal, signal_name))
            out, err = client.communicate(timeout=10)
            assert client.returncode == expected, (out, err)
            results = [event for line in out.splitlines() if (event := json.loads(line))["type"] == "result"]
            assert len(results) == 1 and results[0]["exit_code"] == expected, out
            assert "Cancellation requested" in err
            sid = results[0]["session_id"]
            peer.release.set()

            def settled():
                with sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True) as db:
                    return db.execute("SELECT status, outcome FROM session_admissions WHERE target_session_id=?",
                                      (sid,)).fetchall()

            assert wait_until(lambda: settled() == [("terminal", "interrupted")], timeout=30), settled()
            assert daemon.poll() is None  # only the client's turn stopped, not the gateway
        finally:
            peer.release.set()
            if client is not None and client.poll() is None:
                client.kill()
                client.communicate(timeout=5)
            if daemon.poll() is None:
                daemon.send_signal(signal.SIGINT)
                try:
                    daemon.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait(timeout=5)
            peer.shutdown()
            peer.server_close()

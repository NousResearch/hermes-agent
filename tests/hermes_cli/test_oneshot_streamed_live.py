"""`hermes -z` against a real gateway with streaming on: the one-shot prints the reply (#139666).

With ``streaming.enabled`` the local route's stream consumer "delivers" the reply through
``LocalSessionAdapter`` (every send acknowledges), the turn is marked ``already_sent`` and the
completion frame used to carry no text. A finite viewer drops ``message.delta`` frames and prints
only that text, so `-z` printed a bare newline and exited 1 while the answer sat in state.db."""
import json
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model
from tests.gateway.test_normal_runtime_boot import control


@pytest.mark.platforms("linux")
def test_streamed_oneshot_prints_the_reply(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / "state", tmp_path / "user", tmp_path / "work"
    for path in (user, work):
        path.mkdir()
    home.mkdir(mode=0o700)
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "streaming": {"enabled": True},
        "model": {"provider": "custom", "default": "oneshot-model", "base_url": url},
        "auxiliary": {"title_generation": {"enabled": False}}}))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)

    def hermes(*args):
        return subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], cwd=work, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)

    with (home / "gw.log").open("w") as log:
        daemon = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=root, env=env,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline, desc = time.monotonic() + 60, {}
            while daemon.poll() is None and time.monotonic() < deadline and desc.get("state") != "ready":
                try:
                    desc = control(home, "identify")
                except (OSError, ValueError):
                    time.sleep(.1)
            assert desc.get("state") == "ready", (home / "gw.log").read_text()

            oneshot = hermes("-z", "streamed one shot")
            assert peer.requests and peer.requests[-1].get("stream"), "the turn must stream"
            assert (oneshot.returncode, oneshot.stdout) == (0, "RECOVERY_ACK_streamed one shot\n"), oneshot
            quiet = hermes("chat", "-q", "streamed quiet query", "-Q")
            assert quiet.returncode == 0 and quiet.stdout.strip() == "RECOVERY_ACK_streamed quiet query", quiet
        finally:
            if daemon.poll() is None:
                daemon.send_signal(signal.SIGINT)
                try:
                    daemon.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait(timeout=5)
            peer.shutdown()
            peer.server_close()

"""An unpaired optional adapter must not stop ordinary local CLI sessions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tests.gateway.test_normal_runtime_boot import control


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

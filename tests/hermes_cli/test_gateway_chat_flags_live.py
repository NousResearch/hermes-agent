"""Gateway-path launch flags main supported: bare -c, --resume latest, --list-tools, resume footer."""
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
def test_continue_latest_footer_and_tool_listing(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / "state", tmp_path / "user"
    repo_a, repo_b = tmp_path / "repo-a", tmp_path / "repo-b"
    for path in (user, repo_a, repo_b):
        path.mkdir()
    home.mkdir(mode=0o700)
    for repo in (repo_a, repo_b):
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "flags-model", "base_url": url},
        "auxiliary": {"title_generation": {"enabled": False}}}))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)

    def hermes(*args, cwd, module="hermes_cli.main"):
        result = subprocess.run([sys.executable, "-m", module, *args], cwd=cwd, env=env,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
        sid = next((line.split("Session: ", 1)[1].strip() for line in result.stderr.splitlines()
                    if line.startswith("Session: ")), None)
        return result, sid

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

            empty, _ = hermes("chat", "-c", "-q", "nothing yet", "-Q", cwd=repo_a)
            assert empty.returncode == 1 and "No previous CLI session" in empty.stderr, empty

            first, sid_a = hermes("chat", "-q", "in repo a", cwd=repo_a)
            assert first.returncode == 0 and "RECOVERY_ACK_in repo a" in first.stdout, first
            # Main's single-query exit block: how to continue the session it just printed.
            assert f"Resume this session with:\n  hermes --resume {sid_a}" in first.stdout, first.stdout
            quiet, sid_b = hermes("chat", "-q", "in repo b", "-Q", cwd=repo_b)
            assert quiet.returncode == 0 and "Resume this session" not in quiet.stdout, quiet

            # Bare -c continues THIS workspace's most recent CLI session, not the global newest.
            cont, cont_sid = hermes("chat", "-c", "-q", "again in a", "-Q", cwd=repo_a)
            assert cont.returncode == 0 and cont_sid == sid_a, (cont, sid_a, sid_b)
            history = json.dumps(peer.requests[-1]["messages"])
            assert "in repo a" in history and "in repo b" not in history
            latest, latest_sid = hermes("chat", "--resume", "latest", "-q", "latest in b", "-Q", cwd=repo_b)
            assert latest.returncode == 0 and latest_sid == sid_b, latest

            tools, _ = hermes("--list-toolsets", cwd=repo_a, module="cli")
            assert tools.returncode == 0 and "terminal" in tools.stdout, tools
            listing, _ = hermes("--list-tools", "--toolsets", "file", cwd=repo_a, module="cli")
            assert listing.returncode == 0 and "read_file" in listing.stdout and "[file]" in listing.stdout, listing
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

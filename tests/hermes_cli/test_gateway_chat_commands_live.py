"""Classic `hermes --cli chat` session commands on the shared gateway, driven in a real PTY."""
import json
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model
from tests.gateway.test_normal_runtime_boot import control


@pytest.mark.platforms("linux")
def test_classic_view_title_undo_retry_new_usage_tools_and_reads(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, cwd = tmp_path / "state", tmp_path / "user", tmp_path / "caller"
    home.mkdir(mode=0o700)
    user.mkdir()
    cwd.mkdir()
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "view-model", "base_url": url},
        "auxiliary": {"title_generation": {"enabled": False}}}))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)
    tmux_name = "hermes-cli-cmds-" + uuid.uuid4().hex

    def tmux(*args):
        return subprocess.run(["tmux", "-L", tmux_name, *args], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=10, check=False)

    def screen():
        return tmux("capture-pane", "-p", "-J", "-S", "-3000", "-t", "chat").stdout

    def send(line, expect, timeout=45):
        # The pane is fixed-height (blank rows pad it), so a new answer is one more occurrence.
        seen = screen().count(expect)
        tmux("send-keys", "-t", "chat", "-l", line)
        tmux("send-keys", "-t", "chat", "Enter")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = screen()
            if current.count(expect) > seen:
                return current
            time.sleep(.1)
        raise AssertionError((line, expect, screen(), (home / "gw.log").read_text()))

    def db(sql, *args):
        with sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True) as conn:
            return conn.execute(sql, args).fetchall()

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
            command = shlex.join([sys.executable, "-m", "hermes_cli.main", "--cli", "chat"])
            tmux("new-session", "-d", "-s", "chat", "-x", "200", "-y", "50", "-c", str(cwd),
                 command + '; printf "\\nCLI_RC=%s\\n" "$?"; sleep 120')
            send("", "Welcome to Hermes", 60)
            sid = next(line.split("Session: ", 1)[1].strip() for line in screen().splitlines() if "Session: " in line)
            send("first question", "RECOVERY_ACK_first question")
            send("second question", "RECOVERY_ACK_second question")
            send("/title View Title", "Session title set: View Title")
            assert db("SELECT title FROM sessions WHERE id=?", sid) == [("View Title",)]
            send("/title", "**View Title**")
            send("/usage", "Last turn:")
            send("/tools", "not available on the shared gateway yet")
            send("/retry", "RECOVERY_ACK_second question")
            active = [r[0] for r in db("SELECT content FROM messages WHERE session_id=? AND active=1 AND role='user'", sid)]
            assert active == ["first question", "second question"], active
            send("/undo", "Undid 1 turn")
            # The removed message is back in the composer: Enter alone resubmits it.
            send("", "RECOVERY_ACK_second question")
            send("/memory", "No pending memory writes")
            send("/insights", "**Sessions:** 1")
            send("/fast", "/fast is not available on the shared gateway yet")
            send("/new Second Thread", "Session title set: Second Thread")
            second = [line.split("Session: ", 1)[1].strip() for line in screen().splitlines() if "Session: " in line][-1]
            assert second != sid
            assert db("SELECT title FROM sessions WHERE id=?", second) == [("Second Thread",)]
            send("third question", "RECOVERY_ACK_third question")
            assert db("SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user'", second) == [(1,)]
            send("/quit", "CLI_RC=0")
            assert "Unsupported gateway CLI command" not in screen()
        finally:
            tmux("kill-server")
            if daemon.poll() is None:
                daemon.send_signal(signal.SIGINT)
                try:
                    daemon.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait(timeout=5)
            peer.shutdown()
            peer.server_close()

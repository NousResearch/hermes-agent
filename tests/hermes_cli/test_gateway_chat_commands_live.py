"""Classic `hermes --cli chat` session commands on the shared gateway, driven in a real PTY.

The classic CLI is main's HermesCLI again (gateway-backed agent and session store), so each
expectation is main's own output for the same keystrokes (lanes/pushready/probe_cmds.py, main =
git archive 3b63f7b0a4c). Keys go in as the A/B harness types them: the text, then Enter apart
(main's prompt_toolkit drops an Enter that arrives in the same burst as the text).
"""
import contextlib
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

from hermes_cli.observability.shared_metrics_consent import OFFER_VERSION
from tests.gateway.fixtures.local_recovery_probe import Model
from tests.gateway.test_normal_runtime_boot import control


@contextlib.contextmanager
def classic_session(tmp_path):
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
        "auxiliary": {"title_generation": {"enabled": False}},
        # Answered at the current offer version, as tests/fakes/fake_llm_provider.write_hermes_home
        # does: main's interactive chat asks it first on a home that never answered.
        "telemetry": {"shared_metrics": {"enabled": False, "send": False, "offer_version": OFFER_VERSION}}}))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url)
    tmux_name = "hermes-cli-cmds-" + uuid.uuid4().hex

    def tmux(*args):
        return subprocess.run(["tmux", "-L", tmux_name, *args], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=10, check=False)

    def screen():
        return tmux("capture-pane", "-p", "-J", "-S", "-3000", "-t", "chat").stdout

    def composer_shows(text):
        return any(row.startswith("❯") and text in row for row in screen().splitlines())

    def send(line, expect, timeout=45, composer=None):
        """``composer``: text the CLI itself put in the composer (``/undo``'s prefill), awaited like
        typed text. An Enter within 50 ms of any buffer change is a newline (#10994's paste guard,
        main's own), so a bare Enter right behind the prefill would only add a line."""
        # The pane is fixed-height (blank rows pad it), so a new answer is one more occurrence.
        seen = screen().count(expect)
        if line or composer:
            if line:
                tmux("send-keys", "-t", "chat", "-l", line)
            echo = time.monotonic() + 10
            while (line not in screen()) if line else not composer_shows(composer):
                assert time.monotonic() < echo, (line or composer, screen())
                time.sleep(.1)
            time.sleep(1)  # health: allow HX013 -- main drops an Enter sent in the same burst as the text (probe_pty.py); the gap is the input timing
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

    def session_id():
        """Main's bare `/title` prints the full id (the banner shows main's 8-character label)."""
        shown = send("/title", "Session ID: ")
        return [line.split("Session ID: ", 1)[1].strip() for line in shown.splitlines() if "Session ID: " in line][-1]

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
            deadline = time.monotonic() + 60
            # The banner prints before prompt_toolkit owns the tty: keys typed then are echoed raw and
            # reach the app in one burst with their Enter (a newline, not a submit). The drawn
            # composer row means the app is reading keys.
            while not ("Welcome to Hermes Agent" in screen() and composer_shows("")):
                assert time.monotonic() < deadline, (screen(), (home / "gw.log").read_text())
                time.sleep(.1)
            yield send, screen, db, session_id
            send("/quit", "CLI_RC=0")
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


@pytest.mark.platforms("linux")
def test_classic_view_title_and_usage_match_main(tmp_path):
    with classic_session(tmp_path) as (send, screen, db, session_id):
        send("first question", "RECOVERY_ACK_first question")
        send("second question", "RECOVERY_ACK_second question")
        send("/title View Title", "Session title set: View Title")
        sid = session_id()
        assert "Title: View Title" in screen()
        assert db("SELECT title FROM sessions WHERE id=?", sid) == [("View Title",)]
        send("/usage", "Session Token Usage")
        active = [r[0] for r in db("SELECT content FROM messages WHERE session_id=? AND active=1 AND role='user'", sid)]
        assert active == ["first question", "second question"], active


@pytest.mark.platforms("linux")
def test_classic_view_retry_undo_new_match_main(tmp_path):
    with classic_session(tmp_path) as (send, screen, db, session_id):
        send("first question", "RECOVERY_ACK_first question")
        send("second question", "RECOVERY_ACK_second question")
        sid = session_id()
        send("/retry", "RECOVERY_ACK_second question")
        assert 'Retrying: "second question"' in screen()
        # Main confirms a destructive command; Enter takes "Approve Once".
        send("/undo", "destroys conversation state")
        send("", "Undid 1 turn")
        # The removed message is back in the composer: Enter alone resubmits it.
        send("", "RECOVERY_ACK_second question", composer="second question")
        send("/new Second Thread", "destroys conversation state")
        send("", "New session started")
        second = session_id()
        assert second != sid
        assert db("SELECT title FROM sessions WHERE id=?", second) == [("Second Thread",)]


@pytest.mark.platforms("linux")
def test_classic_view_plan_and_goal_run_on_the_gateway(tmp_path):
    """/plan queues main's plan-mode turn; /goal sets the session's goal and kicks it off. Both reach
    the model as normal turns of this session (the loopback model echoes the user text)."""
    with classic_session(tmp_path) as (send, _screen, db, session_id):
        send("first question", "RECOVERY_ACK_first question")
        send("/plan add a cache", "RECOVERY_ACK_[/plan")
        send("/goal ship the cache", "Goal set")
        send("/goal status", "Goal (")
        sid = session_id()
        users = [r[0] for r in db("SELECT content FROM messages WHERE session_id=? AND role='user'", sid)]
        assert any("add a cache" in u and "PLAN MODE" in u for u in users), users
        assert "ship the cache" in users, users

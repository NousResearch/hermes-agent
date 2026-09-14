"""Title-keyed resume on the canonical CLI client against an ordinary daemon."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tests.gateway.test_normal_runtime_boot import control, model_peer  # noqa: F401


@pytest.mark.linux_only
def test_title_create_if_missing_then_resume_by_title_is_deterministic(tmp_path, model_peer):
    home = tmp_path / "state"
    home.mkdir(mode=0o700)
    user = tmp_path / "user"
    user.mkdir()
    root = Path(__file__).resolve().parents[2]
    model_url = f"http://127.0.0.1:{model_peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "local-wire-stub", "base_url": model_url},
        "auxiliary": {"title_generation": {"enabled": False}},
    }), encoding="utf-8")
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED="1", OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=model_url)
    witness = tmp_path / "client-owner.json"
    wrapper = tmp_path / "client.py"
    wrapper.write_text(encoding='utf-8', data=f'''import json, runpy, sys
owners = []
def trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == '__init__':
        if type(frame.f_locals.get('self')).__name__ in {{'AIAgent', 'SessionDB', 'GatewayRunner'}}:
            owners.append(type(frame.f_locals.get('self')).__name__)
sys.setprofile(trace)
try:
    runpy.run_module('hermes_cli.main', run_name='__main__')
finally:
    sys.setprofile(None)
    open({str(witness)!r}, 'w', encoding='utf-8').write(json.dumps(owners))
''')

    def chat(*args):
        result = subprocess.run([sys.executable, str(wrapper), "--cli", "chat", *args, "-Q"], env=env,
                                cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
        assert json.loads(witness.read_text(encoding="utf-8")) == [], witness.read_text(encoding="utf-8")
        sid = next((line.split("Session: ", 1)[1].strip() for line in result.stderr.splitlines()
                    if "Session: " in line), None)
        return result, sid

    title = "wave36 titled thread"
    log_path = tmp_path / "gateway.log"
    with log_path.open("w") as log:
        daemon = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=root, env=env,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            descriptor = {}
            while daemon.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, "identify")
                    if descriptor.get("state") == "ready":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            assert descriptor.get("state") == "ready", log_path.read_text(encoding="utf-8")

            missing, _ = chat("-c", title, "-q", "WS_SHARED absent")
            assert missing.returncode == 1 and f"No session found matching '{title}'" in missing.stderr, missing

            created, sid = chat("-c", title, "--create-if-missing", "-q", "WS_SHARED create")
            assert created.returncode == 0 and sid and "LOCAL_ACK_WS_SHARED" in created.stdout, created

            # Bot Mode's exact turn shape (tools.bot_relay.BOT_CHAT_TURN_ARGS): --in + --query-file
            # + --create-if-missing must resume the existing titled thread, never fork a second one.
            query_file = tmp_path / "dm.txt"
            query_file.write_text("WS_SHARED again", encoding="utf-8")
            again, again_sid = chat("--in", str(tmp_path), "-c", title, "--create-if-missing",
                                    "--query-file", str(query_file))
            assert again.returncode == 0 and again_sid == sid, again

            resumed, resumed_sid = chat("--resume", title, "-q", "WS_SHARED resumed")
            assert resumed.returncode == 0 and resumed_sid == sid, resumed
            # The whole thread accumulates on ONE session: the third turn sees both earlier turns.
            history = json.dumps(model_peer.requests[-1]["messages"])
            assert "WS_SHARED create" in history and "WS_SHARED again" in history
            assert len(model_peer.requests) == 3
            print("CLI_TITLE_RECEIPT=" + json.dumps({"session_id": sid, "daemon_pid": daemon.pid,
                                                     "missing": missing.stderr, "resumed": resumed.stdout}))
        finally:
            if daemon.poll() is None:
                daemon.send_signal(signal.SIGINT)
                try:
                    daemon.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait(timeout=5)

"""Native PTY clients against an ordinary daemon and loopback model only.

`hermes --cli chat` (interactive and -q/-Q) is main's HermesCLI again, its agent and session store the
gateway's; `-z` stays on the gateway transport client. The CLI process may read state.db (maintainer
ruling) but builds no AIAgent, no GatewayRunner and no SessionDB writer. The thin client's
`/approve <id>` detach/reconnect leg is retired: main's classic approval is the modal the A/B harness
step 03a_approval_modal compares against main.
"""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

import pytest

from hermes_cli.observability.shared_metrics_consent import OFFER_VERSION

from tests.gateway.test_normal_runtime_boot import control


@pytest.mark.platforms("linux")
def test_native_classic_fresh_resume_and_oneshot(tmp_path, model_peer, request):
    home = tmp_path / "state"
    home.mkdir(mode=0o700)
    user = tmp_path / "user"
    user.mkdir()
    cwd = tmp_path / "caller"
    cwd.mkdir()
    root = Path(__file__).resolve().parents[2]
    gateway_root = Path(request.config.getoption("--client-test-runtime-root", default=None) or root).resolve()
    model_url = f"http://127.0.0.1:{model_peer.server_port}/v1"
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False},
        "approvals": {"mode": "manual", "timeout": 60},
        "model": {"provider": "custom", "default": "local-wire-stub", "base_url": model_url},
        "auxiliary": {"title_generation": {"enabled": False}},
        # Answered at the current offer version, as tests/fakes/fake_llm_provider.write_hermes_home
        # does: main's interactive chat asks it first on a home that never answered.
        "telemetry": {"shared_metrics": {"enabled": False, "send": False, "offer_version": OFFER_VERSION}},
    }))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED="1",
               OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=model_url)
    # Instrument constructor calls without replacing runtime predicates or behavior.
    wrapper = tmp_path / "client.py"
    witness = tmp_path / "client-owner.json"
    wrapper.write_text('''import json, runpy, sys
from pathlib import Path
owners = []
def trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == '__init__':
        obj = frame.f_locals.get('self')
        name = type(obj).__name__
        if name in {'AIAgent', 'GatewayRunner'} or (name == 'SessionDB' and not frame.f_locals.get('read_only')):
            owners.append(name)
sys.setprofile(trace)
try:
    module = 'hermes_cli.main'
    if sys.argv[1:2] == ['--fixture-direct']:
        sys.argv.pop(1)
        module = 'cli'
    runpy.run_module(module, run_name='__main__')
finally:
    sys.setprofile(None)
    Path(sys.argv[0]).with_name('client-owner.json').write_text(json.dumps(owners))
''')
    # argv[0] is replaced by runpy, so the witness path is explicitly bound.
    wrapper.write_text(wrapper.read_text().replace("Path(sys.argv[0]).with_name('client-owner.json')", repr(str(witness)) + " and Path(" + repr(str(witness)) + ")"))
    tmux_name = "hermes-cli-test-" + uuid.uuid4().hex

    def tmux(*args):
        return subprocess.run(["tmux", "-L", tmux_name, *args], env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10, check=False)

    def screen():
        result = tmux("capture-pane", "-p", "-S", "-1000", "-t", "chat")
        return result.stdout

    def until(text, timeout=45):
        deadline = time.monotonic() + timeout
        output = ""
        while time.monotonic() < deadline:
            output = screen()
            if text in output:
                return output
            time.sleep(.1)
        raise AssertionError((text, output, log_path.read_text()))

    def launch(*args):
        command = shlex.join([sys.executable, str(wrapper), *args])
        result = tmux("new-session", "-d", "-s", "chat", "-x", "160", "-y", "45", "-c", str(cwd),
                      command + '; rc=$?; printf "\\nCLI_RC=%s\\n" "$rc"; sleep 120')
        assert result.returncode == 0, result.stderr

    log_path = tmp_path / "gateway.log"
    with log_path.open("w") as log:
        daemon = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=gateway_root,
                                  env={**env, "PYTHONPATH": str(gateway_root)},
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
            assert descriptor.get("state") == "ready", log_path.read_text()
            def type_line(text):
                # As the A/B harness types: main's prompt_toolkit drops an Enter in the same burst.
                tmux("send-keys", "-t", "chat", "-l", text)
                until(text, 10)
                time.sleep(1)  # health: allow HX013 -- main drops an Enter sent in the same burst as the text (probe_pty.py); the gap is the input timing
                tmux("send-keys", "-t", "chat", "Enter")

            launch("--cli", "chat")
            first = until("Welcome to Hermes Agent")
            type_line("WS_SHARED")
            fresh = until("LOCAL_ACK_WS_SHARED")
            type_line("/title")
            # Main's bare /title prints the full id (the banner shows its 8-character label).
            shown = until("Session ID: ")
            sid = next(line.split("Session ID: ", 1)[1].strip() for line in shown.splitlines() if "Session ID: " in line)
            type_line("/quit")
            detached = until("CLI_RC=0")
            assert json.loads(witness.read_text()) == [], witness.read_text()
            assert daemon.poll() is None
            tmux("kill-session", "-t", "chat")

            launch("--cli", "chat", "--resume", sid, "-q", "WS_SHARED resumed", "-Q")
            resumed = until("CLI_RC=0")
            assert "LOCAL_ACK_WS_SHARED" in resumed
            assert json.loads(witness.read_text()) == []
            tmux("kill-session", "-t", "chat")

            launch("-z", "WS_SHARED oneshot")
            oneshot = until("CLI_RC=0")
            assert "LOCAL_ACK_WS_SHARED" in oneshot
            assert json.loads(witness.read_text()) == []
            assert control(home, "identify")["instance_id"] == descriptor["instance_id"]
            assert len(model_peer.requests) == 3
            tmux("kill-session", "-t", "chat")
            launch("--fixture-direct", "-q", "WS_SHARED direct", "--oneshot", "--quiet")
            direct = until("CLI_RC=0")
            assert "LOCAL_ACK_WS_SHARED" in direct
            assert json.loads(witness.read_text()) == []
            tmux("kill-session", "-t", "chat")

            from tests.gateway.fixtures.authority_controls_peer import ModelPeer as ApprovalPeer
            model_peer.RequestHandlerClass = ApprovalPeer
            cwd_supported = "cannot preserve caller cwd" not in first
            cwd_receipt = tmp_path / "cwd-proof.txt"
            model_peer.command = "pwd > " + shlex.quote(str(cwd_receipt))
            launch("--cli", "chat", "-q", "Record the caller directory", "-Q", "--in", str(cwd),
                   "--model", "local-wire-stub", "--toolsets", "terminal")
            if cwd_supported:
                policy = until("CLI_RC=0")
                assert cwd_receipt.read_text().strip() == str(cwd)
                assert "APPROVAL_FINISHED" in policy
            else:
                policy = until("CLI_RC=1")
                assert "does not support --in" in policy
                assert not cwd_receipt.exists()
            assert json.loads(witness.read_text()) == []
            receipt = {"gateway_root": str(gateway_root), "caller_cwd_effect_verified": cwd_supported,
                       "policy": policy, "direct_cli": direct,
                       "session_id": sid, "daemon_pid": daemon.pid,
                       "same_instance": descriptor["instance_id"], "client_agent_db_owners": [],
                       "fresh": fresh, "detach": detached, "resume": resumed, "oneshot": oneshot}
            print("CLI_PTY_RECEIPT=" + json.dumps(receipt))
        finally:
            tmux("kill-server")
            if daemon.poll() is None:
                daemon.send_signal(signal.SIGINT)
                try:
                    daemon.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait(timeout=5)

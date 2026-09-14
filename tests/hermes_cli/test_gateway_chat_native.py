"""Native PTY clients against an ordinary daemon and loopback model only."""
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

from tests.gateway.test_normal_runtime_boot import control, model_peer  # noqa: F401


@pytest.mark.linux_only
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
        if type(obj).__name__ in {'AIAgent', 'SessionDB', 'GatewayRunner'}:
            owners.append(type(obj).__name__)
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
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)

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
            launch("--cli", "chat")
            first = until("You>")
            sid = next(line.split("Session: ", 1)[1].strip() for line in first.splitlines() if "Session: " in line)
            tmux("send-keys", "-t", "chat", "WS_SHARED", "Enter")
            fresh = until("LOCAL_ACK_WS_SHARED")
            tmux("send-keys", "-t", "chat", "/quit", "Enter")
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
            target = tmp_path / "owned-removal"
            target.mkdir()
            (target / "owned.txt").write_text("disposable")
            model_peer.command = "rm -r -- " + shlex.quote(str(target))
            model_peer.RequestHandlerClass = ApprovalPeer
            launch("--cli", "chat", "-q", "Remove the owned fixture")
            approval = until("/approve ")
            approval_sid = next(line.split("Session: ", 1)[1].strip() for line in approval.splitlines() if "Session: " in line)
            prompt_id = next(line.split("/approve ", 1)[1].split()[0] for line in approval.splitlines() if "/approve " in line)
            assert target.exists()
            tmux("send-keys", "-t", "chat", f"/approve {prompt_id} forged", "Enter")
            until("invalid_params")
            assert target.exists(), "Invalid choice must not resolve the real approval"
            tmux("send-keys", "-t", "chat", "/quit", "Enter")
            until("CLI_RC=0")
            assert target.exists()
            tmux("kill-session", "-t", "chat")
            launch("--cli", "chat", "--resume", approval_sid)
            until("/approve ")
            tmux("send-keys", "-t", "chat", f"/approve {prompt_id} once", "Enter")
            approved = until("APPROVAL_FINISHED")
            assert not target.exists(), "Real terminal effect must follow native consent"
            tmux("send-keys", "-t", "chat", "/quit", "Enter")
            until("CLI_RC=0")
            assert json.loads(witness.read_text()) == []
            tmux("kill-session", "-t", "chat")
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
                       "approval_detach_reconnect": approval_sid, "approval": approved,
                       "invalid_choice_no_effect": True, "real_terminal_effect": True,"session_id": sid, "daemon_pid": daemon.pid,
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

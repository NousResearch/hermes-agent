"""#51797: top-level one-shot isolation with a configured/explicit memory toolset.

The real CLI, gateway and agent call a loopback-only model. No real credentials.
"""
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, seed_offline_models_dev
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("flag,explicit_tools,managed", [
    (None, None, False),
    ("--ignore-rules", None, False),
    ("--ignore-rules", "memory,file", False),
    ("--ignore-rules", "memory,file", True),
    ("--safe-mode", "memory,file", False),
    ("--ignore-user-config", "memory,file", False),
])
def test_top_level_memory_isolation(tmp_path, flag, explicit_tools, managed):
    root = Path(__file__).resolve().parents[2]
    home, user, work = (tmp_path / name for name in ("state", "user", "work"))
    for path in (home, user, work):
        path.mkdir(mode=0o700)
    (work / "AGENTS.md").write_text("RULE_SENTINEL\n", encoding="utf-8")
    (home / "SOUL.md").write_text("SOUL_SENTINEL\n", encoding="utf-8")
    memories = home / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("MEMORY_SENTINEL\n", encoding="utf-8")
    (memories / "USER.md").write_text("USER_SENTINEL\n", encoding="utf-8")
    peer = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    config = {
        "gateway": {"multiplex_profiles": False, "managed_workers": managed},
        "model": {"provider": "custom", "default": "configured-model", "base_url": url},
        "agent": {"system_prompt": "CONFIG_SENTINEL", "environment_probe": False, "max_turns": 2},
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "platform_toolsets": {"cli": ["memory", "file"]},
        "auxiliary": {"title_generation": {"enabled": False}},
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
    }
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    seed_offline_models_dev(home)
    witness = tmp_path / "witness"
    witness.mkdir()
    (witness / "sitecustomize.py").write_text(
        "import ipaddress, sys\n"
        "def fence(event, args):\n"
        "    if event == 'socket.connect' and isinstance(args[1], tuple):\n"
        "        if not ipaddress.ip_address(args[1][0]).is_loopback:\n"
        "            raise OSError('test permits loopback networking only')\n"
        "sys.addaudithook(fence)\n", encoding="utf-8")
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"),
               PYTHONPATH=os.pathsep.join((str(witness), str(root))),
               OPENAI_API_KEY="loopback-fixture", OPENAI_BASE_URL=url)
    proc = None
    with (tmp_path / "gateway.log").open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.Popen([sys.executable, "-m", "gateway.run"], cwd=root, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            def ready():
                assert proc.poll() is None, (tmp_path / "gateway.log").read_text(encoding="utf-8")
                try:
                    return control(home, "identify").get("state") == "ready"
                except (OSError, ValueError):
                    return False
            wait_until(ready, timeout=180, interval=.1, what="isolated gateway ready")
            command = [sys.executable, "-m", "hermes_cli.main"]
            command += [flag] if flag else []
            command += ["--model", "explicit-fixture", "--provider", "custom"]
            command += ["--toolsets", explicit_tools] if explicit_tools else []
            command += ["-z", "MEMORY_ISOLATION_PROBE"]
            # Endpoint/key belong to chat; main still dispatches the top-level -z first.
            command += ["chat", "--base-url", url, "--api-key", "loopback-fixture"]
            result = subprocess.run(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=180, check=False)
            assert result.returncode == 0, result.stdout + result.stderr
            assert "RECOVERY_ACK_MEMORY_ISOLATION_PROBE" in result.stdout
            requests = [body for body in peer.requests
                        if any(row.get("role") == "user" and row.get("content") == "MEMORY_ISOLATION_PROBE"
                               for row in body.get("messages", []))
                        and not str(body["messages"][0].get("content", "")).startswith("You name chat sessions")]
            assert len(requests) == 1, peer.requests
            body = requests[0]
            prompt = json.dumps(body["messages"])
            isolated = flag in ("--ignore-rules", "--safe-mode")
            for sentinel in ("RULE_SENTINEL", "SOUL_SENTINEL", "MEMORY_SENTINEL", "USER_SENTINEL"):
                assert (sentinel in prompt) is not isolated, (sentinel, body)
            assert ("CONFIG_SENTINEL" in prompt) is (flag not in ("--safe-mode", "--ignore-user-config"))
            assert body["model"] == "explicit-fixture"
            names = {tool["function"]["name"] for tool in body.get("tools", [])}
            assert "read_file" in names
            if flag != "--safe-mode":
                assert "memory" in names  # isolation must not silently disable the tool
        finally:
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            peer.shutdown()
            peer.server_close()
            thread.join(timeout=5)

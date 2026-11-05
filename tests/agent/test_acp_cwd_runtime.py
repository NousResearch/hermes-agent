"""Regression for #54638: remote ACP cwd survives runtime resolution and delegation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PEER = '''import json, os, sys
for line in sys.stdin:
    req = json.loads(line)
    result = {}
    if req["method"] == "session/new":
        result = {"sessionId": "test", "models": {"availableModels": [
            {"modelId": req["params"]["cwd"], "name": "session cwd"},
            {"modelId": os.getcwd(), "name": "launcher cwd"}]}}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
'''
_DRIVER = '''
import json, os, shutil, socket, sys
from pathlib import Path

def refuse_network(*args, **kwargs):
    raise AssertionError("The ACP cwd peer uses stdio only")
socket.getaddrinfo = refuse_network

from run_agent import AIAgent
from hermes_cli.runtime_provider import resolve_runtime_provider

options = json.loads(sys.argv[1])
if options.get("local_cwd"):
    from agent.runtime_cwd import set_session_cwd
    set_session_cwd(options["local_cwd"])
if "profile_cwd" in options:
    from hermes_cli.profiles import get_profile_dir
    profile = get_profile_dir("coder")
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(json.dumps({"model": {
        "provider": "test-remote-acp", "default": "acp", "acp_cwd": options["profile_cwd"]}}))
    shutil.copytree(Path(os.environ["HERMES_HOME"]) / "plugins", profile / "plugins")
def make_agent(**kwargs):
    return AIAgent(model="acp", quiet_mode=True, enabled_toolsets=[],
                   skip_context_files=True, skip_memory=True, skip_background_review=True, **kwargs)

runtime = resolve_runtime_provider(requested="test-remote-acp")
kwargs = {key: runtime.get(key) for key in
          ("provider", "api_key", "base_url", "api_mode", "command", "args", "acp_cwd")}
if not options.get("resolved", False):
    kwargs = {"provider": "test-remote-acp"}
if options.get("explicit"):
    kwargs["acp_cwd"] = options["expected"]
parent = make_agent(**kwargs)
try:
    if "route" not in options:
        observed = parent.client.list_models(timeout_seconds=10)
    else:
        from tools.delegate_tool import delegate_task
        observed = []
        def run_child(child, *args, **kwargs):
            if options["route"] == "http":
                assert child.acp_command is None and child.acp_cwd is None
                observed.append("http")
            else:
                observed.extend(child.client.list_models(timeout_seconds=10))
            return {"final_response": "verified", "messages": []}
        AIAgent.run_conversation = run_child
        result = json.loads(delegate_task(goal="verify cwd", parent_agent=parent, background=False))
        assert observed, result
    print(json.dumps(observed))
finally:
    parent.close()
'''


@pytest.fixture
def acp_config(tmp_path):
    home = tmp_path / "hermes"
    plugin = home / "plugins" / "model-providers" / "test-remote-acp"
    plugin.mkdir(parents=True)
    peer = tmp_path / "peer.py"
    peer.write_text(_PEER, encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        "name: test-remote-acp\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from agent.acp_client import ACPClient\n"
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "class RemoteProfile(ProviderProfile):\n"
        "    def create_client(self, **kwargs):\n"
        "        return ACPClient(**kwargs)\n"
        "register_provider(RemoteProfile(name='test-remote-acp', auth_type='external_process',\n"
        f"    base_url='acp://test', process_command={sys.executable!r}, process_args=({str(peer)!r},)))\n",
        encoding="utf-8")
    remote = str(tmp_path / "remote-only" / "workspace")
    config = {"model": {"provider": "test-remote-acp", "default": "acp", "acp_cwd": remote},
              "agent": {"max_turns": 1}, "delegation": {"max_iterations": 1}}
    return home, config


def _run(home, config, tmp_path, **options):
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home), "PYTHONPATH": str(_REPO)}
    env.pop("HERMES_IGNORE_USER_CONFIG", None)
    proc = subprocess.run([sys.executable, "-c", _DRIVER, json.dumps(options)], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=90, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:] + proc.stdout[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("cwd_source", ["configured", "explicit", "default"])
def test_session_cwd_preserves_remote_paths_and_local_defaults(acp_config, tmp_path, resolved, cwd_source):
    home, config = acp_config
    explicit = cwd_source == "explicit"
    expected = str(tmp_path / "explicit-remote" / "workspace") if explicit else config["model"]["acp_cwd"]
    if cwd_source == "default":
        config["model"].pop("acp_cwd")
        workspace = tmp_path / "local-workspace"
        workspace.mkdir()
        assert _run(home, config, tmp_path, resolved=resolved, local_cwd=str(workspace)) == [str(workspace)]
        return
    assert not Path(expected).exists()
    assert _run(home, config, tmp_path, resolved=resolved, explicit=explicit, expected=expected) == [expected, str(tmp_path)]


@pytest.mark.parametrize("route", ["inherit", "cwd-only", "pinned", "http", "profile", "profile-default"])
def test_delegation_config_reaches_real_child_session(acp_config, tmp_path, route):
    home, config = acp_config
    expected = config["model"]["acp_cwd"]
    if route in {"cwd-only", "pinned"}:
        expected += "/child"
        config["delegation"]["acp_cwd"] = expected
    if route == "pinned":
        config["delegation"]["provider"] = "test-remote-acp"
    if route == "http":
        config["delegation"].update(base_url="http://localhost:12345/v1", api_key="local-test")
    options = {}
    if route.startswith("profile"):
        config["delegation"]["profile"] = "coder"
        expected += "/profile"
        options["profile_cwd"] = expected if route == "profile" else None
    observed = _run(home, config, tmp_path, route=route, **options)
    if route == "profile-default":
        assert observed == [str(tmp_path)]
    else:
        assert observed == (["http"] if route == "http" else [expected, str(tmp_path)])

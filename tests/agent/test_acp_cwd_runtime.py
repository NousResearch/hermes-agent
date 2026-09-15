"""Regression for #54638: remote ACP cwd survives real runtime and delegation paths."""
import json
import shlex
import sys

import pytest
import yaml


@pytest.fixture
def acp_config(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    # A real stdio peer, but no SSH credentials or provider calls. Echo the received
    # session cwd as a model ID so the wire assertion cannot pass on client state alone.
    peer = tmp_path / "peer.py"
    peer.write_text('''import json, sys
for line in sys.stdin:
    req = json.loads(line)
    result = {}
    if req["method"] == "session/new":
        result = {"sessionId": "test", "models": {"currentModelId": req["params"]["cwd"], "availableModels": [{"modelId": req["params"]["cwd"], "name": "cwd"}]}}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
''', encoding="utf-8")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", sys.executable)
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", shlex.quote(str(peer)))
    remote = str(tmp_path / "remote-only" / "workspace")
    config = {"model": {"provider": "copilot-acp", "default": "copilot-acp", "acp_cwd": remote},
              "agent": {"max_turns": 1}, "delegation": {"max_iterations": 1}}
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, home


def _agent(**kwargs):
    from run_agent import AIAgent
    return AIAgent(model="copilot-acp", quiet_mode=True, enabled_toolsets=[],
                   skip_context_files=True, skip_memory=True, skip_background_review=True, **kwargs)


@pytest.mark.parametrize("resolved", [True, False])
def test_config_cwd_reaches_real_acp_session(acp_config, resolved):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    config, _ = acp_config
    runtime = resolve_runtime_provider(requested="copilot-acp")
    kwargs = {key: runtime.get(key) for key in
              ("provider", "api_key", "base_url", "api_mode", "command", "args", "acp_cwd")} if resolved else {"provider": "copilot-acp"}
    agent = _agent(**kwargs)
    try:
        assert agent.client.list_models(timeout_seconds=5) == [config["model"]["acp_cwd"]]
        assert agent.acp_cwd == config["model"]["acp_cwd"]
    finally:
        agent.close()


@pytest.mark.parametrize("route", ["inherit", "cwd-only", "pinned", "http"])
def test_delegation_config_reaches_real_child_client(acp_config, monkeypatch, route):
    from tools.delegate_tool import delegate_task
    from run_agent import AIAgent
    config, home = acp_config
    parent = _agent(provider="copilot-acp")
    expected = config["model"]["acp_cwd"]
    if route in {"cwd-only", "pinned"}:
        expected += "/child"
        config["delegation"]["acp_cwd"] = expected
    if route == "pinned":
        config["delegation"]["provider"] = "copilot-acp"
    if route == "http":
        config["delegation"].update(base_url="http://localhost:12345/v1", api_key="local-test")
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    observed = []

    def run_child(child, *args, **kwargs):
        if route == "http":
            assert child.acp_cwd is None
            assert child.acp_command is None
            observed.append("http")
        else:
            observed.extend(child.client.list_models(timeout_seconds=5))
            assert child.acp_cwd == expected
        return {"final_response": "verified", "messages": []}

    monkeypatch.setattr(AIAgent, "run_conversation", run_child)
    try:
        result = json.loads(delegate_task(goal="verify cwd", parent_agent=parent, background=False))
        assert observed == (["http"] if route == "http" else [expected]), result
    finally:
        parent.close()

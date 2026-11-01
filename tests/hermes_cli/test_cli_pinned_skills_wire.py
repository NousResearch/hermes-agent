"""Single-profile CLI preload -> real agent -> SDK JSON, with no network."""
import json
import socket

import httpx
import openai
import pytest


@pytest.mark.parametrize("unrelated", [False, True])
def test_cli_pinned_skills_wire(tmp_path, monkeypatch, unrelated):
    blocked = []

    def deny(*args, **kwargs):
        import traceback
        blocked.append("".join(traceback.format_stack(limit=25)))
        raise AssertionError("Network is forbidden in this acceptance test")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    home = tmp_path / "profile"
    skill = home / "skills" / "pinned-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    body = "PINNED BEGIN\nA complete synthetic procedure.\nPINNED END"
    skill.write_text("---\nname: pinned-skill\ndescription: Synthetic.\n---\n\n" + body + "\n")
    config = {
        "model": {"default": "synthetic-model", "provider": "custom", "base_url": "https://capture.invalid/v1"},
        "skills": {"auto_load": ["pinned-skill"]},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": False},
        "display": {"streaming": False},
        "agent": {}, "terminal": {"env_type": "local"},
    }
    import yaml
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    import cli
    import model_tools
    import tools.skills_tool
    import agent.process_bootstrap
    import agent.model_metadata
    monkeypatch.setattr(agent.model_metadata, "get_model_context_length", lambda *a, **k: 128000)
    import agent.context_compressor
    monkeypatch.setattr(agent.context_compressor, "get_model_context_length", lambda *a, **k: 128000)
    import agent.title_generator
    monkeypatch.setattr(agent.title_generator, "_auto_title_enabled", lambda: False)
    monkeypatch.setattr(cli, "CLI_CONFIG", config)
    monkeypatch.setattr(tools.skills_tool, "SKILLS_DIR", home / "skills")
    tools = ([{"type": "function", "function": {"name": "terminal", "description": "Synthetic unused capability", "parameters": {"type": "object", "properties": {}}}}] if unrelated else [])
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: tools)
    monkeypatch.setattr(cli, "get_tool_definitions", lambda *a, **k: tools)
    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    for name in ("_install_tool_callbacks", "_ensure_tirith_security"):
        monkeypatch.setattr(cli.HermesCLI, name, lambda self: None)
    monkeypatch.setattr(cli.HermesCLI, "_ensure_runtime_credentials", lambda self: True)
    import hermes_cli.mcp_startup
    monkeypatch.setattr(hermes_cli.mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: None)
    captured = []

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        if payload.get("stream"):
            chunk = {"id": "synthetic", "object": "chat.completion.chunk", "created": 0, "model": "synthetic-model", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Synthetic answer."}, "finish_reason": "stop"}]}
            return httpx.Response(200, content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "synthetic", "object": "chat.completion", "created": 0, "model": "synthetic-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthetic answer."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    client = openai.OpenAI(api_key="synthetic-not-a-secret", base_url="https://capture.invalid/v1", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond), trust_env=False))
    monkeypatch.setattr(agent.process_bootstrap, "OpenAI", lambda **kwargs: client)
    runtime = {"provider": "custom", "requested_provider": "custom", "api_mode": "chat_completions", "api_key": "synthetic-not-a-secret", "base_url": "https://capture.invalid/v1"}
    obj = cli._build_cli_from_args("synthetic-model", "terminal" if unrelated else "none", "custom", None, "synthetic-not-a-secret", "https://capture.invalid/v1", 2, None, False, True, None, False, False, False, "pinned-skill")
    obj.streaming_enabled = False
    assert obj._init_agent(runtime_override=runtime)
    agent = obj.agent
    expected = {"terminal"} if unrelated else set()
    assert agent.valid_tool_names == expected
    for index in range(2):
        result = agent.run_conversation(user_message="Synthetic query", conversation_history=[])
        assert result.get("final_response") == "Synthetic answer.", result
        if index == 0:
            skill.write_text("MUTATED SYNTHETIC BODY")
            config["skills"]["auto_load"] = []
            (home / "config.yaml").write_text(yaml.safe_dump(config))
            agent._cached_system_prompt = None
    assert len(captured) == 2
    prefixes = []
    for request in captured:
        messages = request["messages"]
        full = json.dumps(messages)
        assert full.count("PINNED BEGIN") == 1
        assert any(body in str(m["content"]) for m in messages)
        assert "MUTATED SYNTHETIC BODY" not in full
        assert {t["function"]["name"] for t in request.get("tools", [])} == expected
        prefixes.append([m for m in messages if m["role"] in ("system", "developer")])
    assert prefixes[0] == prefixes[1]
    assert any(body in str(m["content"]) for m in prefixes[0])
    assert not blocked, "\n".join(blocked)
    assert agent.valid_tool_names == expected
    assert obj.preloaded_skills == ["pinned-skill"]
    client.close()

"""Real request-dump path preserves JSON while redacting corrupting values."""

import json


def test_request_dump_corrupting_secret_is_parseable_and_preserves_safe_fields(
    monkeypatch, tmp_path
):
    import run_agent

    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    agent = run_agent.AIAgent(
        model="gpt-4o",
        base_url="http://127.0.0.1:9208/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.logs_dir = tmp_path
    secret = "DB_PASSWORD=pw1longer" + "value99"

    path = agent._dump_api_request_debug(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": secret}],
            "metadata": {"safe": "keep", "note": f"export {secret}"},
        },
        reason="preflight",
    )

    assert path is not None and path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request"]["body"]["metadata"]["safe"] == "keep"
    assert "pw1longervalue99" not in path.read_text(encoding="utf-8")


def test_request_dump_list_held_env_secret_keeps_sibling_fields(monkeypatch, tmp_path):
    import run_agent

    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    agent = run_agent.AIAgent(
        model="gpt-4o",
        base_url="http://127.0.0.1:9208/v1",
        api_key="test-key",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.logs_dir = tmp_path
    secret = "DB_PASSWORD=pw1longer" + "value99"

    path = agent._dump_api_request_debug(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"safe": "keep", "env": [secret]},
        },
        reason="preflight",
    )

    assert path is not None and path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = payload["request"]["body"]
    assert body["metadata"]["safe"] == "keep"
    assert body["messages"][0]["content"] == "hi"
    assert "pw1longervalue99" not in path.read_text(encoding="utf-8")

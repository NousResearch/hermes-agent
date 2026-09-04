from gateway.run import GatewayRunner


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._service_tier = None
    runner._pending_one_shot_model_overrides = {}
    return runner


def test_gateway_route_ordinary_message_stays_on_mini():
    runner = _make_runner()
    route = runner._resolve_turn_agent_config(
        "Reply exactly: hello",
        "gpt-5.4-mini",
        {
            "provider": "openai-api",
            "base_url": "http://localhost:20128/v1",
            "api_key": "redacted",
            "api_mode": "chat_completions",
        },
    )
    assert route["model"] == "gpt-5.4-mini"


def test_gateway_route_hard_complex_message_uses_gpt54():
    runner = _make_runner()
    route = runner._resolve_turn_agent_config(
        "architecture and multi-file refactor across src/app.py, src/api.py, src/db.py",
        "gpt-5.4-mini",
        {
            "provider": "openai-api",
            "base_url": "http://localhost:20128/v1",
            "api_key": "redacted",
            "api_mode": "chat_completions",
        },
    )
    assert route["model"] == "gpt-5.4"


def test_pending_one_shot_override_wins_for_ordinary_message():
    runner = _make_runner()
    runner._pending_one_shot_model_overrides["telegram:chat-1"] = {"model": "gpt-5.4"}
    route = runner._resolve_turn_agent_config(
        "Reply exactly: hello",
        "gpt-5.4-mini",
        {
            "model": "gpt-5.4",
            "provider": "openai-api",
            "base_url": "http://localhost:20128/v1",
            "api_key": "redacted",
            "api_mode": "chat_completions",
        },
    )
    assert route["model"] == "gpt-5.4"


def test_provider_base_url_and_api_key_are_unchanged():
    runner = _make_runner()
    runtime_kwargs = {
        "provider": "openai-api",
        "base_url": "http://localhost:20128/v1",
        "api_key": "redacted",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
    }
    route = runner._resolve_turn_agent_config(
        "architecture and multi-file refactor across src/app.py, src/api.py, src/db.py",
        "gpt-5.4-mini",
        runtime_kwargs,
    )
    assert route["model"] == "gpt-5.4"
    assert route["runtime"]["provider"] == runtime_kwargs["provider"]
    assert route["runtime"]["base_url"] == runtime_kwargs["base_url"]
    assert route["runtime"]["api_key"] == runtime_kwargs["api_key"]

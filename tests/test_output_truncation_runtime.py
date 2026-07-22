from types import SimpleNamespace


def test_custom_provider_normalizer_preserves_output_cap_fields():
    from hermes_cli.config import get_compatible_custom_providers

    cfg = {
        "custom_providers": [
            {
                "name": "hyperspace-responses",
                "base_url": "http://localhost:6658/openai/v1",
                "api_key": "test-key",
                "model": "gpt-5.5",
                "api_mode": "codex_responses",
                "max_output_tokens": 100000,
            }
        ]
    }

    providers = get_compatible_custom_providers(cfg)

    assert providers[0]["max_output_tokens"] == 100000


def test_cli_runtime_output_cap_reaches_agent_constructor(monkeypatch):
    from tests.cli.test_cli_provider_resolution import _import_cli

    cli = _import_cli()
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._print_fn = None

    monkeypatch.setattr("hermes_cli.mcp_startup.wait_for_mcp_discovery", lambda: None)
    monkeypatch.setattr(cli, "AIAgent", FakeAgent)
    monkeypatch.setattr(cli.HermesCLI, "_ensure_runtime_credentials", lambda self: True)

    shell = cli.HermesCLI(model="gpt-5.5", compact=True, max_turns=1)
    shell.agent = None
    shell.api_key = "test-key"
    shell.base_url = "http://localhost:6658/openai/v1"
    shell.provider = "custom"
    shell.api_mode = "codex_responses"
    shell.max_tokens = None

    assert shell._init_agent(
        model_override="gpt-5.5",
        runtime_override={
            "api_key": "test-key",
            "base_url": "http://localhost:6658/openai/v1",
            "provider": "custom",
            "api_mode": "codex_responses",
            "max_output_tokens": 100000,
        },
    ) is True

    assert captured["api_mode"] == "codex_responses"
    assert captured["max_tokens"] == 100000


def test_turn_route_signature_includes_runtime_output_cap():
    from tests.cli.test_cli_provider_resolution import _import_cli

    cli = _import_cli()
    shell = cli.HermesCLI(model="gpt-5.5", compact=True, max_turns=1)
    shell.api_key = "test-key"
    shell.base_url = "http://localhost:6658/openai/v1"
    shell.provider = "custom"
    shell.api_mode = "codex_responses"
    shell.acp_command = None
    shell.acp_args = []
    shell._credential_pool = None
    shell.max_tokens = 100000
    shell.service_tier = None

    route = shell._resolve_turn_agent_config("hi")

    assert route["runtime"]["max_tokens"] == 100000
    assert route["signature"][-1] == 100000


def test_truncated_tool_retry_uses_configured_or_requested_cap():
    from agent.conversation_loop import _next_truncated_tool_call_cap

    agent = SimpleNamespace(max_tokens=100000)
    assert _next_truncated_tool_call_cap(agent, {"max_completion_tokens": 4096}, 1) == 100000

    agent = SimpleNamespace(max_tokens=32000)
    assert _next_truncated_tool_call_cap(agent, {"max_completion_tokens": 100000}, 1) == 100000

    agent = SimpleNamespace(max_tokens=None)
    assert _next_truncated_tool_call_cap(agent, {}, 1) == 32768

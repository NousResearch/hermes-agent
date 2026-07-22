import sys
import types

from hermes_cli import mcp_startup, oneshot


def test_oneshot_waits_for_mcp_discovery_before_agent_tool_snapshot(monkeypatch):
    events = []

    class FakeAgent:
        def __init__(self, **_kwargs):
            events.append("agent")

        def chat(self, prompt):
            assert prompt == "use mcp"
            return "done"

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", lambda: events.append("wait"))
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "test-model", "provider": "test-provider"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test",
            "base_url": "https://example.invalid",
            "provider": "test-provider",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_args: {"aivex-hq"})

    assert oneshot._run_agent("use mcp") == "done"
    assert events == ["wait", "agent"]

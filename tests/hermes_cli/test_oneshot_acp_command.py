"""external_process providers (opencode-cli, copilot-acp, …) resolve ACP command/args in the
runtime dict; ``hermes -z`` must hand them to ``AIAgent`` or chat falls back to bare ``copilot``."""

import hermes_cli.oneshot as oneshot_mod


def test_run_agent_passes_acp_command_and_args_from_runtime(monkeypatch):
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __setattr__(self, name, _value):
            pass

        def run_conversation(self, _prompt, conversation_history=None):
            return {"final_response": "pong", "session_id": "s"}

        def close(self):
            pass

    runtime = {
        "api_key": "opencode-cli",
        "base_url": "acp://opencode",
        "provider": "opencode-cli",
        "requested_provider": "opencode-cli",
        "api_mode": "chat_completions",
        "command": "/usr/bin/opencode",
        "args": ["acp"],
        "credential_pool": None,
    }
    cfg = {"model": {"default": "opencode/mimo-v2.6-flash-free", "provider": "opencode-cli"}}

    monkeypatch.setenv("HERMES_HOME", str(monkeypatch.tmp_path if hasattr(monkeypatch, "tmp_path") else "/tmp"))
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_kw: dict(runtime)
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: [])
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None
    )
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

    text, _ = oneshot_mod._run_agent("Health check: reply pong.")

    assert text == "pong"
    assert captured["acp_command"] == "/usr/bin/opencode"
    assert captured["acp_args"] == ["acp"]
    assert captured["provider"] == "opencode-cli"

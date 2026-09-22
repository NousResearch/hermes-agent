from types import SimpleNamespace

import pytest

from hermes_cli import main as main_mod
from hermes_cli import oneshot


def test_none_is_an_explicit_empty_toolset():
    assert oneshot._validate_explicit_toolsets("none") == ([], None)
    toolsets, error = oneshot._validate_explicit_toolsets("none,terminal")
    assert toolsets is None
    assert "cannot be combined" in error


def test_run_oneshot_forwards_ignore_rules_and_empty_tools(monkeypatch):
    captured = {}

    def fake_run_agent(prompt, **kwargs):
        captured.update(kwargs, prompt=prompt)
        return "ok", {"final_response": "ok", "completed": True}

    monkeypatch.setattr(oneshot, "_run_agent", fake_run_agent)
    assert oneshot.run_oneshot("hello", toolsets="none", ignore_rules=True) == 0
    assert captured["toolsets"] == []
    assert captured["use_config_toolsets"] is False
    assert captured["ignore_rules"] is True


def test_top_level_oneshot_forwards_ignore_rules(monkeypatch):
    captured = {}

    def fake_exit(prompt, **kwargs):
        captured.update(kwargs, prompt=prompt)
        raise RuntimeError("stop")

    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(main_mod, "_resolve_chat_session_args", lambda _args, use_tui: None)
    monkeypatch.setattr(main_mod, "_run_and_exit_oneshot", fake_exit)
    args = SimpleNamespace(
        oneshot="hello", model=None, provider=None, toolsets="none", skills=None,
        usage_file=None, resume=None, reasoning="low", ignore_rules=True, safe_mode=False,
    )
    with pytest.raises(RuntimeError, match="stop"):
        main_mod._run_oneshot_from_args(args)
    assert captured["ignore_rules"] is True


def test_agent_gets_empty_tools_and_rule_memory_skips(monkeypatch):
    captured = {}
    mcp_discovery_called = False

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = "s"

        def run_conversation(self, _prompt, **_kwargs):
            return {"final_response": "ok", "completed": True}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    def unexpected_mcp(**_kwargs):
        nonlocal mcp_discovery_called
        mcp_discovery_called = True

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_with_fallback",
        lambda *_args, **_kwargs: ({
            "api_key": "key", "base_url": "https://example.test/v1", "provider": "custom",
            "requested_provider": "custom", "api_mode": "chat_completions",
            "credential_pool": None,
        }, None),
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", unexpected_mcp)
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    response, _ = oneshot._run_agent(
        "hello", model="m", provider="custom", toolsets=[], use_config_toolsets=False,
        ignore_rules=True,
    )

    assert response == "ok"
    assert captured["enabled_toolsets"] == []
    assert captured["skip_context_files"] is True
    assert captured["skip_memory"] is True
    assert mcp_discovery_called is False

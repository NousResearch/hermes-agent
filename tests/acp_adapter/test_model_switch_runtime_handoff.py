"""ACP runtime handoff and strict commit tests for the canonical model cut."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from acp_adapter import model_switch_resolution
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState


def test_rebuilt_agent_uses_validated_runtime_without_second_acquisition(monkeypatch):
    seen = []
    pool = object()

    class FakeAgent:
        def __init__(self, **kw):
            seen.append(kw)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "model": {"provider": "openrouter", "default": "legacy"},
    })
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: pytest.fail("second credential or route resolution"),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None,
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *_a: None)

    manager = SessionManager(db=None)
    result = manager._make_agent(
        session_id="s", cwd=".", model="openai/gpt-5.6", requested_provider="openrouter",
        base_url="https://openrouter.ai/api/v1", api_mode="codex_responses",
        resolved_runtime={
            "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "codex_responses", "api_key": "scoped-credential",
            "credential_pool": pool, "runtime_kind": "http",
        },
        enabled_toolsets=["hermes-acp", "mcp-live"], disabled_toolsets=["browser"],
    )
    assert isinstance(result, FakeAgent)
    assert len(seen) == 1
    args = seen[0]
    assert args["model"] == "openai/gpt-5.6"
    assert args["provider"] == "openrouter"
    assert args["api_mode"] == "codex_responses"
    assert args["credential_pool"] is pool
    assert args["enabled_toolsets"] == ["hermes-acp", "mcp-live"]
    assert args["disabled_toolsets"] == ["browser"]


def test_strict_session_persistence_propagates_failed_metadata_writes(monkeypatch):
    class BrokenDB:
        def get_session(self, _sid):
            return object()

        def update_session_meta(self, *_args):
            raise OSError("cannot persist selected model")

    manager = SessionManager()
    agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com",
                            api_mode="anthropic_messages", api_key="never-persist")
    manager._sessions["s"] = SessionState("s", agent, cwd=".", model="model-a")
    monkeypatch.setattr(manager, "_get_db", lambda: BrokenDB())
    manager.save_session("s")  # prior best-effort behavior for ordinary writes
    with pytest.raises(OSError, match="cannot persist"):
        manager.save_session("s", strict=True)


def test_failed_session_commit_restores_live_agent_and_model(monkeypatch):
    previous = SimpleNamespace(provider="anthropic", model="old-model")
    state = SessionState("s", previous, cwd=".", model="old-model")

    class Manager:
        def _make_agent(self, **kwargs):
            return SimpleNamespace(provider=kwargs["requested_provider"], model=kwargs["model"])

        def save_session(self, sid, *, strict=False):
            assert strict and sid == "s"
            raise OSError("commit failed")

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        model_switch_resolution, "resolve_acp_model_switch",
        lambda **_kw: model_switch_resolution.AcpModelRoute(
            model="new-model", provider="anthropic", base_url="https://api.anthropic.com",
            api_mode="anthropic_messages", runtime={"api_key": "hidden"},
        ),
    )
    server = HermesACPAgent(session_manager=Manager())
    with pytest.raises(OSError, match="commit failed"):
        server._switch_model(state, "anthropic:new-model")
    assert state.agent is previous and state.model == "old-model"


def test_missing_credentials_are_invalid_model_selection(monkeypatch):
    monkeypatch.setattr(
        model_switch_resolution, "_acquire",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("No credentials stored for anthropic")),
    )
    with pytest.raises(ValueError, match="No credentials"):
        model_switch_resolution.resolve_acp_model_switch(
            config={}, raw_model="anthropic:claude-sonnet-5",
            current_provider="openrouter", current_model="old-model",
        )

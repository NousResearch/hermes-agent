import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_profile_agents_are_injected_before_project_context(tmp_path, monkeypatch):
    from agent import prompt_builder

    profile = tmp_path / "profile"
    project = tmp_path / "project"
    profile.mkdir()
    project.mkdir()
    (profile / "AGENTS.md").write_text("PROFILE_RULE", encoding="utf-8")
    (project / "AGENTS.md").write_text("PROJECT_RULE", encoding="utf-8")
    monkeypatch.setattr(prompt_builder, "get_hermes_home", lambda: profile)

    rendered = prompt_builder.build_context_files_prompt(
        cwd=str(project), skip_soul=True
    )

    assert "PROFILE_RULE" in rendered
    assert "PROJECT_RULE" in rendered
    assert rendered.index("PROFILE_RULE") < rendered.index("PROJECT_RULE")


def test_profile_agents_are_not_duplicated_when_cwd_is_profile(tmp_path, monkeypatch):
    from agent import prompt_builder

    (tmp_path / "AGENTS.md").write_text("ONLY_ONCE", encoding="utf-8")
    monkeypatch.setattr(prompt_builder, "get_hermes_home", lambda: tmp_path)

    rendered = prompt_builder.build_context_files_prompt(
        cwd=str(tmp_path), skip_soul=True
    )

    assert rendered.count("ONLY_ONCE") == 1


def test_profile_relationship_loader_uses_only_distilled_summary(tmp_path, monkeypatch):
    from agent import prompt_builder

    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "RELATIONSHIP.md").write_text(
        "# Moss Relationship\n\nDISTILLED_RELATIONSHIP", encoding="utf-8"
    )
    (memories / "BOOKMARKS.md").write_text(
        "UNCONSOLIDATED_BOOKMARK", encoding="utf-8"
    )
    awareness = tmp_path / "awareness" / "diaries"
    awareness.mkdir(parents=True)
    (awareness / "entry.md").write_text("PRIVATE_DIARY_EVIDENCE", encoding="utf-8")
    monkeypatch.setattr(prompt_builder, "get_hermes_home", lambda: tmp_path)

    rendered = prompt_builder.load_relationship_md()

    assert "DISTILLED_RELATIONSHIP" in rendered
    assert "UNCONSOLIDATED_BOOKMARK" not in rendered
    assert "PRIVATE_DIARY_EVIDENCE" not in rendered


def _runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._active_session_leases = {}
    runner._persist_active_agents = MagicMock()
    return runner


class _ImmediateThread:
    def __init__(self, *, target, args, **_kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


def test_running_cache_evict_defers_and_then_closes_memory_provider():
    runner = _runner()
    agent = MagicMock()
    runner._agent_cache["session"] = (agent, "sig")
    runner._running_agents["session"] = agent
    runner._running_agents_ts["session"] = 1.0

    runner._evict_cached_agent("session")

    assert runner._pending_soft_release_agents["session"] is agent
    agent.release_memory_provider.assert_not_called()

    with patch("gateway.run.threading.Thread", _ImmediateThread):
        assert runner._release_running_agent_state("session") is True

    agent.release_memory_provider.assert_called_once_with()
    agent.release_clients.assert_called_once_with()


def test_release_memory_provider_is_idempotent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    manager = MagicMock()
    agent._memory_manager = manager

    agent.release_memory_provider()
    agent.release_memory_provider()

    manager.shutdown_all.assert_called_once_with()
    assert agent._memory_manager is None


def test_auxiliary_api_key_provider_prefers_credential_pool():
    from agent import auxiliary_client

    pooled = SimpleNamespace(
        runtime_api_key="pool-key",
        runtime_base_url="https://coding.example/v1",
        provider="alibaba",
    )
    client = MagicMock()
    with patch.object(auxiliary_client, "_select_pool_entry", return_value=(True, pooled)), \
         patch.object(auxiliary_client, "_create_openai_client", return_value=client) as create, \
         patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={"api_key": "legacy-key"}):
        resolved, model = auxiliary_client.resolve_provider_client(
            "alibaba", model="qwen3.5-plus"
        )

    assert resolved is client
    assert model == "qwen3.5-plus"
    assert create.call_args.kwargs["api_key"] == "pool-key"
    assert str(create.call_args.kwargs["base_url"]).startswith("https://coding.example")


def test_named_custom_provider_uses_pool_when_config_has_no_inline_key():
    from agent import auxiliary_client

    pooled = SimpleNamespace(
        runtime_api_key="pool-key",
        runtime_base_url="https://coding.example/v1",
        provider="alibaba",
    )
    client = MagicMock()
    custom = {
        "name": "alibaba",
        "base_url": "https://coding.example/v1",
        "api_key": "",
        "model": "qwen3.5-plus",
    }
    with patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=custom), \
         patch.object(auxiliary_client, "_select_pool_entry", return_value=(True, pooled)), \
         patch.object(auxiliary_client, "_create_openai_client", return_value=client) as create:
        resolved, model = auxiliary_client.resolve_provider_client(
            "alibaba", model="qwen3.5-plus"
        )

    assert resolved is client
    assert model == "qwen3.5-plus"
    assert create.call_args.kwargs["api_key"] == "pool-key"


def test_builtin_provider_honors_profile_base_url_override():
    from agent import auxiliary_client

    pooled = SimpleNamespace(runtime_api_key="pool-key", provider="alibaba")
    client = MagicMock()
    with patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=None), \
         patch.object(auxiliary_client, "_select_pool_entry", return_value=(True, pooled)), \
         patch.object(auxiliary_client, "_create_openai_client", return_value=client) as create, \
         patch("hermes_cli.config.load_config", return_value={
             "providers": {"alibaba": {"base_url": "https://coding.example/v1"}}
         }):
        resolved, model = auxiliary_client.resolve_provider_client(
            "alibaba", model="qwen3.5-plus"
        )

    assert resolved is client
    assert model == "qwen3.5-plus"
    assert str(create.call_args.kwargs["base_url"]).startswith("https://coding.example")

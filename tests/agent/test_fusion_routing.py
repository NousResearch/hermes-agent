"""Behavior-contract tests for Fusion compaction-boundary routing.

Fusion routes only at cache-invalidating context-compaction boundaries. The
summarizer LLM supplies a structured Work Complexity verdict; the conversation
loop consumes and clears it before optionally switching through the existing
switch_model helper.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.system_prompt import build_system_prompt_parts
from hermes_cli.config import DEFAULT_CONFIG


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Implement the clear mechanical edit."},
        {"role": "assistant", "content": "I will inspect the files."},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "Ran tests and found lint failures."},
        {"role": "user", "content": "Fix them."},
    ]


def _make_compressor(owner: Any, *, routing: bool) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="frontier-model",
            provider="frontier-provider",
            api_mode="chat_completions",
            quiet_mode=True,
        )
    # Attributes are set here so the test remains about behavior, not a
    # constructor shape. AIAgent.__init__ will populate these in production.
    compressor._fusion_compaction_routing = routing
    compressor._fusion_verdict_owner = owner
    return compressor


def _minimal_prompt_agent(*, fusion_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        _fusion_enabled=fusion_enabled,
        load_soul_identity=False,
        skip_context_files=True,
        context_compressor=None,
        valid_tool_names=["delegate_task"],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _kanban_worker_guidance=False,
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        platform="cli",
        _platform_hint_overrides={},
        _auto_platform_hint_appends={},
        _environment_probe=False,
        _memory_store=None,
        _memory_manager=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        pass_session_id=False,
        session_id="session-1",
    )


def _routing_agent(verdict: str | None = "MECHANICAL", *, suspended: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        _fusion_enabled=True,
        _fusion_compaction_routing=True,
        _fusion_routing_suspended=suspended,
        _fusion_routing_verdict=verdict,
        _fusion_routing_runtime={
            "model": "cheap-mechanical-model",
            "provider": "openrouter",
            "api_key": "«redacted:sk-…»",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
        _fusion_frontier_runtime={
            "model": "frontier-model",
            "provider": "openrouter",
            "api_key": "sk-frontier",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
        model="frontier-model",
        provider="openrouter",
        api_key="sk-frontier",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        _emit_warning=lambda message: None,
    )


class _ModelConfigDB:
    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        self.model_config = model_config or {}
        self.calls: list[tuple[str, str, str]] = []

    def get_session(self, session_id: str) -> dict[str, str]:
        return {"id": session_id, "model_config": json.dumps(self.model_config)}

    def update_session_model_config_key(self, session_id: str, key: str, value_json: str) -> None:
        self.calls.append((session_id, key, value_json))


def test_default_config_declares_disabled_fusion_block():
    """Part D/E4: default fusion config exists but is fully inert."""

    fusion = DEFAULT_CONFIG.get("fusion")
    assert isinstance(fusion, dict), "fusion must be a dict in DEFAULT_CONFIG"
    assert fusion.get("enabled") is False, "fusion.enabled must default to False"
    assert fusion.get("compaction_routing") is False, "fusion.compaction_routing must default to False"
    required_keys = {
        "enabled", "compaction_routing",
        "sidekick_model", "sidekick_provider", "sidekick_base_url",
        "sidekick_api_key", "sidekick_api_mode", "sidekick_toolsets",
        "routing_model", "routing_provider", "routing_base_url",
        "routing_api_key", "routing_api_mode",
        "frontier_model",
    }
    assert required_keys.issubset(fusion.keys()), f"missing keys: {required_keys - set(fusion.keys())}"
    for key in required_keys - {"enabled", "compaction_routing", "sidekick_toolsets"}:
        assert not fusion[key], f"fusion.{key} must be falsy by default (empty string)"
    assert fusion["sidekick_toolsets"] == [], "fusion.sidekick_toolsets must default to empty list"


def test_fusion_main_agent_directive_is_stable_and_gated():
    """Part B/E4: prompt directive appears only when fusion is enabled."""

    patches = [
        patch("run_agent.load_soul_md", return_value="IDENTITY"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
        patch("run_agent.build_skills_system_prompt", return_value=""),
        patch("run_agent.get_toolset_for_tool", return_value=None),
        patch("agent.file_safety._resolve_active_profile_name", return_value="fusion-test"),
    ]
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        enabled = build_system_prompt_parts(_minimal_prompt_agent(fusion_enabled=True))
        disabled = build_system_prompt_parts(_minimal_prompt_agent(fusion_enabled=False))

    assert "Fusion sidekick" in enabled["stable"]
    assert "architect and reviewer" in enabled["stable"].lower()
    assert "Fusion sidekick" not in enabled["context"]
    assert "Fusion sidekick" not in enabled["volatile"]
    assert "Fusion sidekick" not in disabled["stable"]


def test_compaction_prompt_adds_work_complexity_section_and_records_verdict():
    """C1/C2: routing verdict is piggybacked on the existing summary LLM call."""

    owner = SimpleNamespace(_fusion_routing_verdict=None)
    compressor = _make_compressor(owner, routing=True)
    captured: dict[str, str] = {}

    def fake_call_llm(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return _response(
            "## Active Task\nUser asked: fix lint\n\n"
            "## Critical Context\nTests are failing.\n\n"
            "## Work Complexity Assessment\nMECHANICAL"
        )

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        compressor._generate_summary(_messages())

    assert "## Work Complexity Assessment" in captured["prompt"]
    assert "MECHANICAL" in captured["prompt"]
    assert "FRONTIER_JUDGMENT" in captured["prompt"]
    assert owner._fusion_routing_verdict == "MECHANICAL"
    assert "## Work Complexity Assessment" not in (compressor._previous_summary or "")


def test_fusion_verdict_section_is_stripped_without_eating_following_heading():
    """The routing verdict is state, not durable summary content."""

    owner = SimpleNamespace(_fusion_routing_verdict=None)
    compressor = _make_compressor(owner, routing=True)
    summary = (
        "## Active Task\nUser asked for a mechanical cleanup.\n\n"
        "## Work Complexity Assessment\nMECHANICAL\nOne short reason.\n\n"
        "## Critical Context\nKeep the existing public API.\n\n"
        "## Historical Remaining Work\nRun focused tests."
    )

    compressor._record_fusion_verdict(summary)
    stripped = compressor._strip_fusion_verdict_section(summary)

    assert owner._fusion_routing_verdict == "MECHANICAL"
    assert "## Work Complexity Assessment" not in stripped
    assert "One short reason" not in stripped
    assert "## Active Task" in stripped
    assert "## Critical Context" in stripped
    assert "## Historical Remaining Work" in stripped


def test_fusion_verdict_strip_is_noop_when_routing_disabled():
    owner = SimpleNamespace(_fusion_routing_verdict=None)
    compressor = _make_compressor(owner, routing=False)
    summary = "## Work Complexity Assessment\nMECHANICAL\n\n## Active Task\nKeep this text."

    assert compressor._strip_fusion_verdict_section(summary) == summary


def test_compaction_prompt_omits_work_complexity_section_when_routing_disabled():
    """E4/C5: disabled routing must not alter the summarizer prompt."""

    owner = SimpleNamespace(_fusion_routing_verdict=None)
    compressor = _make_compressor(owner, routing=False)
    captured: dict[str, str] = {}

    def fake_call_llm(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return _response("## Active Task\nNone.\n\n## Critical Context\nNone.")

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        compressor._generate_summary(_messages())

    assert "## Work Complexity Assessment" not in captured["prompt"]
    assert owner._fusion_routing_verdict is None


def test_apply_fusion_routing_switches_to_mechanical_target_and_clears_verdict(monkeypatch):
    """C2/C3/E5: MECHANICAL verdict switches once, then cannot be reused stale."""

    import agent.conversation_loop as conversation_loop

    helper = getattr(conversation_loop, "_apply_fusion_routing_verdict", None)
    assert callable(helper)

    calls: list[tuple[str, str, str, str, str]] = []

    def fake_switch(agent, new_model, new_provider, api_key="", base_url="", api_mode=""):
        calls.append((new_model, new_provider, api_key, base_url, api_mode))
        agent.model = new_model
        agent.provider = new_provider
        agent.api_key = api_key
        agent.base_url = base_url
        agent.api_mode = api_mode
        return True

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)
    agent = _routing_agent("MECHANICAL")

    helper(agent)
    helper(agent)

    assert calls == [
        (
            "cheap-mechanical-model",
            "openrouter",
            "«redacted:sk-…»",
            "https://openrouter.ai/api/v1",
            "chat_completions",
        )
    ]
    assert agent._fusion_routing_verdict is None


def test_apply_fusion_routing_persists_post_switch_runtime_without_credentials(monkeypatch):
    """Resumed sessions should restart on routed runtime, but never store keys."""

    import agent.conversation_loop as conversation_loop

    helper = getattr(conversation_loop, "_apply_fusion_routing_verdict", None)
    assert callable(helper)

    def fake_switch(agent, new_model, new_provider, api_key="", base_url="", api_mode=""):
        agent.model = new_model
        agent.provider = new_provider
        agent.api_key = api_key
        agent.base_url = base_url
        agent.api_mode = api_mode
        return True

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)
    agent = _routing_agent("MECHANICAL")
    agent.session_id = "routed-session"
    agent._session_db = _ModelConfigDB({"_branched_from": "parent-session"})

    assert helper(agent) is True

    assert agent._session_db.calls
    session_id, key, value_json = agent._session_db.calls[-1]
    assert session_id == "routed-session"
    assert key == "_fusion_routed_runtime"
    payload = json.loads(value_json)
    assert payload == {
        "model": "cheap-mechanical-model",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    assert "api_key" not in payload


def test_session_model_config_key_update_preserves_existing_metadata(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "session",
            source="cli",
            model="frontier-model",
            model_config={"_branched_from": "parent-session"},
        )
        routed = {
            "model": "cheap-model",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }
        db.update_session_model_config_key(
            "session",
            "_fusion_routed_runtime",
            json.dumps(routed),
        )

        row = db.get_session("session")
        assert row is not None
        model_config = json.loads(row["model_config"])
        assert model_config["_branched_from"] == "parent-session"
        assert model_config["_fusion_routed_runtime"] == routed

        db.update_session_model_config_key("session", "_fusion_routed_runtime", "null")
        row = db.get_session("session")
        assert row is not None
        model_config = json.loads(row["model_config"])
        assert model_config["_branched_from"] == "parent-session"
        assert model_config["_fusion_routed_runtime"] is None
    finally:
        db.close()


def test_apply_fusion_routing_switches_frontier_and_skips_when_already_on_target(monkeypatch):
    """E5: FRONTIER_JUDGMENT restores the recorded frontier runtime, unless already there."""

    import agent.conversation_loop as conversation_loop

    helper = getattr(conversation_loop, "_apply_fusion_routing_verdict", None)
    assert callable(helper)

    calls: list[tuple[str, str, str, str, str]] = []

    def fake_switch(agent, new_model, new_provider, api_key="", base_url="", api_mode=""):
        calls.append((new_model, new_provider, api_key, base_url, api_mode))
        agent.model = new_model
        agent.provider = new_provider
        agent.api_key = api_key
        agent.base_url = base_url
        agent.api_mode = api_mode
        return True

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)
    agent = _routing_agent("FRONTIER_JUDGMENT")
    agent.model = "cheap-mechanical-model"

    assert helper(agent) is True
    assert calls == [
        (
            "frontier-model",
            "openrouter",
            "sk-frontier",
            "https://openrouter.ai/api/v1",
            "chat_completions",
        )
    ]
    assert agent._fusion_routing_verdict is None

    agent._fusion_routing_verdict = "FRONTIER_JUDGMENT"
    assert helper(agent) is False
    assert calls == [
        (
            "frontier-model",
            "openrouter",
            "sk-frontier",
            "https://openrouter.ai/api/v1",
            "chat_completions",
        )
    ]
    assert agent._fusion_routing_verdict is None


def test_apply_fusion_routing_ignores_malformed_or_missing_targets_and_clears(monkeypatch):
    """E5: malformed/missing verdicts never switch and never leak stale state."""

    import agent.conversation_loop as conversation_loop

    helper = getattr(conversation_loop, "_apply_fusion_routing_verdict", None)
    assert callable(helper)

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *args, **kwargs: calls.append(args))

    malformed = _routing_agent("NOT_A_VERDICT")
    assert helper(malformed) is False
    assert malformed._fusion_routing_verdict is None

    missing_target = _routing_agent("MECHANICAL")
    missing_target._fusion_routing_runtime = {"model": ""}
    assert helper(missing_target) is False
    assert missing_target._fusion_routing_verdict is None

    absent = _routing_agent(None)
    assert helper(absent) is False
    assert absent._fusion_routing_verdict is None
    assert calls == []


def test_apply_fusion_routing_skips_cross_provider_route_without_explicit_key_or_endpoint(monkeypatch):
    """Provider safety: cross-provider cheap routes must not inherit frontier credentials/endpoints."""

    import agent.conversation_loop as conversation_loop

    helper = getattr(conversation_loop, "_apply_fusion_routing_verdict", None)
    assert callable(helper)

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *args, **kwargs: calls.append(args))

    for runtime in (
        {
            "model": "cheap-model",
            "provider": "cheap-provider",
            "base_url": "https://cheap.example/v1",
            "api_key": "",
            "api_mode": "chat_completions",
        },
        {
            "model": "cheap-model",
            "provider": "cheap-provider",
            "base_url": "",
            "api_key": "sk-cheap",
            "api_mode": "chat_completions",
        },
    ):
        agent = _routing_agent("MECHANICAL")
        agent.provider = "frontier-provider"
        agent._fusion_routing_runtime = runtime

        assert helper(agent) is False
        assert agent._fusion_routing_verdict is None

    assert calls == []


def test_fusion_routing_runtime_uses_explicit_routing_provider_before_sidekick_provider():
    """Provider ambiguity fix: routing credentials are explicit, then sidekick, then parent."""

    from agent.agent_init import _build_fusion_routing_runtime

    agent = SimpleNamespace(
        model="frontier-model",
        provider="frontier-provider",
        base_url="https://frontier.example/v1",
        api_key="sk-frontier",
        api_mode="chat_completions",
    )

    runtime = _build_fusion_routing_runtime(
        agent,
        {
            "routing_model": "cheap-model",
            "routing_provider": "cheap-provider",
            "routing_base_url": "https://cheap.example/v1",
            "routing_api_key": "sk-cheap",
            "routing_api_mode": "responses",
            "sidekick_model": "sidekick-model",
            "sidekick_provider": "sidekick-provider",
            "sidekick_base_url": "https://sidekick.example/v1",
            "sidekick_api_key": "sk-sidekick",
            "sidekick_api_mode": "chat_completions",
        },
    )

    assert runtime == {
        "model": "cheap-model",
        "provider": "cheap-provider",
        "base_url": "https://cheap.example/v1",
        "api_key": "sk-cheap",
        "api_mode": "responses",
    }

    fallback = _build_fusion_routing_runtime(
        agent,
        {
            "routing_model": "cheap-model",
            "sidekick_provider": "sidekick-provider",
            "sidekick_base_url": "https://sidekick.example/v1",
            "sidekick_api_key": "sk-sidekick",
            "sidekick_api_mode": "chat_completions",
        },
    )
    assert fallback["provider"] == "sidekick-provider"
    assert fallback["base_url"] == "https://sidekick.example/v1"
    assert fallback["api_key"] == "sk-sidekick"
    assert fallback["api_mode"] == "chat_completions"


def test_resume_restores_persisted_fusion_route_with_live_credentials(monkeypatch):
    """Resume uses the stored route shape but re-resolves credentials live."""

    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    routed = {
        "model": "cheap-model",
        "provider": "openrouter",
        "base_url": "https://stored.example/v1",
        "api_mode": "stored-mode",
    }
    agent = SimpleNamespace(
        _fusion_enabled=True,
        _fusion_compaction_routing=True,
        model="frontier-model",
        provider="openrouter",
        base_url="https://frontier.example/v1",
        api_mode="chat_completions",
    )

    class _FakeCLI(CLICommandsMixin):
        pass

    cli = _FakeCLI()
    cli.agent = agent
    cli._session_db = _ModelConfigDB({"_fusion_routed_runtime": routed})
    calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-live",
            "base_url": "https://live-openrouter.example/v1",
            "api_mode": "chat_completions",
        },
    )

    def fake_switch(agent_arg, new_model, new_provider, api_key="", base_url="", api_mode=""):
        calls.append(
            {
                "model": new_model,
                "provider": new_provider,
                "api_key": api_key,
                "base_url": base_url,
                "api_mode": api_mode,
            }
        )
        return True

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)

    assert CLICommandsMixin._restore_fusion_routed_runtime_for_session(cli, "resume-session") is True
    assert calls == [
        {
            "model": "cheap-model",
            "provider": "openrouter",
            "api_key": "sk-live",
            "base_url": "https://live-openrouter.example/v1",
            "api_mode": "chat_completions",
        }
    ]


def test_resume_without_persisted_fusion_route_does_not_switch(monkeypatch):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    agent = SimpleNamespace(
        _fusion_enabled=True,
        _fusion_compaction_routing=True,
        model="frontier-model",
        provider="openrouter",
    )

    class _FakeCLI(CLICommandsMixin):
        pass

    cli = _FakeCLI()
    cli.agent = agent
    cli._session_db = _ModelConfigDB({"_branched_from": "parent"})
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.switch_model",
        lambda *args, **kwargs: pytest.fail("no persisted route should not switch"),
    )

    assert CLICommandsMixin._restore_fusion_routed_runtime_for_session(cli, "resume-session") is False


def test_apply_fusion_routing_suspended_by_user_override(monkeypatch):
    """C4/E8: user model choice wins; compaction verdicts are cleared but ignored."""

    import agent.conversation_loop as conversation_loop

    helper = getattr(conversation_loop, "_apply_fusion_routing_verdict", None)
    assert callable(helper)

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *args, **kwargs: calls.append(args))
    agent = _routing_agent("MECHANICAL", suspended=True)

    helper(agent)

    assert calls == []
    assert agent._fusion_routing_verdict is None


def test_user_facing_switch_model_suspends_future_fusion_routing(monkeypatch):
    """C4/E8: manual /model path sets a session-long fusion-routing suspension marker."""

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._fusion_routing_suspended = False
    agent.session_id = "manual-session"
    agent._session_db = _ModelConfigDB({"_fusion_routed_runtime": {"model": "cheap"}})

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *args, **kwargs: True)

    assert AIAgent.switch_model(agent, "manual-model", "openrouter") is True
    assert agent._fusion_routing_suspended is True
    assert agent._session_db.calls == [
        ("manual-session", "_fusion_routed_runtime", "null")
    ]


def test_user_facing_switch_model_keeps_persisted_route_when_switch_fails(monkeypatch):
    """A failed manual override should not erase the still-active Fusion route."""

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._fusion_routing_suspended = False
    agent.session_id = "manual-session"
    agent._session_db = _ModelConfigDB({"_fusion_routed_runtime": {"model": "cheap"}})

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *args, **kwargs: False)

    assert AIAgent.switch_model(agent, "manual-model", "openrouter") is False
    assert agent._fusion_routing_suspended is True
    assert agent._session_db.calls == []


def test_suspended_routing_keeps_system_prompt_stable_between_compactions(monkeypatch):
    """E8/C5: a suspended verdict does not rebuild prompt bytes between compactions."""

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    setattr(agent, "_fusion_enabled", True)
    setattr(agent, "_fusion_compaction_routing", True)
    setattr(agent, "_fusion_routing_suspended", True)
    setattr(agent, "_fusion_routing_verdict", "MECHANICAL")
    setattr(agent, "_cached_system_prompt", "stable prompt")
    setattr(agent, "_build_system_prompt", lambda system_message=None: "rebuilt prompt")

    monkeypatch.setattr(
        "agent.conversation_compression.compress_context",
        lambda *args, **kwargs: ([{"role": "system", "content": "stable prompt"}], "stable prompt"),
    )
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.switch_model",
        lambda *args, **kwargs: pytest.fail("suspended path must not switch"),
    )

    compressed, prompt = AIAgent._compress_context(
        agent,
        [{"role": "user", "content": "hello"}],
        "stable prompt",
    )

    assert compressed == [{"role": "system", "content": "stable prompt"}]
    assert prompt == "stable prompt"
    assert getattr(agent, "_cached_system_prompt") == "stable prompt"
    assert getattr(agent, "_fusion_routing_verdict") is None

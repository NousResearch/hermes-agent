from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from plugins.context_engine.lifeos import LifeOSContextEngine


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sample_task(title: str, project: str = "", assigned_to: str = "", due: str = "2026-04-17") -> str:
    project_line = f"project: '[[{project}]]'\n" if project else ""
    assigned_line = f"assigned_to: '[[{assigned_to}]]'\n" if assigned_to else ""
    return (
        "---\n"
        f"title: {title}\n"
        "tags:\n  - task\n"
        "status: active\n"
        "priority: high\n"
        f"{project_line}"
        f"{assigned_line}"
        f"due: '{due}'\n"
        "---\n"
        f"# {title}\n"
    )


def _build_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "LifeOS"
    today = datetime.now().strftime("%Y-%m-%d")
    _write(vault / "CLAUDE.md", "# Memory\n\n## Focus\nShip Phoenix and close the budget loop.\n")
    _write(vault / "daily" / f"{today}.md", "# Daily\n\nToday: Phoenix work, budget review, and Todd follow-up.\n")
    _write(vault / "tasks" / "review-budget.md", _sample_task("Review budget", project="Phoenix", assigned_to="Todd Martinez"))
    _write(vault / "memory" / "projects" / "phoenix.md", "---\ntitle: Phoenix\naliases:\n  - project phoenix\n---\n# Phoenix\n\nPhoenix is the active operating project.\n")
    _write(vault / "memory" / "people" / "todd-martinez.md", "---\ntitle: Todd Martinez\naliases:\n  - Todd\n---\n# Todd Martinez\n\nPrimary finance counterpart for the budget review.\n")
    return vault


def _build_hermes_home(tmp_path: Path, vault: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {
            "vault_path": str(vault),
            "max_block_chars": 5000,
            "refresh": {"base_minutes": 20, "projects_minutes": 20, "people_minutes": 20},
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home


def test_lifeos_engine_prefetches_project_context(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-1", hermes_home=str(hermes_home), platform="cli", model="gpt-test")

    text = engine.prefetch("what's next on Phoenix?")

    assert "LIFEOS CONTEXT" in text
    assert "Project — Phoenix" in text
    assert "Review budget" in text
    assert engine.state["active_projects"] == ["Phoenix"]


def test_lifeos_engine_tools_report_and_update_focus(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-2", hermes_home=str(hermes_home), platform="cli", model="gpt-test")

    result = json.loads(engine.handle_tool_call("lifeos_set_focus", {"focus": "Ship Phoenix"}))
    status = json.loads(engine.handle_tool_call("lifeos_context_status", {}))
    refresh = json.loads(engine.handle_tool_call("lifeos_refresh_context", {"scope": "person", "target": "Todd"}))

    assert result["focus"] == "Ship Phoenix"
    assert status["current_focus"] == "Ship Phoenix"
    assert "Todd Martinez" in refresh["preview"]
    assert isinstance(status["tentative_updates"], list)
    assert isinstance(status["promotion_candidates"], list)


def test_lifeos_engine_tracks_tentative_updates_and_promotion_candidates(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-3", hermes_home=str(hermes_home), platform="cli", model="gpt-test")

    tentative = json.loads(
        engine.handle_tool_call(
            "lifeos_capture_update",
            {
                "type": "project_update",
                "target": "Phoenix",
                "content": "Need to confirm roadmap sequencing with finance",
                "confidence": "medium",
            },
        )
    )
    promotion = json.loads(
        engine.handle_tool_call(
            "lifeos_promote_to_project",
            {
                "target": "Phoenix",
                "content": "Confirmed next action: review budget with Todd",
                "why": "clear operational next action mentioned in conversation",
            },
        )
    )

    status = json.loads(engine.handle_tool_call("lifeos_context_status", {}))

    assert tentative["ok"] is True
    assert tentative["tentative_update"]["type"] == "project_update"
    assert promotion["ok"] is True
    assert promotion["promotion_candidate"]["type"] == "lifeos_project"
    assert status["tentative_updates"][0]["target"] == "Phoenix"
    assert status["promotion_candidates"][0]["status"] == "pending"


def test_lifeos_engine_reset_clears_overlay_state(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-4", hermes_home=str(hermes_home), platform="cli", model="gpt-test")
    engine.handle_tool_call(
        "lifeos_capture_update",
        {"type": "focus_shift", "content": "Switch to Phoenix budget review"},
    )
    engine.handle_tool_call(
        "lifeos_promote_to_honcho",
        {
            "content": "Tony prefers a tighter planning review at the start of implementation work",
            "why": "stable process preference",
        },
    )

    engine.on_session_reset()
    status = json.loads(engine.handle_tool_call("lifeos_context_status", {}))

    assert status["current_focus"] == ""
    assert status["tentative_updates"] == []
    assert status["promotion_candidates"] == []


def test_lifeos_engine_writes_promoted_context_to_daily_and_claude(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)
    today = datetime.now().strftime("%Y-%m-%d")

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-5", hermes_home=str(hermes_home), platform="cli", model="gpt-test")

    daily = json.loads(engine.handle_tool_call("lifeos_promote_to_daily", {"content": "Focus on Phoenix budget with Todd"}))
    claude = json.loads(engine.handle_tool_call("lifeos_promote_to_claude", {"content": "Current immediate focus: Phoenix budget review"}))

    daily_text = (vault / "daily" / f"{today}.md").read_text(encoding="utf-8")
    claude_text = (vault / "CLAUDE.md").read_text(encoding="utf-8")

    assert daily["ok"] is True
    assert claude["ok"] is True
    assert "Session Promoted Context" in daily_text
    assert "Focus on Phoenix budget with Todd" in daily_text
    assert "Session Promoted Context" in claude_text
    assert "Current immediate focus: Phoenix budget review" in claude_text


def test_lifeos_engine_can_apply_project_promotion_candidate(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-6", hermes_home=str(hermes_home), platform="cli", model="gpt-test")
    engine.handle_tool_call(
        "lifeos_promote_to_project",
        {
            "target": "Phoenix",
            "content": "Confirmed next action: review budget with Todd",
            "why": "clear operational next action mentioned in conversation",
        },
    )

    approved = json.loads(engine.handle_tool_call("lifeos_review_promotion_candidate", {"index": 0, "action": "approve"}))
    applied = json.loads(engine.handle_tool_call("lifeos_apply_promotion_candidate", {"index": 0}))
    status = json.loads(engine.handle_tool_call("lifeos_context_status", {}))
    project_text = (vault / "memory" / "projects" / "phoenix.md").read_text(encoding="utf-8")

    assert approved["ok"] is True
    assert approved["candidate"]["status"] == "approved"
    assert applied["ok"] is True
    assert applied["applied"]["status"] == "applied"
    assert status["promotion_candidates"][0]["status"] == "applied"
    assert "Session Promoted Context" in project_text
    assert "Confirmed next action: review budget with Todd" in project_text


def test_lifeos_engine_can_reject_promotion_candidate(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-7", hermes_home=str(hermes_home), platform="cli", model="gpt-test")
    engine.handle_tool_call(
        "lifeos_promote_to_task",
        {"content": "Maybe create a follow-up task", "why": "unclear if needed yet"},
    )

    rejected = json.loads(engine.handle_tool_call("lifeos_review_promotion_candidate", {"index": 0, "action": "reject"}))
    status = json.loads(engine.handle_tool_call("lifeos_context_status", {}))

    assert rejected["ok"] is True
    assert rejected["candidate"]["status"] == "rejected"
    assert status["promotion_candidates"][0]["status"] == "rejected"


def test_lifeos_engine_can_apply_honcho_promotion_candidate(tmp_path):
    vault = _build_vault(tmp_path)
    hermes_home = _build_hermes_home(tmp_path, vault)

    engine = LifeOSContextEngine()
    engine.on_session_start("sess-8", hermes_home=str(hermes_home), platform="cli", model="gpt-test")
    engine.handle_tool_call(
        "lifeos_promote_to_honcho",
        {
            "content": "Tony prefers tighter planning reviews at the start of implementation work",
            "why": "stable process preference",
        },
    )

    fake_manager = MagicMock()
    fake_manager.create_conclusion.return_value = True

    with (
        patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config") as cfg_loader,
        patch("plugins.memory.honcho.client.get_honcho_client", return_value=object()),
        patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=fake_manager),
    ):
        cfg = MagicMock()
        cfg.resolve_session_name.return_value = "LifeOS"
        cfg_loader.return_value = cfg
        approved = json.loads(engine.handle_tool_call("lifeos_review_promotion_candidate", {"index": 0, "action": "approve"}))
        applied = json.loads(engine.handle_tool_call("lifeos_apply_promotion_candidate", {"index": 0}))

    status = json.loads(engine.handle_tool_call("lifeos_context_status", {}))

    assert approved["ok"] is True
    assert applied["ok"] is True
    fake_manager.get_or_create.assert_called_once_with("LifeOS")
    fake_manager.create_conclusion.assert_called_once()
    assert status["promotion_candidates"][0]["status"] == "applied"


def _mock_response(content="Done", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _mock_tool_call(name="lifeos_test_tool", arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _StubEngine(LifeOSContextEngine):
    def __init__(self):
        super().__init__()
        self.prefetch_calls = []
        self.session_start_calls = []
        self.session_end_calls = []
        self.reset_calls = 0
        self.tool_calls = []

    def prefetch(self, query: str, **kwargs) -> str:
        self.prefetch_calls.append(query)
        return "[LIFEOS TEST CONTEXT] Focus on Phoenix."

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_start_calls.append({"session_id": session_id, **kwargs})
        super().on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages):
        self.session_end_calls.append({"session_id": session_id, "messages": list(messages)})
        super().on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        self.reset_calls += 1
        super().on_session_reset()

    def get_tool_schemas(self):
        schemas = super().get_tool_schemas()
        schemas.append(
            {
                "name": "lifeos_test_tool",
                "description": "Test-only context engine tool.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        )
        return schemas

    def handle_tool_call(self, name: str, args: dict, **kwargs) -> str:
        if name == "lifeos_test_tool":
            self.tool_calls.append({"name": name, "args": dict(args or {})})
            return json.dumps({"ok": True, "source": "stub"})
        return super().handle_tool_call(name, args, **kwargs)


def test_run_agent_injects_context_engine_prefetch_into_user_turn(tmp_path):
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {"vault_path": str(tmp_path / "LifeOS")},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_response("Done")

        result = agent.run_conversation("what next on Phoenix?")
        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        user_messages = [m for m in sent_messages if m.get("role") == "user"]

    assert result["final_response"] == "Done"
    assert engine.prefetch_calls == ["what next on Phoenix?"]
    assert any("[LIFEOS TEST CONTEXT] Focus on Phoenix." in msg.get("content", "") for msg in user_messages)


def test_run_agent_does_not_inject_context_engine_prefetch_into_system_prompt(tmp_path):
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {"vault_path": str(tmp_path / "LifeOS")},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_response("Done")

        agent.run_conversation("what next on Phoenix?")
        sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        system_messages = [m for m in sent_messages if m.get("role") == "system"]

    assert system_messages
    assert all("[LIFEOS TEST CONTEXT] Focus on Phoenix." not in msg.get("content", "") for msg in system_messages)


def test_agent_registers_context_engine_tool_schemas_and_routes_tool_calls(tmp_path):
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {"vault_path": str(tmp_path / "LifeOS")},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        routed = json.loads(agent.context_compressor.handle_tool_call("lifeos_test_tool", {}, messages=[]))

    assert "lifeos_test_tool" in agent.valid_tool_names
    assert "lifeos_test_tool" in agent._context_engine_tool_names
    assert any(t["function"]["name"] == "lifeos_test_tool" for t in agent.tools)
    assert routed["ok"] is True
    assert engine.tool_calls == [{"name": "lifeos_test_tool", "args": {}}]


def test_agent_calls_context_engine_lifecycle_hooks_for_start_reset_and_end(tmp_path):
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {"vault_path": str(tmp_path / "LifeOS")},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.reset_session_state()
        agent.shutdown_memory_provider(messages=[{"role": "user", "content": "done"}])

    assert len(engine.session_start_calls) == 1
    start_call = engine.session_start_calls[0]
    assert start_call["session_id"] == agent.session_id
    assert start_call["hermes_home"]
    assert start_call["platform"] == "cli"
    assert start_call["model"] == agent.model
    assert start_call["context_length"] == 131_072
    assert engine.reset_calls == 1
    assert len(engine.session_end_calls) == 1
    assert engine.session_end_calls[0]["session_id"] == agent.session_id
    assert engine.session_end_calls[0]["messages"] == [{"role": "user", "content": "done"}]


def test_agent_falls_back_to_builtin_compressor_when_lifeos_engine_load_fails(tmp_path):
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {"vault_path": str(tmp_path / "LifeOS")},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", side_effect=RuntimeError("boom")),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.name == "compressor"
    assert "lifeos_test_tool" not in getattr(agent, "valid_tool_names", set())


def test_run_agent_routes_context_engine_tool_call_end_to_end(tmp_path):
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "lifeos"},
        "lifeos_context": {"vault_path": str(tmp_path / "LifeOS")},
        "agent": {},
    }
    tc = _mock_tool_call(name="lifeos_test_tool", arguments="{}", call_id="c1")
    resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
    resp2 = _mock_response(content="Done", finish_reason="stop")

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        result = agent.run_conversation("use the lifeos tool")

    assert result["final_response"] == "Done"
    assert engine.tool_calls == [{"name": "lifeos_test_tool", "args": {}}]

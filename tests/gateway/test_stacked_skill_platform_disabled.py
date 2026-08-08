"""Regression test for stacked slash-skill invocations bypassing the
per-platform ``skills.platform_disabled`` gate.

``/skill-a /skill-b do XYZ`` loads every leading skill (up to 5), not just
the first (``agent.skill_commands.split_stacked_skill_commands`` /
``build_stacked_skill_invocation_message``). ``gateway.run.GatewayRunner.
_handle_message`` already re-checks the FIRST skill against the
per-platform disabled list before dispatch (``get_skill_commands()`` only
applies the *global* disabled list at scan time), but did not extend that
same check to the additional stacked skills — a skill an operator disabled
for a given platform still had its full SKILL.md content injected into the
agent's context for that turn if it was stacked behind an allowed one.
"""

import asyncio
import builtins
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    from gateway.run import GatewayRunner as _GR
    runner._session_key_for_source = _GR._session_key_for_source.__get__(runner, _GR)
    return runner


def _enable_normal_message_flow(runner):
    state = SimpleNamespace(
        turn=SimpleNamespace(lease=None, agent=None, started_ts=None),
        conversation=SimpleNamespace(model_override=None, one_turn_restore=None),
        persistent=SimpleNamespace(native_image_paths=[]),
    )
    runner._external_drain_active = False
    runner._claim_active_session_slot = lambda *_args, **_kwargs: (None, None)
    runner._session_state = lambda _key: state
    runner._persist_active_agents = lambda: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._restore_moa_one_shot = lambda *_args, **_kwargs: None
    runner._restore_pending_one_turn_model_override = lambda *_args, **_kwargs: None
    runner._release_turn_lease = lambda *_args, **_kwargs: None
    runner._handle_message_with_agent = AsyncMock(
        return_value={"final_response": "agent ok", "messages": []}
    )


def _make_skill(skills_dir, name, body="content"):
    sd = skills_dir / name
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc {name}\n---\n\n# {name}\n\n{body}\n"
    )


def _configure_protected_governance(home):
    (home / "governance").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """\
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: ardyn_engineering
    protected_task_classes:
      - ardyn_engineering
""",
        encoding="utf-8",
    )
    (home / "governance" / "skills-registry.yaml").write_text(
        """\
version: 1
skills:
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
""",
        encoding="utf-8",
    )


@pytest.fixture
def skills_env(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    import tools.skills_tool as skills_tool_module
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    import agent.skill_commands as skill_commands_mod
    skill_commands_mod._skill_commands = {}
    skill_commands_mod._skill_commands_platform = None
    return skills_dir


def test_stacked_second_skill_disabled_for_platform_is_blocked(monkeypatch, skills_env):
    """The whole stacked invocation is rejected when a NON-leading stacked
    skill is disabled for the message's platform — it must not silently load
    that skill's content just because only the first skill was checked."""
    import gateway.run as gateway_run
    import agent.skill_utils as skill_utils_mod

    _make_skill(skills_env, "allowed-skill")
    _make_skill(skills_env, "disabled-skill")

    monkeypatch.setattr(
        skill_utils_mod,
        "get_disabled_skill_names",
        lambda platform=None: {"disabled-skill"} if platform == "telegram" else set(),
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    result = asyncio.run(
        runner._handle_message(_make_event("/allowed-skill /disabled-skill do something"))
    )

    assert result is not None
    assert "disabled-skill" in result
    assert "disabled for telegram" in result


def test_gateway_skill_dispatch_denies_governance_blocked_skill(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    import tools.skills_tool as skills_tool_module
    import agent.skill_commands as skill_commands_mod

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _make_skill(skills_dir, "ToolTrust", body="blocked legacy content")
    home = tmp_path / "home"
    _configure_protected_governance(home)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    skill_commands_mod._skill_commands = {}
    skill_commands_mod._skill_commands_platform = None
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    result = asyncio.run(runner._handle_message(_make_event("/tooltrust do something")))

    assert result is not None
    assert "ToolTrust" in result
    assert "historical intent" in result


def test_gateway_skill_dispatch_denies_protected_setup_failure(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    home = tmp_path / "home"
    _configure_protected_governance(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated governance setup failure")),
    )

    runner = _make_runner()
    result = asyncio.run(runner._handle_message(_make_event("/tooltrust do something")))

    assert result is not None
    assert '"/tooltrust"' in result
    assert "denied" in result.lower()
    assert "protected task class" in result


def test_gateway_skill_dispatch_denies_when_skill_utils_import_fails_and_governance_eval_errors(
    monkeypatch, tmp_path
):
    import gateway.run as gateway_run

    home = tmp_path / "home"
    _configure_protected_governance(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    real_import = builtins.__import__

    def _deny_imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.skill_utils":
            raise ImportError(f"simulated import failure: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        "agent.skill_governance.evaluate_skill_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated governance evaluation failure")
        ),
    )
    monkeypatch.setattr(builtins, "__import__", _deny_imports)

    runner = _make_runner()
    result = asyncio.run(runner._handle_message(_make_event("/tooltrust do something")))

    assert result is not None
    assert '"/tooltrust"' in result
    assert "protected task class" in result


def test_gateway_skill_dispatch_denies_when_config_cannot_be_parsed(
    monkeypatch, tmp_path
):
    import gateway.run as gateway_run

    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("skills: [\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated setup failure")),
    )

    runner = _make_runner()
    result = asyncio.run(runner._handle_message(_make_event("/tooltrust do something")))

    assert result is not None
    assert '"/tooltrust"' in result
    assert "protected task class" in result


def test_gateway_skill_dispatch_keeps_unprotected_behavior_on_setup_failure(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    home = tmp_path / "home"
    _configure_protected_governance(home)
    (home / "config.yaml").write_text(
        """\
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: general_ops
    protected_task_classes:
      - ardyn_engineering
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated governance setup failure")),
    )

    runner = _make_runner()
    _enable_normal_message_flow(runner)
    result = asyncio.run(runner._handle_message(_make_event("/tooltrust do something")))

    assert isinstance(result, dict)
    assert result["final_response"] == "agent ok"
    runner._handle_message_with_agent.assert_awaited_once()

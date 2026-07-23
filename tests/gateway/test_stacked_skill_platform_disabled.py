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


def _make_skill(skills_dir, name, body="content"):
    sd = skills_dir / name
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc {name}\n---\n\n# {name}\n\n{body}\n"
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


@pytest.mark.asyncio
async def test_stacked_second_skill_disabled_for_platform_is_blocked(monkeypatch, skills_env):
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
    result = await runner._handle_message(
        _make_event("/allowed-skill /disabled-skill do something")
    )

    assert result is not None
    assert "disabled-skill" in result
    assert "disabled for telegram" in result


@pytest.mark.asyncio
async def test_stacked_all_enabled_skills_still_load(monkeypatch, skills_env):
    """Positive control: the new platform-disabled check must not over-block
    a stacked invocation where every skill is actually enabled."""
    import gateway.run as gateway_run
    import agent.skill_utils as skill_utils_mod

    _make_skill(skills_env, "alpha-skill", body="ALPHA BODY MARKER")
    _make_skill(skills_env, "beta-skill", body="BETA BODY MARKER")

    monkeypatch.setattr(
        skill_utils_mod, "get_disabled_skill_names", lambda platform=None: set()
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    event = _make_event("/alpha-skill /beta-skill do something")
    result = await runner._handle_message(event)

    # Not rejected: the handler falls through to normal message processing
    # with event.text rewritten to the combined stacked-skill payload.
    assert result is None or "disabled for" not in result
    assert "ALPHA BODY MARKER" in event.text
    assert "BETA BODY MARKER" in event.text

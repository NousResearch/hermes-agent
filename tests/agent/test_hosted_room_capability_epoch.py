"""#135994: hosted room member sessions ("Group: <room_id>") are as eternal as a Bot
Chat — reused for the room's lifetime — so their system prompts carry the capability
epoch stamp, and the restore path rebuilds them ONCE per SOUL/skills/toolset change
instead of restoring the prompt frozen at session creation forever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import _bot_mode_parts
from hermes_state import SessionDB
from tui_gateway.hosted_room_driver import room_session_title
from tools import bot_mode_probe

STORED_PROMPT = (
    "SYSTEM PROMPT BODY\n\nConversation started: Monday, January 05, 2026\n"
    "Model: test-model\nProvider: openrouter\nPlatform: tui"
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


# ── probe helpers ────────────────────────────────────────────────────────────


def test_hosted_room_title_predicate():
    assert bot_mode_probe.is_hosted_room_session_title("Group: room_abc")
    assert not bot_mode_probe.is_hosted_room_session_title("Bot Chat")
    assert not bot_mode_probe.is_hosted_room_session_title("Ordinary session")
    assert not bot_mode_probe.is_hosted_room_session_title("")
    assert not bot_mode_probe.is_hosted_room_session_title(None)  # type: ignore[arg-type]


def test_room_session_title_round_trips_the_predicate():
    assert bot_mode_probe.is_hosted_room_session_title(room_session_title("abc123"))


def test_room_member_legacy_prompt_needs_epoch_once(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    legacy = "old member prompt with no stamp"

    assert bot_mode_probe.stored_room_member_prompt_needs_epoch(legacy)

    stamped = legacy + "\n\n" + bot_mode_probe.epoch_line(home)
    assert not bot_mode_probe.stored_room_member_prompt_needs_epoch(stamped)


# ── prompt build: the member prompt is stamped, without the Bot-to-Bot section ─


def _agent(db, *, title: str) -> MagicMock:
    agent = MagicMock()
    agent._session_title_hint = title
    agent._session_db = db
    agent.session_id = "room-member-session"
    agent._bot_chat_timeless_prompt = False
    return agent


def test_bot_mode_parts_stamps_room_member_without_protocol(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    with SessionDB(db_path=home / "state.db") as db:
        agent = _agent(db, title="Group: room_abc")
        parts = _bot_mode_parts(agent)

    assert parts == [bot_mode_probe.epoch_line(home)]
    assert not agent._bot_chat_timeless_prompt


def test_bot_mode_parts_leaves_ordinary_sessions_unstamped(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    with SessionDB(db_path=home / "state.db") as db:
        assert _bot_mode_parts(_agent(db, title="Ordinary session")) == []


# ── restore path: rebuild once per capability change, under the room's title ──


def _room_member_agent(db, *, title="Group: room_abc", hint=True) -> MagicMock:
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "room-member-session"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.platform = "tui"
    agent._session_db = db
    agent._use_prompt_caching = False
    agent._build_system_prompt = MagicMock(return_value="NEW_MEMBER_PROMPT")
    agent.enabled_toolsets = agent.disabled_toolsets = None
    agent.tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    agent.valid_tool_names = {"web_search"}
    agent._bot_mode_protocol = True
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    # The gateway attaches the hidden title as the build-time hint; when it is
    # absent the DB row must resolve it instead.
    agent._session_title_hint = title if hint else ""
    return agent


def _room_session(tmp_path, stored_prompt: str, *, soul: str | None = None):
    home = tmp_path / ".hermes"
    home.mkdir()
    if soul is not None:
        (home / "SOUL.md").write_text(soul, encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    db.create_session("room-member-session", source="hosted_room")
    db.set_session_title("room-member-session", "Group: room_abc")
    db.update_system_prompt("room-member-session", stored_prompt)
    return home, db


def test_legacy_unstamped_member_prompt_rebuilds_once_under_its_own_title(tmp_path):
    """The issue's exact scenario: a member session created BEFORE this fix has no
    stamp, so its stored prompt must rebuild once — adopting the new SOUL.md — while
    staying a room member prompt (no Bot-to-Bot protocol section via a forced
    "Bot Chat" hint)."""
    home, db = _room_session(tmp_path, STORED_PROMPT, soul="first soul")
    agent = _room_member_agent(db, hint=False)  # title must come from the DB row

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_called_once()
    assert agent._session_title_hint == "Group: room_abc"
    assert agent._cached_system_prompt == "NEW_MEMBER_PROMPT"
    assert db.get_session("room-member-session")["system_prompt"] == "NEW_MEMBER_PROMPT"
    db.close()


def test_stamped_member_prompt_rebuilds_when_soul_changes(tmp_path):
    home, db = _room_session(tmp_path, STORED_PROMPT, soul="first soul")
    stamped = STORED_PROMPT + "\n\n" + bot_mode_probe.epoch_line(home)
    db.update_system_prompt("room-member-session", stamped)

    (home / "SOUL.md").write_text("second soul: knows about teammate Bob", encoding="utf-8")
    agent = _room_member_agent(db)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_called_once()
    assert agent._session_title_hint == "Group: room_abc"
    db.close()


def test_stamped_member_prompt_is_restored_verbatim_when_surface_unchanged(tmp_path):
    home, db = _room_session(tmp_path, STORED_PROMPT, soul="first soul")
    stamped = STORED_PROMPT + "\n\n" + bot_mode_probe.epoch_line(home)
    db.update_system_prompt("room-member-session", stamped)

    agent = _room_member_agent(db)
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == stamped
    db.close()


def test_ordinary_unstamped_session_is_not_rebuilt(tmp_path):
    """A regular session's prompt carries no stamp and must keep restoring verbatim —
    the legacy migration is title-gated to hosted room member sessions."""
    home = tmp_path / ".hermes"
    home.mkdir()
    db = SessionDB(db_path=home / "state.db")
    db.create_session("ordinary-session", source="tui")
    db.set_session_title("ordinary-session", "Ordinary session")
    db.update_system_prompt("ordinary-session", STORED_PROMPT)

    agent = _room_member_agent(db, title="Ordinary session")
    agent.session_id = "ordinary-session"
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == STORED_PROMPT
    db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""A /model (or route) commit must not null a continuing session's stored system prompt."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB

SESSION_ID = "switch-session"


def _stored_prompt(model: str, provider: str) -> str:
    return (
        "You are Hermes Agent.\n\n"
        "Conversation started: Thursday, September 24, 2026\n"
        f"Model: {model}\n"
        f"Provider: {provider}"
    )


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Real SessionDB on a temp state.db, isolated from the live HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


_ROUTE_COMMITS = {
    "model": lambda db: db.update_session_model(
        SESSION_ID, "anthropic/claude-opus-4.8", provider="anthropic", base_url="https://a/v1",
    ),
    "runtime_lock": lambda db: db.update_session_runtime_lock(
        SESSION_ID, model="anthropic/claude-opus-4.8", provider="anthropic", confirmed=True,
    ),
    "billing_route": lambda db: db.update_session_billing_route(
        SESSION_ID, provider="openrouter", base_url="https://o/v1",
    ),
}


@pytest.mark.parametrize("commit", _ROUTE_COMMITS.values(), ids=_ROUTE_COMMITS.keys())
def test_route_commits_keep_the_stored_prompt(db, commit):
    """Each switch-path writer leaves the stored prompt in place."""
    prompt = _stored_prompt("x-ai/grok-4.5", "nous")
    db.create_session(SESSION_ID, source="discord", model="x-ai/grok-4.5")
    db.update_system_prompt(SESSION_ID, prompt)

    commit(db)

    assert db.get_session(SESSION_ID)["system_prompt"] == prompt


def test_compression_tip_adoption_applies_the_identity_check(db):
    """The other reader that seeds ``_cached_system_prompt`` from a stored row.

    ``_adopt_live_compression_child`` bypasses ``_restore_or_build_system_prompt`` (the turn
    only restores while the slot is None), so with the NULL gone it must apply the identity
    check itself: a tip whose route moved since its last persist stays unseeded (the next
    restore rebuilds), while a matching tip is adopted verbatim.

    The slot starts NON-NULL — it holds the parent's prompt, exactly as a live agent that
    cached a turn before the parent rotated. Adoption moves ``agent.session_id`` onto the
    child, so a rejected child prompt must not leave the parent's bytes behind: the turn gate
    (``turn_context``: restore/rebuild only while the slot is None) would then send the parent
    session's prompt for the child turn unvalidated.
    """
    from agent.conversation_compression import _adopt_live_compression_child

    stale = _stored_prompt("model-a", "prov-a")
    parent_prompt = _stored_prompt("parent-model", "parent-provider")
    db.create_session("parent", source="discord", model="model-a")
    db.end_session("parent", "compression")
    db.create_session("child", source="discord", model="model-a", parent_session_id="parent")
    db.update_system_prompt("child", stale)
    db.append_message("child", "user", "hi")
    db.update_session_model("child", "model-b", provider="prov-b", base_url="https://b/v1")

    def _adopt(model: str, provider: str) -> MagicMock:
        agent = MagicMock()
        agent._cached_system_prompt = parent_prompt
        agent.session_id = "parent"
        agent.model, agent.provider = model, provider
        agent.pass_session_id = False
        agent.context_compressor = None
        agent._memory_manager = None
        assert _adopt_live_compression_child(agent, db, "parent") is not None
        return agent

    rejected = _adopt("model-b", "prov-b")
    assert rejected.session_id == "child"
    assert rejected._cached_system_prompt is None
    assert _adopt("model-a", "prov-a")._cached_system_prompt == stale


def _turn_agent(db, model: str, provider: str, prose: str = "") -> MagicMock:
    """A fresh per-turn agent (the gateway shape) whose rebuild renders the real trailer."""
    from agent.system_prompt import _timestamp_line

    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = SESSION_ID
    agent.model, agent.provider = model, provider
    agent.platform = "discord"
    agent.pass_session_id = False
    agent.session_start = None
    agent._bot_chat_timeless_prompt = False
    agent._persist_disabled = False
    agent._use_prompt_caching = False
    agent._session_db = db
    agent._platform_hint_overrides = None
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    agent.enabled_toolsets = agent.disabled_toolsets = None
    agent._build_system_prompt = MagicMock(side_effect=lambda _sm: f"You are Hermes Agent.\n\n{prose}{_timestamp_line(agent)}")
    return agent


def _run_turn(db, model: str, provider: str, prose: str = "") -> MagicMock:
    from agent.conversation_loop import _restore_or_build_system_prompt

    agent = _turn_agent(db, model, provider, prose)
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
    return agent


@pytest.mark.parametrize(
    ("live_model", "live_provider"),
    [("x-ai/grok-4.5", ""), ("", "nous")],
    ids=["provider_empty", "model_empty"],
)
def test_emptied_live_identity_rebuilds_once_then_reuses(db, live_model, live_provider):
    """A stored ``Provider:``/``Model:`` line with an empty live value is a stale route: route
    commits keep the stored prompt, so this check is the only rebuild trigger. The rebuilt prompt
    omits the empty line, and memory/context prose carrying its own ``Provider:``/``Model:`` lines
    must not stand in for it, or every turn would rebuild (a prompt-cache miss per turn)."""
    prose = "MEMORY\nProvider: openrouter\nModel: some/other-model\n\n"
    stale = _stored_prompt("x-ai/grok-4.5", "nous")
    db.create_session(SESSION_ID, source="discord", model="x-ai/grok-4.5")
    db.update_system_prompt(SESSION_ID, stale)

    _run_turn(db, live_model, live_provider, prose)._build_system_prompt.assert_called_once()
    rebuilt = db.get_session(SESSION_ID)["system_prompt"]
    assert rebuilt != stale

    second = _run_turn(db, live_model, live_provider, prose)
    second._build_system_prompt.assert_not_called()
    assert second._cached_system_prompt == rebuilt


def test_prompt_without_identity_lines_keeps_reusing(db):
    """Pre-trailer prompts carry no identity lines; they must not rebuild on upgrade."""
    legacy = "You are Hermes Agent.\n\nConversation started: Thursday, September 24, 2026"
    db.create_session(SESSION_ID, source="discord", model="x-ai/grok-4.5")
    db.update_system_prompt(SESSION_ID, legacy)

    agent = _run_turn(db, "x-ai/grok-4.5", "")
    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == legacy

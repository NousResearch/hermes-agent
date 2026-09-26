"""A /model (or route) commit must not null a continuing session's stored system prompt.

Regression for the WARNING ``Stored system prompt for session X is null; rebuilding from
scratch this turn ... Investigate the previous turn's update_system_prompt write path``:
three switch-path writers cleared ``system_prompt``/``system_prompt_hash`` "so stale
Model:/Provider: footers rebuild" —

  * ``SessionDB.update_session_model``      (every /model commit)
  * ``SessionDB.update_session_runtime_lock`` (Browser / API-client lock)
  * ``SessionDB.update_session_billing_route`` (billing route; ``switch_model`` after the swap)

— so the next turn read a NULL row and took the broken-row branch of
``agent.conversation_loop._restore_or_build_system_prompt``: a WARNING blaming the previous
turn's ``update_system_prompt`` write path, plus a full prompt rebuild (the whole provider
prefix re-billed) even on a commit that never moved the route — the picker re-committing the
session's *current* model, which the production log shows verbatim
(``switched from deepseek-v4.1-flash to deepseek-v4.1-flash via hyper``).

The runtime-identity check (``_stored_prompt_matches_runtime``) is the single mechanism that
already rebuilds exactly when the stored footer is stale, so the writers preserve the row and
the check decides: a real switch rebuilds (INFO) on the next turn, a same-route commit reuses
the bytes verbatim, and the WARNING goes back to meaning "the stored prompt was lost".

Real ``SessionDB`` on a temp file; no mock stands in for the DB layer.
"""

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
    """Each switch-path writer leaves the prompt snapshot (and its dedup row) in place."""
    prompt = _stored_prompt("x-ai/grok-4.5", "nous")
    db.create_session(SESSION_ID, source="discord", model="x-ai/grok-4.5")
    db.update_system_prompt(SESSION_ID, prompt)

    commit(db)

    assert db.get_session(SESSION_ID)["system_prompt"] == prompt
    # Content-addressed storage intact: the row still resolves through its hash.
    raw = db._conn.execute(
        "SELECT system_prompt, system_prompt_hash FROM sessions WHERE id = ?", (SESSION_ID,)
    ).fetchone()
    assert raw["system_prompt"] is None
    assert raw["system_prompt_hash"] is not None
    assert db._conn.execute(
        "SELECT COUNT(*) FROM system_prompts WHERE hash = ?", (raw["system_prompt_hash"],)
    ).fetchone()[0] == 1


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

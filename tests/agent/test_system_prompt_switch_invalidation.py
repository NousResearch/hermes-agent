"""A /model (or route) commit must not null a continuing session's stored system prompt.

Regression for the WARNING ``Stored system prompt for session X is null; rebuilding from
scratch this turn ... Investigate the previous turn's update_system_prompt write path``:
three switch-path writers cleared ``system_prompt``/``system_prompt_hash`` "so stale
Model:/Provider: footers rebuild" —

  * ``SessionDB.update_session_model``      (every /model commit)
  * ``SessionDB.update_session_runtime_lock`` (Browser / API-client lock)
  * ``SessionDB.update_session_billing_route`` (billing route, also on provider fallbacks)

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

import logging
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
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


def _make_agent(db, *, model: str, provider: str, prebuilt: str) -> MagicMock:
    """The minimal agent ``_restore_or_build_system_prompt`` needs; the DB is real."""
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = SESSION_ID
    agent.model = model
    agent.provider = provider
    agent.platform = "discord"
    agent._session_db = db
    agent._use_prompt_caching = False
    agent._persist_disabled = True  # no on_session_start hook, no tool-pin rewrite
    agent.enabled_toolsets = agent.disabled_toolsets = None
    agent.tools = []
    agent._build_system_prompt = MagicMock(return_value=prebuilt)
    return agent


def _continue_turn(db, *, model: str, provider: str, prebuilt: str, caplog):
    """Run the next turn of a continuing session and return (agent, warnings)."""
    agent = _make_agent(db, model=model, provider=provider, prebuilt=prebuilt)
    with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    return agent, warnings


class TestSwitchCommitsKeepTheStoredPrompt:
    """Each switch-path writer leaves the prompt snapshot (and its dedup row) in place."""

    def _seed(self, db) -> str:
        prompt = _stored_prompt("x-ai/grok-4.5", "nous")
        db.create_session(SESSION_ID, source="discord", model="x-ai/grok-4.5")
        db.update_system_prompt(SESSION_ID, prompt)
        return prompt

    def test_model_commit_keeps_the_snapshot(self, db):
        prompt = self._seed(db)

        db.update_session_model(
            SESSION_ID, "anthropic/claude-opus-4.8", provider="anthropic", base_url="https://a/v1",
        )

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

    def test_runtime_lock_keeps_the_snapshot(self, db):
        prompt = self._seed(db)

        db.update_session_runtime_lock(
            SESSION_ID, model="anthropic/claude-opus-4.8", provider="anthropic", confirmed=True,
        )

        assert db.get_session(SESSION_ID)["system_prompt"] == prompt

    def test_billing_route_keeps_the_snapshot(self, db):
        prompt = self._seed(db)

        db.update_session_billing_route(SESSION_ID, provider="openrouter", base_url="https://o/v1")

        assert db.get_session(SESSION_ID)["system_prompt"] == prompt


class TestNextTurnAfterASwitchCommit:
    def test_same_route_commit_reuses_the_stored_bytes(self, db, caplog):
        """The picker re-committing the session's current route must not cost a rebuild."""
        prompt = _stored_prompt("glm-5.3", "zro")
        db.create_session(SESSION_ID, source="discord", model="glm-5.3")
        db.update_system_prompt(SESSION_ID, prompt)

        db.update_session_model(SESSION_ID, "glm-5.3", provider="zro", base_url="https://hyper/v1")

        agent, warnings = _continue_turn(
            db, model="glm-5.3", provider="zro", prebuilt="REBUILT", caplog=caplog,
        )

        assert agent._cached_system_prompt == prompt
        agent._build_system_prompt.assert_not_called()
        assert warnings == []

    def test_route_change_rebuilds_through_the_identity_check(self, db, caplog):
        """A real switch still rebuilds — on the next turn, via the stale footer, not a null row."""
        db.create_session(SESSION_ID, source="discord", model="deepseek-v4.1-flash")
        db.update_system_prompt(SESSION_ID, _stored_prompt("deepseek-v4.1-flash", "hyper"))

        db.update_session_model(
            SESSION_ID, "glm-5.3", provider="zro", base_url="https://hyper/v1",
        )

        rebuilt = _stored_prompt("glm-5.3", "zro")
        agent, warnings = _continue_turn(
            db, model="glm-5.3", provider="zro", prebuilt=rebuilt, caplog=caplog,
        )

        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == rebuilt
        # The rebuilt bytes are persisted so the following turns reuse them verbatim.
        assert db.get_session(SESSION_ID)["system_prompt"] == rebuilt
        assert warnings == []
        assert any(
            r.levelno == logging.INFO and "stale runtime identity" in r.getMessage()
            for r in caplog.records
        )

    def test_a_genuinely_lost_prompt_still_warns(self, db, caplog):
        """The WARNING keeps its real meaning: nobody switched — the row was lost."""
        db.create_session(SESSION_ID, source="discord", model="glm-5.3")
        db.update_system_prompt(SESSION_ID, _stored_prompt("glm-5.3", "zro"))
        db.update_system_prompt(SESSION_ID, None)  # a write that lost the prompt

        agent, warnings = _continue_turn(
            db, model="glm-5.3", provider="zro", prebuilt="REBUILT", caplog=caplog,
        )

        assert agent._cached_system_prompt == "REBUILT"
        assert any(
            "is null; rebuilding" in w.getMessage() and "update_system_prompt write path" in w.getMessage()
            for w in warnings
        )

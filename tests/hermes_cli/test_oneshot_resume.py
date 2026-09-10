"""Tests for `hermes -z --resume <session>` (#105892).

The oneshot path used to accept ``--resume``/``-c`` in the parser but silently drop
them: ``_run_oneshot_from_args`` ran before any session-arg normalization and never
forwarded ``args.resume``, so every resumed one-shot turn started a FRESH session —
the wire carried only ``[system, current user]`` and the model "forgot" everything.
These tests pin the loader contract (chain redirect, unknown-session error,
session_meta filtering, empty-session fresh start, ended-row reopen) and the resume
kwarg wiring, plus the stored-runtime restore (review on #105957): a resumed one-shot
must run on the session's stored model/provider runtime and on a reopened session row.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from hermes_cli.oneshot import (
    _apply_stored_session_runtime,
    _load_resume_target,
    _ModelChoice,
    run_oneshot,
)


def _db_with_session(tmp_path, sid, *, messages=()):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id=sid, source="cli")
    for role, content in messages:
        db.append_message(sid, role, content=content)
    return db


class TestLoadResumeTarget:
    def test_no_resume_is_a_noop(self, tmp_path):
        db = _db_with_session(tmp_path, "s1")
        try:
            assert _load_resume_target(db, None) == (None, [], None)
            assert _load_resume_target(db, "") == (None, [], None)
        finally:
            db.close()

    def test_missing_store_raises_rather_than_silent_fresh(self):
        # An explicit --resume with no session store must fail loudly; starting a
        # fresh session here is exactly the history-dropping bug this fixes.
        with pytest.raises(RuntimeError, match="session store unavailable"):
            _load_resume_target(None, "s1")

    def test_unknown_session_raises(self, tmp_path):
        db = _db_with_session(tmp_path, "s1")
        try:
            with pytest.raises(RuntimeError, match="session not found: missing-sid"):
                _load_resume_target(db, "missing-sid")
        finally:
            db.close()

    def test_returns_history_for_stored_session(self, tmp_path):
        db = _db_with_session(
            tmp_path, "s1",
            messages=[("user", "Remember the secret word ZEBRA42"),
                      ("assistant", "I've noted the secret word: ZEBRA42.")],
        )
        try:
            sid, history, meta = _load_resume_target(db, "s1")
            assert sid == "s1"
            assert [m["role"] for m in history] == ["user", "assistant"]
            assert history[0]["content"] == "Remember the secret word ZEBRA42"
            assert meta["id"] == "s1"
        finally:
            db.close()

    def test_empty_session_keeps_resolved_id(self, tmp_path):
        # Chat's contract: a resumed session with no messages starts fresh (no rows to
        # replay) but the turn is recorded under the SELECTED id. Dropping the id here
        # re-minted a session for `hermes -z "hello" -c <title> --create-if-missing`,
        # leaving the freshly created titled session empty (review on #105957).
        db = _db_with_session(tmp_path, "s1")
        try:
            sid, history, meta = _load_resume_target(db, "s1")
            assert sid == "s1"
            assert history == []
            assert meta["id"] == "s1"
        finally:
            db.close()

    def test_compression_chain_redirects_to_child_with_messages(self, tmp_path):
        # Compression ends a session and forks a child that holds the rows; the loader
        # must land on the child, not the empty parent (resolve_resume_session_id).
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli", parent_session_id="parent")
        db.append_message("child", "user", content="hi")
        try:
            sid, history, _meta = _load_resume_target(db, "parent")
            assert sid == "child"
            assert [m["role"] for m in history] == ["user"]
        finally:
            db.close()

    def test_resume_reopens_ended_session_row(self, tmp_path):
        # The previous run stamped ended_at; without reopen_session() the resumed turn is
        # recorded under a row that stays closed and end_session() cannot stamp the new
        # boundary (it only writes rows whose ended_at is null) — review on #105957.
        db = _db_with_session(tmp_path, "s1", messages=[("user", "hi")])
        db.end_session("s1", "agent_close")
        row = db.get_session("s1")
        assert row["ended_at"] is not None and row["end_reason"] == "agent_close"
        try:
            sid, _history, _meta = _load_resume_target(db, "s1")
            assert sid == "s1"
            reopened = db.get_session("s1")
            assert reopened["ended_at"] is None
            assert reopened["end_reason"] is None
        finally:
            db.close()

class TestApplyStoredSessionRuntime:
    """A resumed one-shot must run on the session's stored runtime, not the ambient
    config (review on #105957): with ambient ``openrouter/ambient-model`` and a stored
    ``custom:stored/stored-model`` session, the agent previously received the ambient
    model/provider on the wire."""

    _AMBIENT_KEY = object()  # sentinel: any non-None token; resume must drop it when the provider changes

    @staticmethod
    def _stored_db(tmp_path, *, model="stored-model", route=None):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="s1", source="cli")
        if model:
            db.update_session_model("s1", model)
        if route is not None:
            # Same two shapes _persist_model_switch_to_session writes (nested gateway_runtime
            # for CLI resume, top-level keys for TUI).
            db.patch_session_model_config("s1", {"gateway_runtime": route, **route})
        meta = db.get_session("s1")
        db.close()
        return meta

    def test_stored_runtime_replaces_ambient_choice(self, tmp_path):
        meta = self._stored_db(tmp_path, route={"provider": "custom:stored", "base_url": "http://stored:9", "api_mode": "responses"})
        choice = _apply_stored_session_runtime(
            _ModelChoice("ambient-model", "openrouter", api_key=self._AMBIENT_KEY),
            meta, explicit_model=False)
        assert choice.model == "stored-model"
        assert choice.provider == "custom:stored"
        assert choice.base_url == "http://stored:9"
        assert choice.api_mode == "responses"
        # The ambient key belongs to the ambient endpoint and must not ride along.
        assert choice.api_key is None

    def test_explicit_model_flag_wins(self, tmp_path):
        meta = self._stored_db(tmp_path, route={"provider": "custom:stored"})
        choice = _apply_stored_session_runtime(
            _ModelChoice("explicit-model", "openrouter", api_key=self._AMBIENT_KEY),
            meta, explicit_model=True)
        assert choice.model == "explicit-model"
        assert choice.provider == "openrouter"
        assert choice.api_key is self._AMBIENT_KEY

    def test_matching_route_is_noop_and_keeps_credentials(self, tmp_path):
        meta = self._stored_db(tmp_path, route={"provider": "openrouter"})
        choice = _apply_stored_session_runtime(
            _ModelChoice("stored-model", "openrouter", base_url=None, api_key=self._AMBIENT_KEY),
            meta, explicit_model=False)
        assert choice.api_key is self._AMBIENT_KEY
        assert choice.api_mode is None

class TestRunAgentResumeRuntime:
    """End-to-end wiring: ``_run_agent`` must hand AIAgent the session's stored runtime
    and a reopened session row (both regressions from the review on #105957)."""

    def test_run_agent_uses_stored_runtime_and_reopens_row(self, tmp_path, monkeypatch):
        import hermes_cli.oneshot as oneshot_mod

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", content="hi")
        db.update_session_model("s1", "stored-model")
        db.patch_session_model_config(
            "s1", {"gateway_runtime": {"provider": "custom:stored", "base_url": "http://stored:9"},
                   "provider": "custom:stored", "base_url": "http://stored:9"})
        db.end_session("s1", "agent_close")

        captured = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
            def __setattr__(self, name, _value):
                pass
            def run_conversation(self, _prompt, conversation_history=None):
                captured["history"] = conversation_history
                return {"final_response": "ok", "session_id": "s1"}
            def close(self):
                pass

        monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: db)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "ambient-model", "provider": "openrouter"}})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_kw: {"api_key": "resolved", "base_url": None, "provider": "custom:stored",
                          "requested_provider": "custom:stored", "api_mode": "chat", "credential_pool": None})
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: [])
        monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
        monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

        try:
            text, result = oneshot_mod._run_agent("hello", resume="s1")
            assert text == "ok" and result["final_response"] == "ok"
            assert captured["model"] == "stored-model"
            assert captured["provider"] == "custom:stored"
            assert captured["session_id"] == "s1"
            assert captured["history"][0]["role"] == "user"
            row = db.get_session("s1")
            assert row["ended_at"] is None and row["end_reason"] is None
        finally:
            db.close()

    def test_run_agent_explicit_model_beats_stored_runtime(self, tmp_path, monkeypatch):
        import hermes_cli.oneshot as oneshot_mod

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="s1", source="cli")
        db.update_session_model("s1", "stored-model")

        captured = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
            def __setattr__(self, name, _value):
                pass
            def run_conversation(self, _prompt, conversation_history=None):
                return {"final_response": "ok"}
            def close(self):
                pass

        monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: db)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "ambient-model", "provider": "openrouter"}})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_kw: {"api_key": "resolved", "base_url": None, "provider": "openrouter",
                          "requested_provider": "openrouter", "api_mode": "chat", "credential_pool": None})
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: [])
        monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
        monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

        try:
            oneshot_mod._run_agent("hello", model="explicit-model", provider="openrouter", resume="s1")
            assert captured["model"] == "explicit-model"
            assert captured["provider"] == "openrouter"
        finally:
            db.close()


class TestRunOneshotForwardsResume:
    def test_resume_kwarg_reaches_run_agent(self, monkeypatch):
        captured = {}

        def _fake_run_agent(prompt, **kwargs):
            captured.update(kwargs, prompt=prompt)
            return "ok", {"final_response": "ok"}

        monkeypatch.setattr("hermes_cli.oneshot._run_agent", _fake_run_agent)
        rc = run_oneshot("hello", model="m", provider="custom", resume="sess-1")
        assert rc == 0
        assert captured["prompt"] == "hello"
        assert captured["resume"] == "sess-1"

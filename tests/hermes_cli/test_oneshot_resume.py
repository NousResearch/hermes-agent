"""Tests for `hermes -z --resume <session>` (#105892).

The oneshot path used to accept ``--resume``/``-c`` in the parser but silently drop
them: ``_run_oneshot_from_args`` ran before any session-arg normalization and never
forwarded ``args.resume``, so every resumed one-shot turn started a FRESH session —
the wire carried only ``[system, current user]`` and the model "forgot" everything.
These tests pin the loader contract (chain redirect, unknown-session error,
session_meta filtering, empty-session fresh start) and the resume kwarg wiring.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from hermes_cli.oneshot import _load_resume_target, run_oneshot


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
            assert _load_resume_target(db, None) == (None, [])
            assert _load_resume_target(db, "") == (None, [])
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
            sid, history = _load_resume_target(db, "s1")
            assert sid == "s1"
            assert [m["role"] for m in history] == ["user", "assistant"]
            assert history[0]["content"] == "Remember the secret word ZEBRA42"
        finally:
            db.close()

    def test_session_meta_rows_are_dropped(self, tmp_path):
        db = _db_with_session(
            tmp_path, "s1",
            messages=[("session_meta", "model switch"), ("user", "hello")],
        )
        try:
            sid, history = _load_resume_target(db, "s1")
            assert sid == "s1"
            assert [m["role"] for m in history] == ["user"]
        finally:
            db.close()

    def test_empty_session_starts_fresh(self, tmp_path):
        # Chat's contract: a resumed session with no messages starts fresh (same session
        # id, no history rows to replay) — oneshot must not crash or resurrect the id.
        db = _db_with_session(tmp_path, "s1")
        try:
            assert _load_resume_target(db, "s1") == (None, [])
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
            sid, history = _load_resume_target(db, "parent")
            assert sid == "child"
            assert [m["role"] for m in history] == ["user"]
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

    def test_no_resume_forwards_none(self, monkeypatch):
        captured = {}

        def _fake_run_agent(prompt, **kwargs):
            captured.update(kwargs)
            return "ok", {"final_response": "ok"}

        monkeypatch.setattr("hermes_cli.oneshot._run_agent", _fake_run_agent)
        rc = run_oneshot("hello", model="m", provider="custom")
        assert rc == 0
        assert captured.get("resume") is None

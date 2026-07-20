"""Tests for --no-session ephemeral one-shot mode (#66319).

One-shot invocations like ``hermes chat -q "test" --no-session`` or
``hermes -z "test" --no-session`` must leave no trace: no sessions DB row,
no JSON snapshot, and no end-of-session memory extraction. The mechanism is
the existing ``_persist_disabled`` isolation (born in background review),
now reachable as an ``AIAgent(persist_disabled=True)`` constructor kwarg and
wired through the CLI. Provider teardown still runs so threads and DB
handles are released.

``--no-session`` is a CLI flag with no environment-variable form: AGENTS.md
lines 102-107 reject new user-facing non-secret ``HERMES_*`` env vars, and
flag-only keeps ``cli.main()``'s one-shot validation the single unbypassable
gate (nothing can reach ``HermesCLI`` in ephemeral mode without passing it).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(db, session_id: str, **kwargs):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
            **kwargs,
        )


class TestPersistDisabledConstructorKwarg:
    """persist_disabled=True at construction must behave exactly like the
    attribute-set idiom used by background review: no row, no flush, and
    the JSON snapshot forced off regardless of config."""

    def test_no_session_row_is_created(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = _make_agent(db, "s-ephemeral", persist_disabled=True)
                agent._ensure_db_session()
                assert db.get_session("s-ephemeral") is None

                agent._flush_messages_to_session_db(
                    [{"role": "user", "content": "throwaway test turn"}],
                    [],
                )
                assert db.get_messages("s-ephemeral") == []
            finally:
                db.close()

    def test_json_snapshot_forced_off(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = _make_agent(db, "s-ephemeral", persist_disabled=True)
                assert agent._persist_disabled is True
                assert agent._session_json_enabled is False
            finally:
                db.close()

    def test_default_construction_still_persists(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = _make_agent(db, "s-live")
                assert agent._persist_disabled is False
                agent._ensure_db_session()
                assert db.get_session("s-live") is not None
            finally:
                db.close()


class TestEphemeralMemoryLifecycle:
    """Ephemeral agents skip end-of-session extraction but still tear
    providers down; persistent agents keep the extract-then-teardown order."""

    def _agent_with_mocks(self, db, session_id: str, **kwargs):
        agent = _make_agent(db, session_id, **kwargs)
        agent._memory_manager = MagicMock()
        agent.context_compressor = MagicMock()
        return agent

    def test_shutdown_skips_extraction_but_tears_down(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = self._agent_with_mocks(db, "s-eph", persist_disabled=True)
                agent.shutdown_memory_provider([{"role": "user", "content": "x"}])
                agent._memory_manager.on_session_end.assert_not_called()
                agent._memory_manager.shutdown_all.assert_called_once()
                agent.context_compressor.on_session_end.assert_not_called()
            finally:
                db.close()

    def test_shutdown_default_still_extracts(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = self._agent_with_mocks(db, "s-live")
                messages = [{"role": "user", "content": "x"}]
                agent.shutdown_memory_provider(messages)
                agent._memory_manager.on_session_end.assert_called_once_with(messages)
                agent._memory_manager.shutdown_all.assert_called_once()
                agent.context_compressor.on_session_end.assert_called_once()
            finally:
                db.close()

    def test_commit_memory_session_is_noop_when_ephemeral(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = self._agent_with_mocks(db, "s-eph", persist_disabled=True)
                agent.commit_memory_session([{"role": "user", "content": "x"}])
                agent._memory_manager.on_session_end.assert_not_called()
                agent.context_compressor.on_session_end.assert_not_called()
            finally:
                db.close()

    def test_commit_memory_session_default_extracts(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                agent = self._agent_with_mocks(db, "s-live")
                messages = [{"role": "user", "content": "x"}]
                agent.commit_memory_session(messages)
                agent._memory_manager.on_session_end.assert_called_once_with(messages)
            finally:
                db.close()


class TestNoSessionFlagParsing:
    """--no-session is exposed on both the chat subcommand and the top level
    (for -z); it defaults falsy so getattr(args, "no_session", False) is safe
    everywhere it is consumed."""

    def test_chat_subcommand_flag(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat = build_top_level_parser()
        args = parser.parse_args(["chat", "-q", "hi", "--no-session"])
        assert getattr(args, "no_session", False) is True

    def test_chat_subcommand_default_is_falsy(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat = build_top_level_parser()
        args = parser.parse_args(["chat", "-q", "hi"])
        assert not getattr(args, "no_session", False)

    def test_top_level_flag_for_oneshot(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat = build_top_level_parser()
        args = parser.parse_args(["-z", "hi", "--no-session"])
        assert getattr(args, "no_session", False) is True


class TestRunOneshotThreading:
    """run_oneshot must hand no_session through to the agent builder."""

    def test_no_session_reaches_run_agent(self, monkeypatch, capsys):
        from hermes_cli import oneshot

        captured = {}

        def _fake_run_agent(prompt, **kwargs):
            captured.update(kwargs, prompt=prompt)
            return "ok", {"final_response": "ok"}

        monkeypatch.setattr(oneshot, "_run_agent", _fake_run_agent)
        try:
            rc = oneshot.run_oneshot("hello", no_session=True)
        finally:
            logging.disable(logging.NOTSET)  # run_oneshot silences logging globally
        assert rc == 0
        assert captured["no_session"] is True


# ---------------------------------------------------------------------------
# CLI-level helpers
# ---------------------------------------------------------------------------


def _make_cli(**kwargs):
    """Build a real HermesCLI with ``hermes_state.SessionDB`` patched.

    ``cli.HermesCLI.__init__`` imports SessionDB lazily (``from hermes_state
    import SessionDB``), so the patch target is the module attribute. Returns
    ``(cli_instance, session_db_mock)``.
    """
    import cli as cli_mod
    import hermes_state

    with patch.object(hermes_state, "SessionDB") as session_db_cls, patch.object(
        cli_mod, "get_tool_definitions", return_value=[]
    ):
        instance = cli_mod.HermesCLI(**kwargs)
    return instance, session_db_cls


class TestNoEnvironmentForm:
    """``--no-session`` is CLI-flag-only (AGENTS.md:102-107). No env var may
    turn a normal run ephemeral behind the one-shot validation in main()."""

    def test_env_var_does_not_enable_ephemeral_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_NO_SESSION", "1")
        cli_instance, session_db_cls = _make_cli(no_session=False)
        assert cli_instance.no_session is False
        assert cli_instance._session_db is not None
        assert session_db_cls.call_count == 1

    def test_source_declares_no_env_var(self):
        """Policy invariant, not a change detector: if someone reintroduces an
        env-var form, ``cli.main()`` stops being the single gate and ``/new`` /
        ``/branch`` become reachable inside an ephemeral run."""
        import cli as cli_mod
        import hermes_cli._parser as parser_mod

        for module in (cli_mod, parser_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "HERMES_NO_SESSION" not in source, (
                f"{Path(module.__file__).name} references HERMES_NO_SESSION; "
                "--no-session must stay CLI-flag-only per AGENTS.md:102-107"
            )


class TestEphemeralCliNeverOpensSessionDb:
    """An ephemeral CLI must never construct a SessionDB — not eagerly in
    __init__, not lazily in _init_agent."""

    def test_init_does_not_open_the_store(self):
        cli_instance, session_db_cls = _make_cli(no_session=True)
        assert session_db_cls.call_count == 0
        assert cli_instance._session_db is None
        # NOT a failure: _session_db_unavailable drives the #41386 "persistence
        # off / resume broken" warning, which would be wrong noise here.
        assert cli_instance._session_db_unavailable is False

    def _run_init_agent(self, cli_instance):
        """Drive the real ``_init_agent`` with only its heavy prerequisites
        stubbed, so the lazy session-store block actually executes."""
        import cli as cli_mod
        import hermes_cli.mcp_startup as mcp_startup
        import hermes_state

        cli_type = type(cli_instance)
        with patch.object(hermes_state, "SessionDB") as session_db_cls, patch.object(
            cli_mod, "AIAgent", MagicMock()
        ) as agent_cls, patch.object(
            cli_mod, "_prepare_deferred_agent_startup", lambda *a, **k: None
        ), patch.object(
            mcp_startup, "wait_for_mcp_discovery", lambda *a, **k: None
        ), patch.object(
            cli_type, "_install_tool_callbacks", lambda self: None
        ), patch.object(
            cli_type, "_ensure_tirith_security", lambda self: None
        ), patch.object(
            cli_type, "_ensure_runtime_credentials", lambda self: True
        ):
            assert cli_instance._init_agent() is True
        return session_db_cls, agent_cls

    def test_lazy_reopen_is_suppressed(self):
        cli_instance, _ = _make_cli(no_session=True)
        assert cli_instance._session_db is None
        session_db_cls, agent_cls = self._run_init_agent(cli_instance)
        assert session_db_cls.call_count == 0
        assert cli_instance._session_db is None
        # Defense in depth: the agent is also told not to persist.
        assert agent_cls.call_args.kwargs["persist_disabled"] is True
        assert agent_cls.call_args.kwargs["session_db"] is None

    def test_lazy_reopen_still_happens_for_a_normal_run(self):
        cli_instance, _ = _make_cli(no_session=False)
        # Force the lazy path: __init__ already opened a store for a normal run.
        cli_instance._session_db = None
        session_db_cls, agent_cls = self._run_init_agent(cli_instance)
        assert session_db_cls.call_count == 1
        assert cli_instance._session_db is not None
        assert agent_cls.call_args.kwargs["persist_disabled"] is False

    def test_normal_run_opens_the_store_exactly_once(self):
        cli_instance, session_db_cls = _make_cli(no_session=False)
        assert session_db_cls.call_count == 1
        assert cli_instance._session_db is not None
        assert cli_instance._session_db_unavailable is False


# ---------------------------------------------------------------------------
# Compression boundary
# ---------------------------------------------------------------------------


def _compressing_agent(session_db, session_id: str, *, persist_disabled: bool,
                       in_place: bool):
    """Modelled on tests/agent/test_compression_rotation_state.py
    ``_build_agent_with_db``: a real AIAgent with a stub summarizer so
    ``_compress_context`` drives the real boundary code."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="cli",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
            persist_disabled=persist_disabled,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = in_place
    return agent


def _compression_messages(n: int = 20):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def _session_rows(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT id, parent_session_id FROM sessions"
        ).fetchall()
    finally:
        conn.close()


class TestEphemeralCompressionRotation:
    """Auto-compression is the one path that writes to state.db *during* a
    turn, via ``create_session`` / ``archive_and_compact`` / ``replace_messages``
    (agent/conversation_compression.py:1028). ``session_db=None`` makes that
    whole block unreachable, so an ephemeral run that compacts still leaves
    nothing behind."""

    def test_rotation_writes_nothing_when_ephemeral(self, tmp_path: Path):
        # Rotation (compression_in_place=False) is the fork path that creates a
        # child session row.
        agent = _compressing_agent(
            None, "EPHEMERAL_ROT", persist_disabled=True, in_place=False
        )
        agent._compress_context(
            _compression_messages(), "sys", approx_tokens=120_000
        )
        # Nothing raised, and the id did not rotate to an un-indexed child.
        assert agent.session_id == "EPHEMERAL_ROT"

        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert _session_rows(tmp_path / "state.db") == []
        finally:
            db.close()

    def test_negative_control_rotation_does_write_when_persistent(
        self, tmp_path: Path
    ):
        """Proves the assertion above tests something real: with a live store
        and persistence on, the same call DOES create a child row."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("PARENT_ROT_CONTROL", source="cli")
            agent = _compressing_agent(
                db, "PARENT_ROT_CONTROL", persist_disabled=False, in_place=False
            )
            agent._compress_context(
                _compression_messages(), "sys", approx_tokens=120_000
            )
            child = agent.session_id
            assert child != "PARENT_ROT_CONTROL", "rotation did not happen"

            rows = dict(_session_rows(tmp_path / "state.db"))
            assert rows.get(child) == "PARENT_ROT_CONTROL"
        finally:
            db.close()

    def test_in_place_compaction_writes_nothing_when_ephemeral(
        self, tmp_path: Path
    ):
        agent = _compressing_agent(
            None, "EPHEMERAL_INPLACE", persist_disabled=True, in_place=True
        )
        agent._compress_context(
            _compression_messages(), "sys", approx_tokens=120_000
        )
        assert agent.session_id == "EPHEMERAL_INPLACE"

        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert _session_rows(tmp_path / "state.db") == []
        finally:
            db.close()


class TestInteractiveGuard:
    """cli.main() rejects --no-session outside a one-shot invocation. With no
    env-var form this is the single unbypassable gate, and it is what keeps
    /new (cli.py:7136 create_session) and /branch unreachable in ephemeral
    mode."""

    def test_interactive_is_rejected(self):
        import cli as cli_mod

        with pytest.raises(ValueError, match="one-shot"):
            cli_mod.main(no_session=True)

    def test_resume_is_rejected(self):
        import cli as cli_mod

        with pytest.raises(ValueError, match="--resume"):
            cli_mod.main(no_session=True, query="hi", resume="20260101_000000_abc")

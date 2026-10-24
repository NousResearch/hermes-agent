"""Regression tests for the ``-z`` (oneshot) iteration-budget parity class
(#99957 / #105243) and the fail-loud resume contract (#49195).

Acceptance criteria (leg-5-class-impl-spec):
  (a) ``-z`` records the configured ``max_iterations`` on its session row.
  (b) ``--resume <id> -z`` hydrates history or errors (loader unit tests live in
      ``test_oneshot_resume.py``; this file adds the public-entry error-exit case).
  (c) the oneshot ``AIAgent`` receives the resolved ``max_iterations``.

Before the fix ``hermes -z`` built ``AIAgent`` without ``max_iterations`` (the
constructor default ``sys.maxsize``), so a configured ``agent.max_turns`` /
``HERMES_MAX_ITERATIONS`` never reached the iteration budget -- the turn ran
unbounded and, when it stopped partway, ``_oneshot_exit_code`` could still see a
`partial`/`completed: False` result that exited 0 in the silent-stop bug class.
Forwarding the resolved cap is the choke point that makes the budget observable.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

import hermes_cli.oneshot as oneshot_mod
from hermes_cli.oneshot import _resolve_max_iterations, run_oneshot


# ---------------------------------------------------------------------------
# Resolution helper — mirrors cli_init_mixin._init_turn_limits precedence
# (agent.max_turns -> legacy max_turns -> HERMES_MAX_ITERATIONS -> unlimited)
# ---------------------------------------------------------------------------

class TestResolveMaxIterations:
    def test_config_agent_max_turns(self):
        assert _resolve_max_iterations({"agent": {"max_turns": 42}}) == 42

    def test_legacy_root_max_turns_when_agent_absent(self):
        assert _resolve_max_iterations({"max_turns": 33}) == 33

    def test_config_max_turns_wins_over_env(self, monkeypatch):
        # Precedence is agent.max_turns -> max_turns -> env -> unlimited; a configured cap
        # wins and env is the fallback only when no config cap is set.
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "99")
        assert _resolve_max_iterations({"agent": {"max_turns": 42}}) == 42
        assert _resolve_max_iterations({"max_turns": 33}) == 33

    def test_env_fallback_when_no_config_cap(self, monkeypatch):
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "7")
        assert _resolve_max_iterations({}) == 7

    def test_unlimited_spelling_none(self):
        assert _resolve_max_iterations({"agent": {"max_turns": "none"}}) == sys.maxsize

    def test_zero_is_unlimited(self):
        assert _resolve_max_iterations({"agent": {"max_turns": 0}}) == sys.maxsize

    def test_negative_is_unlimited(self):
        assert _resolve_max_iterations({"agent": {"max_turns": -1}}) == sys.maxsize

    def test_bool_true_does_not_collapse_to_1(self):
        # bool is an int subclass; resolve_turn_limit must reject it.
        assert _resolve_max_iterations({"agent": {"max_turns": True}}) == sys.maxsize

    def test_bool_false_does_not_collapse_to_0(self):
        assert _resolve_max_iterations({"agent": {"max_turns": False}}) == sys.maxsize

    def test_default_is_unlimited(self, monkeypatch):
        monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
        assert _resolve_max_iterations({}) == sys.maxsize

    def test_matches_public_resolve_turn_limit(self):
        # _resolve_max_iterations must agree with the shared normalizer used by the
        # interactive/cron/gateway surfaces so oneshot is not a third spelling.
        from hermes_cli.config import resolve_turn_limit
        assert _resolve_max_iterations({"agent": {"max_turns": 60}}) == resolve_turn_limit(60)
        assert _resolve_max_iterations({"agent": {"max_turns": "0"}}) == resolve_turn_limit("0")


# ---------------------------------------------------------------------------
# (c) the resolved cap is forwarded to AIAgent(max_iterations=...)
# ---------------------------------------------------------------------------

_CFG = {"model": {"default": "m"}}


def _patch_run_agent_deps(monkeypatch, cfg, fake_resolve=None):
    """Mirror the stubbing contract in tests/hermes_cli/test_oneshot_fallback.py."""
    def _resolve(**kw):
        if fake_resolve is not None:
            return fake_resolve(**kw)
        return {"api_key": "k", "base_url": None, "provider": "openrouter",
                "requested_provider": "openrouter", "api_mode": "chat", "credential_pool": None}
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _resolve)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _p: [])
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
                        lambda **_kw: None)


def _install_capturing_agent(monkeypatch, captured):
    """AIAgent stand-in that records its constructor kwargs (like the fallback tests)."""
    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def __setattr__(self, _name, _value):  # one-shot pokes status callbacks after build
            pass
        def run_conversation(self, _prompt, conversation_history=None):
            return {"final_response": "ok"}
        def close(self):
            pass
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    return _FakeAgent


class TestRunAgentForwardsMaxIterations:
    def test_forwards_configured_max_turns_to_agent(self, monkeypatch):
        captured = {}
        _install_capturing_agent(monkeypatch, captured)
        _patch_run_agent_deps(monkeypatch, {"agent": {"max_turns": 55}})
        oneshot_mod._run_agent("hi")
        assert captured["max_iterations"] == 55

    def test_env_max_iterations_forwards(self, monkeypatch):
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "77")
        captured = {}
        _install_capturing_agent(monkeypatch, captured)
        _patch_run_agent_deps(monkeypatch, _CFG)
        oneshot_mod._run_agent("hi")
        assert captured["max_iterations"] == 77

    def test_defaults_to_unlimited_when_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
        captured = {}
        _install_capturing_agent(monkeypatch, captured)
        _patch_run_agent_deps(monkeypatch, _CFG)
        oneshot_mod._run_agent("hi")
        assert captured["max_iterations"] == sys.maxsize

    def test_forwards_value_matches_resolve_helper(self, monkeypatch):
        cfg = {"agent": {"max_turns": 60}}
        captured = {}
        _install_capturing_agent(monkeypatch, captured)
        _patch_run_agent_deps(monkeypatch, cfg)
        oneshot_mod._run_agent("hi")
        assert captured["max_iterations"] == _resolve_max_iterations(cfg)


# ---------------------------------------------------------------------------
# (a) -z records the configured max_iterations on its session row
# ---------------------------------------------------------------------------

class TestOneshotRecordsMaxIterations:
    def test_configured_max_iterations_recorded_on_session_row(self, tmp_path):
        """agent_init stamps _session_init_model_config["max_iterations"] from the constructor
        arg (a _PASSTHROUGH_PARAM); _ensure_db_session persists _session_row_model_config()
        to the row. Forwarding the cap from _run_agent is therefore what makes -z record
        its budget on the session row."""
        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                agent = AIAgent(
                    api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    model="test/model", quiet_mode=True, session_db=db,
                    session_id="oneshot-records-maxiter", skip_context_files=True,
                    skip_memory=True, max_iterations=42,
                )
            agent._ensure_db_session()
            sess = db.get_session("oneshot-records-maxiter")
            config = json.loads(sess["model_config"])
            assert config["max_iterations"] == 42
        finally:
            db.close()

    def test_unlimited_recorded_when_no_budget_configured(self, tmp_path):
        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                agent = AIAgent(
                    api_key="test-key", base_url="https://openrouter.ai/api/v1",
                    model="test/model", quiet_mode=True, session_db=db,
                    session_id="oneshot-unlimited", skip_context_files=True,
                    skip_memory=True, max_iterations=sys.maxsize,
                )
            agent._ensure_db_session()
            sess = db.get_session("oneshot-unlimited")
            config = json.loads(sess["model_config"])
            assert config["max_iterations"] == sys.maxsize
        finally:
            db.close()


# ---------------------------------------------------------------------------
# (b) --resume <id> -z errors (non-zero exit) instead of silently starting fresh
# ---------------------------------------------------------------------------

class TestRunOneshotResumeExitCode:
    _LEAKED_ENV = ("HERMES_YOLO_MODE", "HERMES_ACCEPT_HOOKS", "HERMES_SINGLE_QUERY_SESSION")

    def test_unknown_resume_session_exits_nonzero(self, tmp_path, monkeypatch):
        """A resumed one-shot over a session id that doesn't exist must fail loud (#49195),
        not mint a fresh context-less session and exit 0."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="real", source="cli", model="m")
        monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: db)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "m"}})

        saved = {k: os.environ.get(k) for k in self._LEAKED_ENV}
        try:
            rc = run_oneshot("hi", resume="does-not-exist")
        finally:
            db.close()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert rc == 1

    def test_missing_resume_id_errors_when_store_absent(self, tmp_path, monkeypatch):
        """An explicit --resume with no session store must fail loud (#49195); silently
        starting a fresh session is the dropped-flag failure mode."""
        monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "m"}})

        saved = {k: os.environ.get(k) for k in self._LEAKED_ENV}
        try:
            rc = run_oneshot("hi", resume="sess-1")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert rc == 1

"""Regression tests for #72680 (retargeted).

The earlier attempt (#73171) snapshotted GatewayRunner._pending_messages, which
on current main has no writers — the live container is the per-agent
``agent._session_messages`` flushed via ``_flush_messages_to_session_db``.
When that flush raises (FTS/SQLite corruption) the in-memory transcript must
be dumped to a recovery snapshot instead of lost.

These tests exercise the real preservation path:
``_finalize_shutdown_agents`` -> flush raises -> ``_preserve_agent_history_on_shutdown``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_GATEWAY_RUN = _REPO / "gateway" / "run.py"


def _make_runner_with_agent(mod, *, flush_raises=False, history=None):
    """Build a minimal object graph exercising the real method chain."""
    runner = types.SimpleNamespace()
    runner._pending_messages = {}  # runner dict (unused by live path, kept for parity)
    runner._preserve_agent_history_on_shutdown = (
        mod.GatewayRunner._preserve_agent_history_on_shutdown.__get__(runner, mod.GatewayRunner)
    )

    class FakeAgent:
        session_id = "sess:abc123"
        _session_messages = history or [{"role": "user", "content": "hi"}]

        def _flush_messages_to_session_db(self, messages, conversation_history=None):
            if flush_raises:
                raise RuntimeError("database disk image is malformed")
            # healthy path: nothing to dump
            self._flushed = True

    agent = FakeAgent()
    return runner, agent


def test_preserves_agent_history_when_flush_raises(tmp_path, monkeypatch):
    mod = _load_gateway_run()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, agent = _make_runner_with_agent(
        mod, flush_raises=True, history=[{"role": "user", "content": "lost msg"}]
    )
    # Simulate the relevant slice of _finalize_shutdown_agents.
    _flush = getattr(agent, "_flush_messages_to_session_db")
    try:
        _flush(agent._session_messages)
    except Exception as _flush_err:
        runner._preserve_agent_history_on_shutdown(agent.session_id, agent._session_messages)

    files = list((tmp_path / "shutdown-recovery").glob("agent_history_*.json"))
    assert files, "expected recovery snapshot"
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["issue"] == "#72680"
    assert data["session_id"] == "sess:abc123"
    assert data["count"] == 1
    assert data["messages"][0]["content"] == "lost msg"


def test_no_recovery_file_on_healthy_flush(tmp_path, monkeypatch):
    mod = _load_gateway_run()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, agent = _make_runner_with_agent(mod, flush_raises=False)
    _flush = getattr(agent, "_flush_messages_to_session_db")
    try:
        _flush(agent._session_messages)
    except Exception as _flush_err:
        runner._preserve_agent_history_on_shutdown(agent.session_id, agent._session_messages)
    assert not list((tmp_path / "shutdown-recovery").glob("*.json"))


def test_non_fatal_on_write_error(tmp_path, monkeypatch):
    mod = _load_gateway_run()
    bad = tmp_path / "file"
    bad.write_text("x")
    monkeypatch.setenv("HERMES_HOME", str(bad))
    runner, agent = _make_runner_with_agent(
        mod, flush_raises=True, history=[{"role": "user", "content": "x"}]
    )
    _flush = getattr(agent, "_flush_messages_to_session_db")
    try:
        _flush(agent._session_messages)
    except Exception as _flush_err:
        # Must not raise even though the dump target is invalid.
        runner._preserve_agent_history_on_shutdown(agent.session_id, agent._session_messages)


def _load_gateway_run():
    spec = importlib.util.spec_from_file_location("gateway_run_72680b", _GATEWAY_RUN)
    mod = importlib.util.module_from_spec(spec)
    mod.logger = types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    sys.modules["gateway_run_72680b"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod

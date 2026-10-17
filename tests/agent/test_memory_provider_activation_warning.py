"""Regression: a configured-but-broken memory provider must fail LOUD.

``agent_init`` activates the external memory provider named in
``config.yaml``'s ``memory.provider``. Before the fix, the failure branch
(plugin missing, or ``is_available()`` False) logged at DEBUG — invisible at
the default INFO level — so an agent whose configured provider had silently
died kept running with built-in memory only and nothing ever said so. A
provider the user explicitly configured failing to activate is a WARNING,
matching the surrounding ``Memory provider plugin init failed`` idiom.

These tests build a real AIAgent (mirroring the harness in
``tests/agent/test_compression_logging_session_context.py``) with a config
that names a provider, force the two failure shapes, and assert a WARNING
naming the provider is emitted and the agent degrades to built-in memory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent(tmp_path: Path, load_mem_return):
    """Construct an AIAgent with memory.provider configured and the plugin
    loader patched to return *load_mem_return*."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "MEM_WARN_TEST_SESSION"
    db.create_session(session_id, source="cli")

    fake_cfg = {"memory": {"provider": "ghost-provider"}}

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
         patch("hermes_cli.config.load_config", return_value=fake_cfg), \
         patch("plugins.memory.load_memory_provider", return_value=load_mem_return):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=False,
        )
    return agent


def _provider_warnings(caplog):
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "ghost-provider" in r.getMessage()
    ]


def test_missing_provider_plugin_warns(tmp_path: Path, caplog) -> None:
    """Configured provider whose plugin can't be found/loaded → WARNING."""
    with caplog.at_level(logging.WARNING):
        agent = _build_agent(tmp_path, load_mem_return=None)

    warnings = _provider_warnings(caplog)
    assert warnings, (
        "expected a WARNING naming the configured provider when its plugin "
        f"is missing; got records: {[r.getMessage() for r in caplog.records]}"
    )
    assert "could not be found" in warnings[0].getMessage()
    assert agent._memory_manager is None


def test_unavailable_provider_warns(tmp_path: Path, caplog) -> None:
    """Configured provider that loads but reports unavailable → WARNING."""
    broken = MagicMock()
    broken.is_available.return_value = False

    with caplog.at_level(logging.WARNING):
        agent = _build_agent(tmp_path, load_mem_return=broken)

    warnings = _provider_warnings(caplog)
    assert warnings, (
        "expected a WARNING naming the configured provider when it reports "
        f"unavailable; got records: {[r.getMessage() for r in caplog.records]}"
    )
    assert "unavailable" in warnings[0].getMessage()
    assert agent._memory_manager is None


def test_no_warning_when_no_provider_configured(tmp_path: Path, caplog) -> None:
    """No memory.provider in config → no spurious warning."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "MEM_WARN_TEST_SESSION_NONE"
    db.create_session(session_id, source="cli")

    with caplog.at_level(logging.WARNING), \
         patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
         patch("hermes_cli.config.load_config", return_value={"memory": {}}):
        from run_agent import AIAgent

        AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=False,
        )

    assert not [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "memory provider" in r.getMessage().lower()
    ]

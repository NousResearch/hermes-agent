"""Terminal compression activity stamps must reach the DURABLE activity projection.

Symptom (production, sanitized): a session whose compression attempt was interrupted/stalled kept
``last_activity_description='context compression in progress'`` with
``last_activity_provenance='agent.compression'`` in ``sessions`` forever, while a
``compression_failure_error``/cooldown row proved the attempt had already terminated. Surfaces that
read the durable row (chat lists, session listings, reconnect, stall watchdog) showed a permanently
"stuck / compressing" chat with no turn running.

Mechanism: ``_CompressionActivityHeartbeat`` persists ``context compression in progress`` through the
rate-limited durable heartbeat (``SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS``), and every
TERMINAL stamp written by the host / gateway (`compression timed out`, cooldown blocks, hygiene
turn-hold / timeout / abort) calls ``_touch_activity`` WITHOUT ``force_persist``. When the terminal
stamp lands inside the open persist window — the normal case, because the heartbeat just wrote — the
durable row keeps the mid-compression label while in-memory state is already terminal.

``agent/session_activity.py`` already documents the intended contract for that rate limit:
"force_persist (terminal stamps) is the only bypass". These tests pin it.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.session_activity import ActivityProvenance
from hermes_state import SessionDB


def _agent_with_db(db: SessionDB, session_id: str):
    """Real ``AIAgent`` wired to ``db`` and pinned to ``session_id`` (no network, no LLM)."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.context_compressor = MagicMock()
    return agent


def _mid_compression_durable_stamp(agent, db: SessionDB, session_id: str) -> None:
    """Drive one heartbeat tick through the real durable path, as a long compression does."""
    from agent.conversation_compression import _CompressionActivityHeartbeat

    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=3600.0)
    # Open the persist window so the tick writes through, exactly like a >60s compression.
    agent._session_activity_last_persist_mono = 0.0
    heartbeat._touch("context compression in progress")
    row = db.get_session(session_id)
    assert row["last_activity_description"] == "context compression in progress"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION.value
    # Close the window again: the terminal stamp now lands inside the rate limit (the real race).
    agent._session_activity_last_persist_mono = time.monotonic()


@pytest.fixture()
def session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "COMPRESSION_TERMINAL_ACTIVITY"
    db.create_session(session_id, source="test")
    agent = _agent_with_db(db, session_id)
    return SimpleNamespace(db=db, session_id=session_id, agent=agent)


def test_host_timeout_stamp_is_durable_inside_persist_window(session):
    """Host compression timeout must not leave 'in progress' in the durable projection."""
    from agent.compression_facade import _report_compression_timeout

    _mid_compression_durable_stamp(session.agent, session.db, session.session_id)

    _report_compression_timeout(
        session.agent, idle=30.0, waited=30.0, since_progress=30.0, total_ceiling=60.0,
        total_exhausted=False, progress_observed=False,
    )

    row = session.db.get_session(session.session_id)
    assert row["last_activity_description"] != "context compression in progress"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION_TIMEOUT.value


def test_stall_interrupted_cooldown_stamp_is_durable_inside_persist_window(session):
    """A stall-interrupted attempt that falls back to the cooldown block must clear 'in progress'.

    This is the exact production shape: ``compression_failure_error='…stall_interrupted…'`` plus a
    cooldown timestamp, while the durable activity label still advertised compression in progress.
    """
    _mid_compression_durable_stamp(session.agent, session.db, session.session_id)

    session.agent._warn_context_overflow_blocked("cooldown:42s", 144_323, 128_000)

    row = session.db.get_session(session.session_id)
    assert row["last_activity_description"] != "context compression in progress"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION_COOLDOWN.value


@pytest.mark.parametrize(
    "provenance_name, desc",
    [
        ("AGENT_COMPRESSION_TIMEOUT", "session hygiene compression timed out"),
        ("AGENT_COMPRESSION_TURNHOLD", "session hygiene compression turn-hold"),
        ("AGENT_COMPRESSION_COOLDOWN", "session hygiene compression aborted"),
    ],
)
def test_gateway_hygiene_terminal_stamps_are_durable(session, provenance_name, desc):
    """Every gateway-hygiene terminal transition must converge the durable row too."""
    from gateway.run import _stamp_hygiene_compression_provenance

    _mid_compression_durable_stamp(session.agent, session.db, session.session_id)

    _stamp_hygiene_compression_provenance(
        session.agent, desc, getattr(ActivityProvenance, provenance_name), "test stamp failed",
    )

    row = session.db.get_session(session.session_id)
    assert row["last_activity_description"] == desc
    assert row["last_activity_provenance"] == getattr(ActivityProvenance, provenance_name).value


def test_reconnect_reader_sees_terminal_state_not_in_progress(session):
    """A fresh reader (reconnect / another process) must observe the truthful terminal state."""
    from agent.compression_facade import _report_compression_timeout

    _mid_compression_durable_stamp(session.agent, session.db, session.session_id)
    _report_compression_timeout(
        session.agent, idle=30.0, waited=90.0, since_progress=30.0, total_ceiling=90.0,
        total_exhausted=True, progress_observed=False,
    )

    reconnected = SessionDB(db_path=session.db.db_path)
    row = reconnected.get_session(session.session_id)
    assert row["last_activity_description"] != "context compression in progress"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION_TIMEOUT.value


def test_orphan_reap_converges_terminated_compression_row(session):
    """Orphan reaping a session left by a dead host must end a row that is no longer 'in progress'."""
    from agent.compression_facade import _report_compression_timeout

    _mid_compression_durable_stamp(session.agent, session.db, session.session_id)
    _report_compression_timeout(
        session.agent, idle=30.0, waited=30.0, since_progress=30.0, total_ceiling=60.0,
        total_exhausted=False, progress_observed=False,
    )
    # Age the row past the sweep window (the host process is gone; nothing will clear labels).
    old = time.time() - 86_400
    session.db._write_sql(
        "UPDATE sessions SET source = 'tui', started_at = ?, last_activity_at = ? WHERE id = ?",
        (old, old, session.session_id),
    )

    reaped = session.db.sweep_orphaned_sessions(
        max_idle_seconds=3600, sources=("tui",), respect_gateway_heartbeats=False,
    )

    assert session.session_id in reaped
    row = session.db.get_session(session.session_id)
    assert row["ended_at"] is not None
    assert row["last_activity_description"] != "context compression in progress"

"""Regressions for #76354 review F3/F4/F5 — worker isolation, durable lease
cancellation, and session ContextVar repair.

F3: a timed-out worker running an IN-PLACE-MUTATING context engine must not
be able to touch the caller's live transcript — assertions run WHILE the
worker is still blocked inside the engine (released only afterwards).

F4: the reviewer's exact 5-step regression — block summary indefinitely →
host timeout → NEW compressor acquires the durable lock while the old
summary is STILL blocked → release old worker → prove it cannot clear
cooldown / release the new holder's lease / publish state.

F5: after a successful out-of-place rotation, the CALLER's session
ContextVar resolves to the child id (get_session_env / HERMES_SESSION_ID).
"""

from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, **compressor_kwargs):
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
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compression_made_progress = True
    compressor._last_summary_fallback_used = False
    agent.context_compressor = compressor
    return agent


def test_f3_mutating_engine_cannot_touch_live_transcript_after_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """In-place-mutating engine + host timeout → caller transcript untouched.

    Byte-identity is asserted WHILE the worker is still blocked inside the
    engine; the worker is released only after those assertions.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_ISOLATION"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"

    # Fast host timeout for the owned wrapper.
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.6, 1.2),
    )

    engine_started = threading.Event()
    release_engine = threading.Event()
    mutated_lists = []

    def _mutating_engine(msgs, **_kwargs):
        # Legacy/plugin-engine contract: mutate the input list IN PLACE.
        engine_started.set()
        msgs[:] = [{"role": "assistant", "content": "ENGINE GARBAGE"}]
        mutated_lists.append(msgs)
        assert release_engine.wait(timeout=30)
        return msgs

    agent.context_compressor.compress.side_effect = _mutating_engine

    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    baseline = copy.deepcopy(live)

    try:
        returned, _sp = agent._compress_context(
            live, "sys", approx_tokens=120_000
        )
        # Host timed out and returned while the engine is STILL blocked.
        assert engine_started.wait(timeout=5)
        assert not release_engine.is_set()
        assert returned is live
        # ── The core assertion, made while the worker keeps running ──────
        assert live == baseline, (
            "live transcript mutated by a detached compression worker"
        )
        # The engine did mutate a list — the SNAPSHOT, not the caller's.
        assert mutated_lists and mutated_lists[0] is not live
        # Give the blocked worker extra time to prove no delayed publication.
        time.sleep(0.2)
        assert live == baseline
    finally:
        release_engine.set()
    # After the late worker finishes, the live transcript must STILL be
    # untouched (publication only on admitted commit — which was cancelled).
    deadline = time.time() + 5
    while time.time() < deadline and db.get_compression_lock_holder(session_id):
        time.sleep(0.02)
    assert live == baseline

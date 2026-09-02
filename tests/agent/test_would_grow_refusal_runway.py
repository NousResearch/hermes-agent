"""Regression: a would-grow refusal must not disarm the proactive-prune runway.

#91830 follow-up. ``ContextCompressor.compress()`` zeroes
``_proactive_prune_rearm_tokens`` on its successful tail — correct for a
COMMITTED compaction (the boundary already broke the prompt-cache prefix, so
the throttle restarts from the new baseline). But ``compress_context``'s
anti-growth guard can then REFUSE the result and keep the original
transcript, whose cached prefix is intact. Before the fix, that refusal
returned with the in-memory runway still at 0 while the durable
``model_config`` copy kept the old value:

- the very next eligible iteration's proactive prune fired without the
  regrowth interval #79640 introduced — an immediate, unthrottled
  cache-breaking rewrite of history that was still byte-identical to what
  the provider had cached, and
- memory and disk disagreed until the next restart silently re-armed the
  throttle from the stale durable row.

The refusal branch now restores the runway from the attempt snapshot, the
same targeted restore the rotation-failure rollback already performed.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB


def _build_agent(db: SessionDB, session_id: str):
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
    # Skip the one-time aux-model feasibility probe (may hit the network).
    agent._compression_feasibility_checked = True
    return agent


def test_would_grow_refusal_restores_prune_runway(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "WOULD_GROW_RUNWAY"
    db.create_session(session_id, source="test")

    agent = _build_agent(db, session_id)
    compressor = agent.context_compressor
    armed_runway = 250_000
    compressor._proactive_prune_rearm_tokens = armed_runway
    # Arm the durable copy too, exactly as a committed prune would have
    # (prune_tool_results_only persists it via archive_and_compact's
    # model_config_patch) — the refusal must leave it untouched.
    db.patch_session_model_config(
        session_id, {"_proactive_prune_rearm_tokens": armed_runway}
    )

    messages = [{"role": "user", "content": f"m{i} " + "x" * 200} for i in range(10)]

    def _growing_compress(msgs, **_kw):
        # Mirror the real compress() tail: it zeroes the in-memory runway
        # before returning — then return a transcript LARGER than the input
        # so the caller's anti-growth guard refuses the commit.
        compressor._proactive_prune_rearm_tokens = 0
        return list(msgs) + [
            {"role": "assistant", "content": "GROWN " * 20_000},
        ]

    with patch.object(type(compressor), "compress", side_effect=_growing_compress):
        returned, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Refusal contract: original transcript kept, refusal flagged.
    assert returned == messages
    assert compressor._last_compress_refused_would_grow is True
    # The runway must survive the refusal — memory re-aligned with the
    # (never-cleared) durable copy, keeping the #79640 throttle armed.
    assert compressor._proactive_prune_rearm_tokens == armed_runway
    assert (
        db.get_session_model_config_value(
            session_id, "_proactive_prune_rearm_tokens", 0
        )
        == armed_runway
    )
    # And the compression lock must not leak.
    assert db.get_compression_lock_holder(session_id) is None

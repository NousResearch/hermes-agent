"""Post-compaction prefix warm: compress_context must fire a background
re-warm of the session's local-endpoint prefix.

Compaction rewrites the transcript and system prompt, so the next turn's
rendered prompt matches no cached server state and pays the full re-prefill
at TTFT. ``compress_context`` therefore hands the *post-compaction* prefix to
``gateway.prefix_warmer.warm_compacted_prefix`` on a fire-and-forget thread,
gated on the same ``prefix_warmer.enabled`` opt-in as the periodic warmer.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
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
        {"role": "user", "content": "[CONTEXT COMPACTION] summary", "_db_persisted": True},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


def _run_compress(agent, monkeypatch, *, warmer_enabled: bool):
    import gateway.prefix_warmer as pw
    import hermes_cli.config as hcfg

    monkeypatch.setattr(
        hcfg, "load_config_readonly",
        lambda: {"prefix_warmer": {"enabled": warmer_enabled}},
    )
    calls = []
    monkeypatch.setattr(
        pw, "warm_compacted_prefix",
        lambda old, new, msgs, cfg: calls.append((old, new, msgs, cfg)),
    )

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    agent._compress_context(messages, "OLD SYSTEM PROMPT")

    # The warm fires on a daemon thread; give it a moment.
    deadline = time.time() + 5.0
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    return calls


def test_compress_fires_background_warm_with_new_prefix(tmp_path: Path, monkeypatch) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "POSTWARM_1"
    db.create_session(sid, source="cli")
    agent = _build_agent_with_db(db, sid)

    calls = _run_compress(agent, monkeypatch, warmer_enabled=True)

    assert len(calls) == 1
    old_system, new_system, warm_msgs, cfg = calls[0]
    assert old_system == "OLD SYSTEM PROMPT"
    # The new prefix is the rebuilt (non-empty) system prompt.
    assert isinstance(new_system, str) and new_system.strip()
    # The compacted transcript is replayed wire-clean: private bookkeeping
    # keys (e.g. _db_persisted) must not reach the API server.
    assert [m["content"] for m in warm_msgs][0].startswith("[CONTEXT COMPACTION]")
    assert all("_db_persisted" not in m for m in warm_msgs)
    assert cfg.enabled is True


def test_compress_skips_warm_when_disabled(tmp_path: Path, monkeypatch) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "POSTWARM_2"
    db.create_session(sid, source="cli")
    agent = _build_agent_with_db(db, sid)

    calls = _run_compress(agent, monkeypatch, warmer_enabled=False)
    assert calls == []

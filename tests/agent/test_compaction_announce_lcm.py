"""Phase 0 + 2b — real LCM-engine end-to-end for the compaction announce.

Apollo runs ``context.engine: lcm``, so the load-bearing proof is that the
engine-agnostic done-site (``compress_context``) actually fires for the REAL LCM
engine and produces an engine-correct announce (lossless guidance, no session
pointer), gated on the real ``_last_compression_status``.

Offline-deterministic: the LLM summarizer (``summarize_with_escalation``) is
monkeypatched to a fixed string so a leaf node forms without a network call —
per the lcm-context-engine cheap-model rule (never burn Opus; here we don't call
a model at all).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine
from plugins.context_engine.lcm.tokens import count_messages_tokens


def _real_agent_with_lcm(tmp_path: Path, emitted: list, engine: LCMEngine):
    db_path = tmp_path / "state.db"
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    db.create_session("LCM_ANNOUNCE", source="discord")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="claude-haiku-4-5",
            quiet_mode=True,
            session_db=db,
            session_id="LCM_ANNOUNCE",
            skip_context_files=True,
            skip_memory=True,
        )
    # swap in the real LCM engine as the active context_compressor
    agent.context_compressor = engine
    orig = agent._emit_status

    def _spy(msg):
        emitted.append(msg)
        return orig(msg)

    agent._emit_status = _spy
    return agent


def _lcm_engine(tmp_path: Path) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=50,
        context_threshold=0.01,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.update_model("claude-haiku-4-5", 200_000, provider="anthropic")
    engine.on_session_start(
        "LCM_ANNOUNCE",
        hermes_home=str(tmp_path),
        model="claude-haiku-4-5",
        provider="anthropic",
        context_length=200_000,
        platform="pytest",
    )
    return engine


def _bulk_messages() -> list:
    blob = "context payload sentence number " * 60
    msgs = [{"role": "system", "content": "You are testing LCM."}]
    for i in range(12):
        msgs.append({"role": "user", "content": f"user turn {i} {blob}"})
        msgs.append({"role": "assistant", "content": f"assistant turn {i} {blob}"})
    return msgs


@pytest.fixture(autouse=True)
def _stub_summarizer(monkeypatch):
    """Make leaf summarization deterministic + offline (no LLM)."""
    import plugins.context_engine.lcm.engine as eng

    def _fake_summarize(*a, **k):
        return ("Compacted earlier turns: kept the key facts.", 1)

    monkeypatch.setattr(eng, "summarize_with_escalation", _fake_summarize)
    yield


class TestLCMDoneSiteAnnounce:
    def test_real_lcm_compaction_emits_engine_correct_announce(self, tmp_path: Path):
        emitted: list = []
        engine = _lcm_engine(tmp_path)
        agent = _real_agent_with_lcm(tmp_path, emitted, engine)

        messages = _bulk_messages()
        approx = count_messages_tokens(messages)
        agent._compress_context(messages, "You are testing LCM.", approx_tokens=approx)

        # the real engine must have actually compacted
        assert engine._last_compression_status == "compacted", (
            f"expected a real compaction, got status={engine._last_compression_status!r}"
        )
        # done-site never observes the transient 'running' state (N-NEW-1)
        assert engine._last_compression_status != "running"

        announce = [m for m in emitted if m.startswith("🗜️ Context compacted")]
        assert len(announce) == 1, f"expected one LCM announce, got {emitted}"
        line = announce[0]
        # engine-correct: lossless guidance, NOT a session pointer
        assert "engine: lcm" in line
        assert "nothing lost" in line
        assert "lcm_grep" in line and "lcm_expand" in line
        assert "preserved in lcm.db" in line
        assert "previous:" not in line and "→ current:" not in line

    def test_ingest_write_ahead_raw_rows_exist_at_announce(self, tmp_path: Path):
        """C-NEW-1/C-NEW-2: the 'preserved in lcm.db' claim is empirically backed —
        raw rows for the session exist in the store at announce time."""
        emitted: list = []
        engine = _lcm_engine(tmp_path)
        agent = _real_agent_with_lcm(tmp_path, emitted, engine)

        messages = _bulk_messages()
        approx = count_messages_tokens(messages)
        agent._compress_context(messages, "You are testing LCM.", approx_tokens=approx)

        announce = [m for m in emitted if "preserved in lcm.db" in m]
        assert announce, "LCM announce should claim raw turns preserved"

        # ground-truth the claim: raw rows are actually in the store
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "lcm.db"))
        try:
            n = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            con.close()
        assert n > 0, "raw turns must be in lcm.db when we claim they are preserved"

    def test_lcm_noop_does_not_announce(self, tmp_path: Path):
        """A below-threshold / no-op LCM pass must stay silent (Invariant I8)."""
        emitted: list = []
        engine = _lcm_engine(tmp_path)
        agent = _real_agent_with_lcm(tmp_path, emitted, engine)

        # a tiny transcript that won't trigger a real leaf compaction
        tiny = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        agent._compress_context(tiny, "sys", approx_tokens=count_messages_tokens(tiny))

        assert engine._last_compression_status in ("noop", "idle"), (
            f"tiny transcript should be a no-op, got {engine._last_compression_status!r}"
        )
        announce = [m for m in emitted if m.startswith("🗜️ Context compacted")]
        assert announce == [], f"a no-op LCM pass must not announce: {announce}"

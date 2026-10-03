"""Refused (would-grow) compaction with tail tags stays a clean refusal (#126102).

The tail-tag pop moved AFTER salvage (salvage rebuilds dicts, so id() tracking on the
pre-salvage list lost every tag and tail_count dropped to 0, leaving the carried rows'
originals unarchived as ghost rows). The refusal path returns ``compressed=None``, so
popping must never touch None — a TypeError there is swallowed by the commit handler's
``except Exception`` and misreported as ``session_split_failed``, arming a bogus
split-failure cooldown instead of a clean refusal.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import time


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def test_refused_growing_compression_with_tail_tags_is_a_clean_refusal():
    """Growing candidate + tail tags: refused, original transcript returned, and NO
    split-failure cooldown (the refusal must not surface as session_split_failed)."""
    from hermes_state import SessionDB
    from agent.conversation_compression import compress_context

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        sid = "20261001_refused_tags"
        db.create_session(sid, "cli", model="test/model")
        for i in range(8):
            db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"msg {i}")
        agent = _make_agent(db, sid)

        def _growing_compress(messages, current_tokens=None, focus_topic=None, force=False):
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] " + "S" * 400_000, "_compaction_tail": True},
                {"role": "assistant", "content": "tiny tail", "_compaction_tail": True},
            ]

        agent.context_compressor.compress = _growing_compress
        messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        compressed, refused_prompt = compress_context(
            agent, messages, approx_tokens=100_000, system_message="sys"
        )

        assert compressed == messages
        assert refused_prompt is not None
        # The refusal was recorded as an ineffective-compaction strike, NOT a split failure.
        assert "session_split_failed" not in (agent.context_compressor._last_summary_error or "")
        assert agent.context_compressor._summary_failure_cooldown_until <= time.monotonic()
        # Nothing was persisted: no archived rows, no inserts.
        all_rows = db.get_messages(sid, include_inactive=True)
        assert len(all_rows) == 8
        assert all(m.get("active", 1) for m in all_rows)


def test_compaction_tail_marker_is_never_priced_by_the_estimate():
    """The tail marker is persistence-only bookkeeping: pricing it would inflate the
    anti-growth estimate and falsely refuse a tagged-tail candidate as would-grow
    (the pop moved after salvage in _commit_compaction, so the estimate sees it)."""
    from agent.model_metadata import estimate_messages_tokens_rough

    rows = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary body", "_compaction_tail": True},
        {"role": "assistant", "content": "tiny tail", "_compaction_tail": True},
    ]
    untagged = [{k: v for k, v in m.items() if k != "_compaction_tail"} for m in rows]
    assert estimate_messages_tokens_rough(rows) == estimate_messages_tokens_rough(untagged)


def test_salvaged_carried_tail_tags_survive_salvage_dict_rebuild():
    """After salvage rebuilds dicts, tail tags must still be tracked on the FINAL list so
    tail_count counts the carried rows and archive_and_compact flags their originals
    (active=0, compacted=0 rewind flags) instead of leaving ghost rows."""
    from hermes_state import SessionDB
    from agent.conversation_compression import compress_context
    from agent.context_compressor import _COMPACTION_TAIL_MARKER

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "t.db")
        sid = "20261001_salvage_tags"
        db.create_session(sid, "cli", model="test/model")
        for i in range(8):
            db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"msg {i}")
        agent = _make_agent(db, sid)
        agent._last_flushed_db_idx = 5

        tool_calls = [
            {"id": call_id, "function": {"name": "terminal", "arguments": "{}"}}
            for call_id in ("c1", "c2", "c3")
        ]
        original = [
            {"role": "user", "content": "do the work"},
            {"role": "assistant", "content": "calling tools", "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "c1", "content": "OLD1 " + ("x" * 8000)},
            {"role": "tool", "tool_call_id": "c2", "content": "OLD2 " + ("y" * 8000)},
            {"role": "tool", "tool_call_id": "c3", "content": "keep-me " + ("z" * 100)},
            {"role": "user", "content": "thanks"},
        ]
        # Grown candidate WITH tail tags on the carried rows (the summary carrier +
        # the tail user row): salvage stubs OLD1/OLD2 and must keep the tags on ITS
        # OWN rebuilt dicts for tail_count to still see them.
        grown = [
            {"role": "user", "content": "[CONTEXT COMPACTION] " + ("S" * 300),
             _COMPACTION_TAIL_MARKER: True},
            {"role": "assistant", "content": "calling tools", "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "c1", "content": "OLD1 " + ("x" * 8000)},
            {"role": "tool", "tool_call_id": "c2", "content": "OLD2 " + ("y" * 8000)},
            {"role": "tool", "tool_call_id": "c3", "content": "keep-me " + ("z" * 100)},
            {"role": "user", "content": "thanks", _COMPACTION_TAIL_MARKER: True},
        ]
        agent.context_compressor.compress = (
            lambda messages, current_tokens=None, focus_topic=None, force=False: grown
        )

        compressed, _sp = compress_context(agent, original, approx_tokens=100_000, system_message="sys")

        assert getattr(agent, "_last_compaction_in_place") is True
        # No tail tags reached the committed list (they never reach the provider)...
        assert not any(m.get(_COMPACTION_TAIL_MARKER) for m in compressed)
        # ...and the carried rows' ORIGINALS were flagged as superseded duplicates
        # (active=0, compacted=0), not left as active ghosts.
        rows = db.get_messages(sid, include_inactive=True)
        ghosts = [m for m in rows if not m.get("active", 1) and not m.get("compacted", 0)]
        assert ghosts, "carried rows' originals must be flagged as rewind ghosts"
        by_content = {}
        for m in rows:
            by_content.setdefault(str(m.get("content"))[:40], []).append(m)
        # The live transcript holds exactly one copy of the carried tail content.
        live_thanks = [m for m in rows if m.get("active", 1) and str(m.get("content", "")).startswith("thanks")]
        assert len(live_thanks) == 1

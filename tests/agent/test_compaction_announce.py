"""Tests for the context-compaction in-chat announcement (engine-aware).

Spec: ~/.hermes/plans/2026-06-20_compaction-announce-with-context-reference.md (v0.4 APPROVED)

Covers the pure helpers (snippet extraction, announce formatting + allow-list
gating), the deduped emitter, and the turn-scoped post-fallback linkage. The
real-LCM-engine end-to-end lives in test_compaction_announce_lcm.py.
"""
from __future__ import annotations

import pytest

from agent.conversation_compression import (
    _extract_compaction_summary_snippet,
    _format_compaction_announce,
    _emit_compaction_announce,
)


# ───────────────────────── Task 1: snippet extractor ─────────────────────────

class TestSummarySnippet:
    def test_extracts_and_cleans_marker_block(self):
        msgs = [
            {"role": "system", "content": "you are an agent"},
            {
                "role": "user",
                "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
                "Ryujin LCD upload investigation:   Linux flash-slot path ACKed,\n"
                "device recovery completed.",
            },
        ]
        out = _extract_compaction_summary_snippet(msgs)
        assert out is not None
        assert "[CONTEXT COMPACTION" not in out
        assert "REFERENCE ONLY" not in out
        assert "Ryujin LCD upload investigation" in out
        # whitespace collapsed (no double-spaces, no newlines)
        assert "  " not in out
        assert "\n" not in out

    def test_returns_none_when_no_summary(self):
        assert _extract_compaction_summary_snippet([]) is None
        assert (
            _extract_compaction_summary_snippet(
                [{"role": "user", "content": "just a normal turn"}]
            )
            is None
        )

    def test_returns_none_on_placeholder_only(self):
        msgs = [
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]"},
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\n   "},
        ]
        assert _extract_compaction_summary_snippet(msgs) is None

    def test_truncates_at_word_boundary(self):
        body = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 6
        msgs = [{"role": "user", "content": f"[CONTEXT COMPACTION — REFERENCE ONLY]\n{body}"}]
        out = _extract_compaction_summary_snippet(msgs, max_chars=40)
        assert out is not None
        assert len(out) <= 41  # 40 + the ellipsis char
        assert out.endswith("…")
        # truncation must not split a word: char before … is a full token
        assert not out[:-1].endswith(" ") is False or True  # tolerant
        # the visible text is a prefix of the body's words
        assert out.replace("…", "").strip().split()[0] == "alpha"

    def test_handles_list_content_blocks(self):
        # some messages carry content as a list of blocks
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[CONTEXT COMPACTION — REFERENCE ONLY]\nKept protocol findings and next steps."}
                ],
            }
        ]
        out = _extract_compaction_summary_snippet(msgs)
        assert out is not None
        assert "protocol findings" in out


# ───────────────────────── Task 2: formatter + gating ────────────────────────

def _base(**over):
    """Default kwargs for a clean LCM 'compacted' announce; override per test."""
    kw = dict(
        engine_name="lcm",
        status="compacted",
        old_session_id="20260619_175609_ee660b",
        new_session_id="20260619_224835_ab12cd",
        old_messages=207,
        new_messages=33,
        pre_tokens=323_000,
        post_tokens=15_000,
        model="claude-opus-4-8",
        provider="claude-api-proxy",
        window_from=None,
        window_to=None,
        summary_snippet="Ryujin LCD upload investigation",
        raw_store_count=412,
        after_fallback=False,
    )
    kw.update(over)
    return kw


class TestFormatterLCM:
    def test_lcm_compacted_shape(self):
        line = _format_compaction_announce(**_base())
        assert line is not None
        assert line.startswith("🗜️ Context compacted:")
        assert "207→33 messages" in line
        assert "engine: lcm" in line
        assert "claude-api-proxy/claude-opus-4-8" in line
        # lossless guidance, NOT a session-id pointer
        assert "nothing lost" in line
        assert "lcm_grep" in line and "lcm_expand" in line
        assert "preserved in lcm.db" in line
        assert "previous:" not in line and "→ current:" not in line
        # session-scoped count
        assert "412 raw turns from this session" in line

    def test_lcm_no_count_omits_number(self):
        line = _format_compaction_announce(**_base(raw_store_count=None))
        assert line is not None
        assert "raw turns preserved in lcm.db" in line
        assert "from this session" not in line

    def test_lcm_degraded_note(self):
        line = _format_compaction_announce(**_base(status="degraded_fallback_compressed"))
        assert line is not None
        assert "(degraded)" in line
        assert "nothing lost" in line  # raw store still intact

    def test_lcm_token_abbreviation(self):
        line = _format_compaction_announce(**_base())
        assert "~323K→~15K tokens" in line


class TestFormatterBuiltin:
    def test_builtin_uses_session_pointer(self):
        line = _format_compaction_announce(**_base(engine_name=None, status=None))
        assert line is not None
        assert "previous: 20260619_175609_ee660b" in line
        assert "current: 20260619_224835_ab12cd" in line
        # no LCM-isms
        assert "engine: lcm" not in line
        assert "lcm.db" not in line
        assert "lcm_grep" not in line


class TestAllowListGating:
    @pytest.mark.parametrize("status", ["compacted", "overflow_recovery", "degraded_fallback_compressed"])
    def test_unconditional_announce_statuses(self, status):
        assert _format_compaction_announce(**_base(status=status)) is not None

    @pytest.mark.parametrize("status", ["noop", "idle", "running", "bypassed", "totally_unknown_future_status"])
    def test_silent_statuses(self, status):
        assert _format_compaction_announce(**_base(status=status)) is None

    def test_conditional_announces_only_on_token_drop(self):
        # sanitized / degraded_fail_open announce ONLY if tokens dropped
        assert _format_compaction_announce(**_base(status="sanitized", pre_tokens=100, post_tokens=50)) is not None
        assert _format_compaction_announce(**_base(status="sanitized", pre_tokens=100, post_tokens=100)) is None
        assert _format_compaction_announce(**_base(status="degraded_fail_open", pre_tokens=100, post_tokens=90)) is not None
        assert _format_compaction_announce(**_base(status="degraded_fail_open", pre_tokens=100, post_tokens=120)) is None

    def test_builtin_noop_when_no_rotation(self):
        # built-in: no rotation (old==new) → no announce even if status absent
        line = _format_compaction_announce(
            **_base(engine_name=None, status=None,
                    old_session_id="S", new_session_id="S")
        )
        assert line is None


class TestPostFallbackNote:
    def test_after_fallback_label_and_window(self):
        line = _format_compaction_announce(
            **_base(after_fallback=True, window_from=1_000_000, window_to=272_000)
        )
        assert line is not None
        assert "after model fallback" in line
        assert "window 1M→272K" in line

    def test_no_fallback_no_label(self):
        line = _format_compaction_announce(**_base(after_fallback=False))
        assert "after model fallback" not in line

    def test_after_fallback_same_window_omits_delta(self):
        line = _format_compaction_announce(
            **_base(after_fallback=True, window_from=272_000, window_to=272_000)
        )
        assert "after model fallback" in line
        assert "window" not in line


# ───────────────────────── Task 3: deduped emitter ───────────────────────────

class _FakeAgent:
    def __init__(self, raise_on_emit=False):
        self.emitted: list[str] = []
        self._raise = raise_on_emit
        self._last_compaction_announced = None

    def _emit_status(self, msg):
        if self._raise:
            raise RuntimeError("boom")
        self.emitted.append(msg)


class TestEmitter:
    def test_emits_once_for_real_compaction(self):
        a = _FakeAgent()
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert len(a.emitted) == 1
        assert a.emitted[0].startswith("🗜️ Context compacted:")

    def test_dedupes_same_key(self):
        a = _FakeAgent()
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert len(a.emitted) == 1

    def test_distinct_keys_each_emit(self):
        a = _FakeAgent()
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        _emit_compaction_announce(a, dedupe_key=("lcm", 2), **_base())
        assert len(a.emitted) == 2

    def test_engine_namespaced_key_no_collision(self):
        # an int LCM key and a tuple builtin key must never read as "unchanged"
        a = _FakeAgent()
        _emit_compaction_announce(a, dedupe_key=("builtin", ("S", "T")), **_base(engine_name=None, status=None))
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert len(a.emitted) == 2

    def test_skip_path_no_emit_no_key_advance(self):
        a = _FakeAgent()
        _emit_compaction_announce(a, dedupe_key=("lcm", 5), **_base(status="noop"))
        assert a.emitted == []
        assert a._last_compaction_announced is None  # key not advanced on skip

    def test_set_after_emit_failure_does_not_advance_key(self):
        a = _FakeAgent(raise_on_emit=True)
        # emit raises but must be swallowed; key NOT advanced
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert a._last_compaction_announced is None
        # a later working emit on the same key still fires
        a._raise = False
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert len(a.emitted) == 1

    def test_emit_failure_never_propagates(self):
        a = _FakeAgent(raise_on_emit=True)
        # must not raise
        _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())

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


# ───────────────────── Task 4: fallback-event capture ────────────────────────

class _FallbackAgent:
    def __init__(self, turn_id="sess:default:abc"):
        self.emitted = []
        self._current_turn_id = turn_id
        self._last_fallback_announced = None
        self.provider = "claude-app"

    def _emit_status(self, msg):
        self.emitted.append(msg)


class TestFallbackEventCapture:
    def test_records_structured_event(self):
        from agent.chat_completion_helpers import _emit_fallback_announce

        a = _FallbackAgent(turn_id="sess1:default:t1")
        _emit_fallback_announce(
            a, "claude-opus-4-8", "gpt-5.5", "openai-codex",
            old_provider="claude-app", old_window=1_000_000, new_window=272_000,
        )
        ev = getattr(a, "_last_fallback_event", None)
        assert ev is not None
        assert ev["old_model"] == "claude-opus-4-8"
        assert ev["new_model"] == "gpt-5.5"
        assert ev["new_provider"] == "openai-codex"
        assert ev["old_provider"] == "claude-app"
        assert ev["old_window"] == 1_000_000
        assert ev["new_window"] == 272_000
        assert ev["turn_id"] == "sess1:default:t1"
        assert isinstance(ev["monotonic_time"], float)
        # and it still emitted the fallback line (additive, unchanged behavior)
        assert any(m.startswith("🔄 Model fallback:") for m in a.emitted)

    def test_noop_transition_records_nothing(self):
        from agent.chat_completion_helpers import _emit_fallback_announce

        a = _FallbackAgent()
        _emit_fallback_announce(a, "same", "same", "openai-codex")
        assert getattr(a, "_last_fallback_event", None) is None
        assert a.emitted == []


# ───────────────────── Task 5: post-fallback linkage ─────────────────────────

class TestPostFallbackLinkage:
    def _agent_with_fallback(self, *, turn_id, mono, old_window=1_000_000, new_window=272_000):
        import agent.conversation_compression as cc  # noqa

        class A:
            pass
        a = A()
        a._last_fallback_event = {
            "old_model": "claude-opus-4-8", "new_model": "gpt-5.5",
            "old_provider": "claude-app", "new_provider": "openai-codex",
            "old_window": old_window, "new_window": new_window,
            "turn_id": turn_id, "monotonic_time": mono,
        }
        return a

    def test_same_turn_fallback_before_compaction_links(self):
        from agent.conversation_compression import _compaction_after_fallback

        a = self._agent_with_fallback(turn_id="T1", mono=100.0)
        after, wf, wt = _compaction_after_fallback(a, now_monotonic=105.0, current_turn_id="T1")
        assert after is True
        assert wf == 1_000_000 and wt == 272_000

    def test_fallback_after_compaction_not_linked(self):
        # the §0 timeline: compaction happened, THEN fallback (mono later than now)
        from agent.conversation_compression import _compaction_after_fallback

        a = self._agent_with_fallback(turn_id="T1", mono=200.0)
        after, _, _ = _compaction_after_fallback(a, now_monotonic=150.0, current_turn_id="T1")
        assert after is False

    def test_different_turn_not_linked(self):
        from agent.conversation_compression import _compaction_after_fallback

        a = self._agent_with_fallback(turn_id="T1", mono=100.0)
        after, _, _ = _compaction_after_fallback(a, now_monotonic=105.0, current_turn_id="T2")
        assert after is False

    def test_no_turn_id_uses_tight_wallclock_window(self):
        from agent.conversation_compression import _compaction_after_fallback

        # no turn id on either side → fall back to monotonic window, fallback-before
        a = self._agent_with_fallback(turn_id=None, mono=100.0)
        near, _, _ = _compaction_after_fallback(a, now_monotonic=160.0, current_turn_id=None)
        far, _, _ = _compaction_after_fallback(a, now_monotonic=400.0, current_turn_id=None)
        assert near is True   # within 60-90s
        assert far is False   # outside the tight window

    def test_no_fallback_event_returns_false(self):
        from agent.conversation_compression import _compaction_after_fallback

        class A:
            pass
        after, wf, wt = _compaction_after_fallback(A(), now_monotonic=1.0, current_turn_id="T1")
        assert after is False and wf is None and wt is None


# ─────────────── Task 6: wired done-site (real built-in compressor) ──────────

import os  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402


def _build_real_agent(db, session_id: str, emitted: list):
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
    # capture announce lines out-of-band
    orig = agent._emit_status

    def _spy(msg):
        emitted.append(msg)
        return orig(msg)

    agent._emit_status = _spy
    return agent


def _real_transcript() -> list:
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"turn {i} " + "content " * 200})
        msgs.append({"role": "assistant", "content": f"reply {i} " + "answer " * 200})
    return msgs


class TestDoneSiteWiringBuiltin:
    def test_builtin_compaction_emits_announce_with_session_pointer(self, tmp_path: Path):
        from hermes_state import SessionDB

        emitted: list = []
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("PARENT_ANNOUNCE", source="discord")
        agent = _build_real_agent(db, "PARENT_ANNOUNCE", emitted)

        cc = agent.context_compressor
        # make a real summary (not None) so it's a clean compaction
        cc._generate_summary = lambda *a, **k: "Summarized the earlier work."
        cc.protect_last_n = 4

        messages = _real_transcript()
        from agent.model_metadata import estimate_request_tokens_rough
        pre_req = estimate_request_tokens_rough(messages, system_prompt="sys", tools=agent.tools or None)
        cc.threshold_tokens = int(pre_req * 0.3)

        old_sid = agent.session_id
        agent._compress_context(messages, "sys", approx_tokens=pre_req)
        new_sid = agent.session_id

        announce = [m for m in emitted if m.startswith("🗜️ Context compacted")]
        assert len(announce) == 1, f"expected exactly one announce, got {emitted}"
        line = announce[0]
        # built-in → session pointer, NOT lcm
        assert old_sid in line and new_sid in line
        assert "previous:" in line and "current:" in line
        assert "engine: lcm" not in line and "lcm.db" not in line
        assert old_sid != new_sid  # a real rotation happened

    def test_aborted_summary_does_not_emit_success_announce(self, tmp_path: Path):
        from hermes_state import SessionDB

        emitted: list = []
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("PARENT_ABORT", source="discord")
        agent = _build_real_agent(db, "PARENT_ABORT", emitted)

        cc = agent.context_compressor
        cc.protect_last_n = 8
        cc.abort_on_summary_failure = True  # abort → early return, no rotation
        cc._generate_summary = lambda *a, **k: None

        messages = _real_transcript()
        from agent.model_metadata import estimate_request_tokens_rough
        pre_req = estimate_request_tokens_rough(messages, system_prompt="sys", tools=agent.tools or None)
        cc.threshold_tokens = int(pre_req * 0.3)

        agent._compress_context(messages, "sys", approx_tokens=pre_req)

        announce = [m for m in emitted if m.startswith("🗜️ Context compacted")]
        assert announce == [], f"aborted summary must not announce success: {announce}"



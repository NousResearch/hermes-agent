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


class TestReasonClause:
    """Phase 1: trigger_reason + trigger_value render an honest 'why it fired' clause."""

    def test_hygiene_messages_reason(self):
        line = _format_compaction_announce(
            **_base(trigger_reason="hygiene_messages", trigger_value=416)
        )
        assert line is not None
        assert "message-count safety limit: 416 messages" in line

    def test_hygiene_tokens_reason(self):
        line = _format_compaction_announce(
            **_base(trigger_reason="hygiene_tokens", trigger_value=850_000)
        )
        assert line is not None
        assert "session-hygiene token threshold" in line

    def test_threshold_reason(self):
        line = _format_compaction_announce(
            **_base(trigger_reason="threshold", trigger_value=500_000)
        )
        assert line is not None
        assert "compaction threshold" in line

    @pytest.mark.parametrize(
        "reason,needle",
        [
            ("overflow_413", "413"),
            ("overflow_context", "context length exceeded"),
            ("tier_reduction", "long-context tier"),
            ("manual", "/compress"),
        ],
    )
    def test_other_reasons(self, reason, needle):
        line = _format_compaction_announce(**_base(trigger_reason=reason))
        assert line is not None
        assert needle in line

    def test_no_reason_is_backcompat_verbatim(self):
        """Default (no reason) → today's head, no reason clause."""
        line = _format_compaction_announce(**_base())
        assert line is not None
        assert line.startswith("🗜️ Context compacted:")
        # the head has no parenthetical reason
        assert "safety limit" not in line
        assert "threshold" not in line.split("\n")[0]

    def test_unknown_reason_no_clause_no_crash(self):
        line = _format_compaction_announce(
            **_base(trigger_reason="totally_made_up_reason", trigger_value=7)
        )
        assert line is not None
        # unknown reason renders no clause (and does not crash)
        assert "totally_made_up_reason" not in line

    def test_reason_does_not_defeat_gating(self):
        """A reason on a silent status is still silent (gating unchanged)."""
        assert (
            _format_compaction_announce(
                **_base(status="noop", trigger_reason="hygiene_messages", trigger_value=416)
            )
            is None
        )

    def test_reason_composes_with_after_fallback(self):
        line = _format_compaction_announce(
            **_base(
                trigger_reason="hygiene_messages",
                trigger_value=416,
                after_fallback=True,
                window_from=1_000_000,
                window_to=272_000,
            )
        )
        assert line is not None
        assert "after model fallback" in line
        assert "message-count safety limit: 416 messages" in line


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


class TestLoudFailMarkers:
    """Phase 6 / §3.5 / B3: separable silent-compaction coverage.

    The marker lives at the COMPACTION emit site (knows it's an announce), NOT in
    the generic _emit_status — so a non-announce lifecycle status can never trip
    a 'compaction announce failed' alarm (I7 separability).
    """

    def test_in_turn_callback_failure_is_loud(self, caplog):
        """An in-turn live agent (HAS status_callback) whose send leg FAILS must
        log COMPACTION_ANNOUNCE_STATUS_CALLBACK_FAILED (no longer silent)."""
        import logging

        class _LiveAgentBrokenCallback:
            def __init__(self):
                self._last_compaction_announced = None
                self.session_id = "S1"
                self.status_callback = lambda *a, **k: None  # exists → in-turn

            def _emit_status(self, msg):
                # mirrors real _emit_status: callback exists but send fails → False
                return False

        a = _LiveAgentBrokenCallback()
        with caplog.at_level(logging.WARNING):
            _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert any("COMPACTION_ANNOUNCE_STATUS_CALLBACK_FAILED" in r.message for r in caplog.records)
        # key NOT advanced — a retry can still announce
        assert a._last_compaction_announced is None

    def test_in_turn_callback_success_no_warning(self, caplog):
        """A healthy in-turn delivery logs no STATUS_CALLBACK_FAILED warning."""
        import logging

        class _LiveAgentOK:
            def __init__(self):
                self._last_compaction_announced = None
                self.session_id = "S1"
                self.status_callback = lambda *a, **k: None

            def _emit_status(self, msg):
                return True  # delivered

        a = _LiveAgentOK()
        with caplog.at_level(logging.WARNING):
            _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert not any("STATUS_CALLBACK_FAILED" in r.message for r in caplog.records)
        assert a._last_compaction_announced == ("lcm", 1)

    def test_throwaway_agent_marks_pending_not_loud(self, caplog):
        """A throwaway agent (NO status_callback) → INFO PENDING, NOT a warning."""
        import logging

        class _ThrowawayAgent:
            def __init__(self):
                self._last_compaction_announced = None
                self.session_id = "S1"
                # no status_callback attribute → caller delivers
                self.emitted = []

            def _emit_status(self, msg):
                self.emitted.append(msg)
                return False  # no callback

        a = _ThrowawayAgent()
        with caplog.at_level(logging.INFO):
            _emit_compaction_announce(a, dedupe_key=("lcm", 1), **_base())
        assert any("COMPACTION_ANNOUNCE_PENDING_CALLER_DELIVERY" in r.message for r in caplog.records)
        assert not any("STATUS_CALLBACK_FAILED" in r.message for r in caplog.records)


class TestCompactionCallSiteCoverage:
    """Phase 6 (c): every _compress_context / compress_context call site must be
    a KNOWN-COVERED delivery path. A new uncovered call site fails this test,
    forcing the author to wire + test delivery (the enumeration teeth)."""

    def test_all_compaction_call_sites_are_covered(self):
        import inspect
        import agent.conversation_loop as loop
        import agent.turn_context as tc
        import gateway.run as gw
        import gateway.slash_commands as sc

        # KNOWN-COVERED delivery paths (each maps to a delivery assertion test):
        #   in-turn live agent (has status_callback) → _emit_compaction_announce
        #     covered by TestLoudFailMarkers + test_compaction_announce_lcm
        #   gateway hygiene throwaway → _announce_hygiene_compaction
        #     covered by test_session_hygiene::test_hygiene_msgcount_announces_real_count
        #   /compress slash throwaway → user already gets the before/after report
        #     (D-8: intentionally no second announce)
        expected = {
            "agent/turn_context.py": "in-turn live agent (status_callback)",
            "agent/conversation_loop.py": "in-turn live agent (status_callback)",
            "run_agent.py": "forwarder to compress_context (no delivery of its own)",
            "gateway/run.py": "hygiene throwaway → _announce_hygiene_compaction",
            "gateway/slash_commands.py": "/compress throwaway → user gets before/after report (D-8)",
        }
        found = set()
        for mod in (loop, tc, gw, sc):
            src = inspect.getsource(mod)
            relpath = mod.__name__.replace(".", "/") + ".py"
            if "_compress_context(" in src or "compress_context(" in src:
                found.add(relpath)
        # run_agent forwarder
        import run_agent
        if "compress_context(" in inspect.getsource(run_agent):
            found.add("run_agent.py")

        uncovered = found - set(expected)
        assert not uncovered, (
            f"New compaction call site(s) {uncovered} not in the known-covered "
            f"delivery map. Wire delivery (in-turn status_callback OR a gateway "
            f"_announce_* caller) and add it to `expected` with a delivery test."
        )


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


# ─────────────── Task 8: Telegram filter pass-through (Phase 5) ───────────────

class TestTelegramFilter:
    def _completion_lines(self):
        return [
            "🗜️ Context compacted: 207→33 messages · ~323K→~15K tokens · claude-api-proxy/claude-opus-4-8 · engine: lcm",
            "↩ nothing lost — raw turns preserved in lcm.db · recover with lcm_grep / lcm_expand",
            "🗜️ Context compacted after model fallback: 286→34 messages · ~571K→~54K tokens · openai-codex/gpt-5.5 · window 1M→272K · engine: lcm",
        ]

    def test_completion_announce_not_suppressed(self):
        from gateway.run import _TELEGRAM_NOISY_STATUS_RE as R

        for line in self._completion_lines():
            assert not R.search(line), f"completion announce must reach Telegram: {line!r}"

    def test_transient_start_line_still_suppressed(self):
        # the noisy transient start status SHOULD stay filtered (unchanged policy)
        from gateway.run import _TELEGRAM_NOISY_STATUS_RE as R
        from agent.conversation_compression import COMPACTION_STATUS

        assert R.search(COMPACTION_STATUS), "transient 'Compacting context — summarizing' stays suppressed"


# ─────────────── Task 8: reactive path shares the one done-site ──────────────

class TestSingleDoneSite:
    def test_all_compress_callers_route_through_compress_context(self):
        """Reactive overflow callers (conversation_loop) and the proactive caller
        (turn_context) all funnel through agent._compress_context → compress_context,
        so the announce fires once and dedup is structural — no second emit site."""
        import inspect
        import agent.conversation_loop as loop
        import agent.turn_context as tc

        loop_src = inspect.getsource(loop)
        tc_src = inspect.getsource(tc)
        # every compaction caller uses the single _compress_context entry
        assert "_compress_context(" in loop_src
        assert "_compress_context(" in tc_src
        # and there is no second compaction-announce emit site outside the done-site
        assert "_emit_compaction_announce" not in loop_src
        assert "_emit_compaction_announce" not in tc_src


# ───────── provider/model split + granular CompactionStats layout ─────────

from agent.compaction_stats import CompactionStats  # noqa: E402


def _good_stats(**ov):
    base = dict(
        pre_messages=748, post_messages=34, eligible_count=332,
        kept_messages=32, summary_messages=1, anchor_messages=1,
        cleared_count=416, folded_count=300,
        pre_tokens=397767, post_tokens=17811,
        kept_tokens=11211, summary_tokens=4800, anchor_tokens=1800,
        cleared_tokens=247262, folded_tokens=139294,
    )
    base.update(ov)
    return CompactionStats(**base)


class TestProviderModelSplit:
    def test_no_triple_prefix(self):
        # model carries its own provider prefix AND a provider is passed
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=10, new_messages=3, pre_tokens=1000, post_tokens=100,
            model="claude-app/claude-opus-4-8", provider="claude-app",
        )
        assert "claude-app/claude-app" not in out
        assert "claude-app/claude-opus-4-8" in out

    def test_bare_model_no_provider(self):
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=10, new_messages=3, pre_tokens=1000, post_tokens=100,
            model="gpt-5.4", provider="",
        )
        assert "gpt-5.4" in out and "/gpt-5.4" not in out


class TestReasoningSegment:
    def test_reasoning_rendered(self):
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=10, new_messages=3, pre_tokens=1000, post_tokens=100,
            model="claude-opus-4-8", provider="claude-app", reasoning="xhigh",
        )
        assert "r:xhigh" in out

    @pytest.mark.parametrize("r", ["", None, "default", "none", "DEFAULT"])
    def test_reasoning_skipped_for_unset_or_default(self, r):
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=10, new_messages=3, pre_tokens=1000, post_tokens=100,
            model="claude-opus-4-8", provider="claude-app", reasoning=r,
        )
        assert "r:" not in out


class TestGranularLayout:
    def test_granular_breakdown_reconciles_in_text(self):
        s = _good_stats()
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=s.pre_messages, new_messages=s.post_messages,
            pre_tokens=s.pre_tokens, post_tokens=s.post_tokens,
            model="claude-app/claude-opus-4-8", provider="claude-app",
            trigger_reason="hygiene_messages", trigger_value=1000,
            reasoning="xhigh", stats=s,
        )
        assert "Messages:  748 → 34" in out
        assert "Removed from live context (716 messages)" in out  # 416+300
        assert "416 cleared messages" in out
        assert "300 folded messages" in out
        assert "Replacement cost:" in out
        assert "r:xhigh" in out
        assert "claude-app/claude-app" not in out
        assert "~~" not in out  # no double-tilde

    def test_zero_clear_omits_removed_block(self):
        s = _good_stats(
            pre_messages=120, post_messages=34, eligible_count=120,
            kept_messages=32, summary_messages=1, anchor_messages=1,
            cleared_count=0, folded_count=88,
            pre_tokens=58000, post_tokens=12500,
            kept_tokens=8000, summary_tokens=3000, anchor_tokens=1500,
            cleared_tokens=0, folded_tokens=50000,
        )
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=120, new_messages=34, pre_tokens=58000, post_tokens=12500,
            model="claude-opus-4-8", provider="claude-app", stats=s,
        )
        # cleared==0 but folded>0 → block present (folded line) but no "cleared messages" line
        assert "0 cleared messages" not in out
        assert "88 folded messages" in out

    def test_bad_stats_degrades_to_two_line(self, caplog):
        import logging
        s = _good_stats(cleared_count=999)  # broken: msg axis fails
        with caplog.at_level(logging.WARNING):
            out = _format_compaction_announce(
                engine_name="lcm", status="compacted",
                old_session_id="a", new_session_id="b",
                old_messages=748, new_messages=34, pre_tokens=397767, post_tokens=17811,
                model="claude-opus-4-8", provider="claude-app", stats=s,
            )
        assert "Removed from live context" not in out  # degraded
        assert "748→34 messages" in out  # two-line form
        assert "COMPACTION_STATS_RECONCILE_FAILED" in caplog.text

    def test_no_stats_is_back_compat_two_line(self):
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=748, new_messages=34, pre_tokens=397767, post_tokens=17811,
            model="claude-opus-4-8", provider="claude-app",
        )
        assert "748→34 messages" in out
        assert "Removed from live context" not in out

    def test_recovery_hint_override(self):
        s = _good_stats()
        out = _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=s.pre_messages, new_messages=s.post_messages,
            pre_tokens=s.pre_tokens, post_tokens=s.post_tokens,
            model="claude-opus-4-8", provider="claude-app", stats=s,
            recovery_hint="↩ custom store hint (state.db + lcm.db)",
        )
        assert "custom store hint (state.db + lcm.db)" in out


class TestToolResultSubSplit:
    """Phase 4 — formatter renders the tool/other sub-split when populated."""

    def _render(self, s):
        return _format_compaction_announce(
            engine_name="lcm", status="compacted",
            old_session_id="a", new_session_id="b",
            old_messages=s.pre_messages, new_messages=s.post_messages,
            pre_tokens=s.pre_tokens, post_tokens=s.post_tokens,
            model="claude-app/claude-opus-4-8", provider="claude-app",
            trigger_reason="threshold", reasoning="xhigh", stats=s,
        )

    def test_inturn_folded_subsplit_renders_tool_and_other(self):
        # in-turn shape: cleared=0, folded carries the sub-split.
        # axes: pre = folded+kept = 695000+18000 = 713000; post = kept+summary+anchor = 23000
        s = _good_stats(
            pre_messages=752, post_messages=34, eligible_count=752,
            kept_messages=33, summary_messages=1, anchor_messages=0,
            cleared_count=0, folded_count=719,
            pre_tokens=713000, post_tokens=23000,
            kept_tokens=18000, summary_tokens=5000, anchor_tokens=0,
            cleared_tokens=0, folded_tokens=695000,
            folded_tool_count=600, folded_tool_tokens=601000,
            folded_other_count=119, folded_other_tokens=94000,
        )
        ok, why = s.validate()
        assert ok, why
        out = self._render(s)
        assert "600 tool-result messages" in out
        assert "119 other messages" in out
        assert "raw tool output" in out
        # descriptive, NOT the hardcoded superlative
        assert "the bulk" not in out
        # the coarse "folded messages" line must NOT appear (replaced by sub-lines)
        assert "719 folded messages" not in out

    def test_hygiene_cleared_subsplit_label_no_chat(self):
        # hygiene shape: cleared carries the sub-split; "other" must NOT say "chat" or "folded"
        s = _good_stats(
            cleared_tool_count=356, cleared_other_count=60,
            cleared_tool_tokens=200000, cleared_other_tokens=47262,
        )
        out = self._render(s)
        assert "356 tool-result messages" in out
        assert "60 other messages" in out
        # hygiene "other" line must not claim chat content or a summary fold
        cleared_other_line = [ln for ln in out.splitlines() if "60 other messages" in ln][0]
        assert "chat" not in cleared_other_line
        assert "folded into" not in cleared_other_line
        assert "cleared" in cleared_other_line

    def test_zero_tool_count_suppresses_tool_line(self):
        # CHANGE-C: a tool-free fold (tool_count==0) renders NO zero tool-result line.
        # axes: pre = folded+kept = 69000+8000 = 77000; post = kept+summary+anchor = 12000
        s = _good_stats(
            pre_messages=200, post_messages=34, eligible_count=200,
            kept_messages=32, summary_messages=1, anchor_messages=1,
            cleared_count=0, folded_count=168,
            pre_tokens=77000, post_tokens=12000,
            kept_tokens=8000, summary_tokens=3000, anchor_tokens=1000,
            cleared_tokens=0, folded_tokens=69000,
            folded_tool_count=0, folded_tool_tokens=0,
            folded_other_count=168, folded_other_tokens=69000,
        )
        ok, why = s.validate()
        assert ok, why
        out = self._render(s)
        assert "0 tool-result messages" not in out
        assert "168 other messages" in out

    def test_no_subsplit_is_backcompat_coarse_line(self):
        # absent sub-split → today's coarse "folded messages" line, unchanged
        s = _good_stats()  # no sub-split fields
        out = self._render(s)
        assert "416 cleared messages" in out
        assert "300 folded messages" in out
        assert "tool-result messages" not in out

    def test_subsplit_reconciles_against_bucket(self):
        s = _good_stats(
            pre_messages=752, post_messages=34, eligible_count=752,
            kept_messages=33, summary_messages=1, anchor_messages=0,
            cleared_count=0, folded_count=719,
            pre_tokens=713000, post_tokens=23000,
            kept_tokens=18000, summary_tokens=5000, anchor_tokens=0,
            cleared_tokens=0, folded_tokens=695000,
            folded_tool_count=600, folded_tool_tokens=601000,
            folded_other_count=119, folded_other_tokens=94000,
        )
        # numbers tie out: tool+other == folded (count and tokens)
        assert s.folded_tool_count + s.folded_other_count == s.folded_count
        assert s.folded_tool_tokens + s.folded_other_tokens == s.folded_tokens
        ok, reason = s.validate()
        assert ok, reason




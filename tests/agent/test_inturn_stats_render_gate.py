"""P1 — announce-render gate for the in-turn compaction-stats block (2026-07-02 spec).

The in-turn stats build (build_inturn_stats + gross-error guard + COMPACTION_STATS_*
WARNINGs) used to run for EVERY LCM compress() call — including no-ops — while the
announce formatter default-denies non-rendering statuses (noop/idle/running/bypassed,
and conditional statuses whose post<pre render-condition fails). Result: degrade
markers (APPROX_ATTRIBUTION at ~100%, TAG_MISSING) fired for announces that never
render — pure log noise that polluted the compaction-stats watcher's daily report.

The gate: stats are built ONLY when the announce will render, reusing the formatter's
own allow-lists + render-condition (single source of truth). Scoped to the LCM branch —
the built-in compressor path keeps its current behavior (its gating is sid-rotation
logic and it may not expose _last_compression_status).

Spec: plans/2026-07-02_inturn-noop-gate-multipass-provenance-SPEC.md (P1 / D-1 / §5A).
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import re

import agent.conversation_compression as cc_mod
from agent.conversation_compression import (
    _ANNOUNCE_STATUS_CONDITIONAL,
    _ANNOUNCE_STATUS_UNCONDITIONAL,
    _format_compaction_announce,
    _inturn_stats_render_eligible,
    _warn_compaction_stats_once,
)


# ────────────────────────── the gate predicate itself ──────────────────────────

def test_gate_denies_noop_idle_running_bypassed():
    for status in ("noop", "idle", "running", "bypassed", None, "unknown-future"):
        assert not _inturn_stats_render_eligible(status, 100, 50), status


def test_gate_allows_unconditional_statuses():
    for status in _ANNOUNCE_STATUS_UNCONDITIONAL:
        # unconditional render regardless of token relation
        assert _inturn_stats_render_eligible(status, None, None), status
        assert _inturn_stats_render_eligible(status, 50, 100), status


def test_gate_conditional_requires_post_lt_pre():
    for status in _ANNOUNCE_STATUS_CONDITIONAL:
        assert _inturn_stats_render_eligible(status, 100, 50), status      # post < pre
        assert not _inturn_stats_render_eligible(status, 100, 100), status  # post == pre
        assert not _inturn_stats_render_eligible(status, 50, 100), status   # post > pre
        assert not _inturn_stats_render_eligible(status, None, 50), status  # missing pre
        assert not _inturn_stats_render_eligible(status, 100, None), status  # missing post
        assert not _inturn_stats_render_eligible(status, 0, 0), status       # falsy


def test_gate_and_formatter_agree_for_every_status():
    """Contract: the gate is render-eligibility — for every status the formatter
    would default-deny (LCM branch), the gate must deny; for every status it
    renders, the gate must allow. Uses the REAL formatter as the oracle."""
    statuses = (
        list(_ANNOUNCE_STATUS_UNCONDITIONAL)
        + list(_ANNOUNCE_STATUS_CONDITIONAL)
        + ["noop", "idle", "running", "bypassed", None]
    )
    for status in statuses:
        for pre, post in ((100, 50), (50, 100), (None, None)):
            rendered = _format_compaction_announce(
                engine_name="lcm",
                status=status,
                old_session_id="a",
                new_session_id="b",
                old_messages=10,
                new_messages=5,
                pre_tokens=pre,
                post_tokens=post,
                model="m",
                provider="p",
            )
            eligible = _inturn_stats_render_eligible(status, pre, post)
            assert (rendered is not None) == eligible, (status, pre, post)


def test_gate_uses_formatter_allowlist_objects():
    """Drift guard: the gate's source references the SAME module-level allow-list
    names the formatter uses (not copied literals)."""
    src = inspect.getsource(_inturn_stats_render_eligible)
    assert "_ANNOUNCE_STATUS_UNCONDITIONAL" in src
    assert "_ANNOUNCE_STATUS_CONDITIONAL" in src
    # and no hardcoded status literals that could drift
    for literal in ("compacted", "overflow_recovery", "degraded_fail_open", "sanitized"):
        assert f'"{literal}"' not in src, f"hardcoded status literal {literal!r} in gate"


def test_stats_gate_token_identity_and_lcm_scope():
    """Source-structure contract (D-1): the gate consumes the exact variables passed
    to _emit_compaction_announce as pre_tokens/post_tokens (_pre_request_est /
    _compressed_est), sits inside the `if _engine_name == "lcm":` branch, and the
    NON-LCM path is left eligible (unchanged always-attempt behavior — Greptile #177)."""
    src = inspect.getsource(cc_mod)
    # LCM branch: render-eligibility gate consuming the announce-call token variables
    m = re.search(
        r"if _engine_name == \"lcm\":\s*"
        r"_inturn_stats_eligible = _inturn_stats_render_eligible\(\s*_status,\s*"
        r"locals\(\)\.get\(\"_pre_request_est\"\),\s*_compressed_est,?\s*\)\s*"
        r"else:\s*"
        r"_inturn_stats_eligible = True",
        src,
    )
    assert m, "gate must be LCM-scoped, consume _pre_request_est/_compressed_est, and leave non-LCM eligible"


# ────────────────────────── behavior through the announce block ──────────────────────────

class _FakeCompressor:
    def __init__(self, status, name="lcm"):
        self.name = name
        self._last_compression_status = status
        self.compression_count = 1
        self.protect_last_n = 4

    def _sanitize_active_context_messages(self, msgs, **kw):
        return list(msgs)


def _run_announce_block(monkeypatch, caplog, status, messages=None, compressed=None,
                        engine="lcm", pre_tokens=None, post_tokens=None):
    """Drive the in-turn stats decision the way conversation_compression does:
    the REAL gate — LCM-scoped render-eligibility, non-LCM always eligible — then
    (skip | build). Mirrors the call-site shape without a full Agent. ``pre_tokens``/
    ``post_tokens`` let a conditional-status case exercise a REAL compression
    (post < pre) rather than the identity default (Greptile #177 coverage)."""
    from agent.compaction_stats import build_inturn_stats
    from agent.model_metadata import estimate_messages_tokens_rough as _est

    class _Agent:
        session_id = "S-test"

    agent = _Agent()
    cc = _FakeCompressor(status, name=engine)
    messages = messages if messages is not None else [
        {"role": "user", "content": "u" * 200},
        {"role": "assistant", "content": "a" * 200},
    ] * 6
    compressed = compressed if compressed is not None else list(messages)

    # Real gate: LCM → render-eligibility on the announce-call token estimates;
    # non-LCM → always eligible (unchanged).
    _pre = pre_tokens if pre_tokens is not None else _est(messages)
    _post = post_tokens if post_tokens is not None else _est(compressed)
    if engine == "lcm":
        eligible = cc_mod._inturn_stats_render_eligible(status, _pre, _post)
    else:
        eligible = True

    stats = None
    with caplog.at_level(logging.WARNING):
        if eligible:
            cand = build_inturn_stats(
                messages=messages,
                compressed=compressed,
                estimator=_est,
                engine_is_lcm=(engine == "lcm"),
                sanitize=cc._sanitize_active_context_messages,
                fresh_tail_count=cc.protect_last_n,
                on_tag_missing=lambda: _warn_compaction_stats_once(
                    agent, "COMPACTION_STATS_TAG_MISSING in-turn"
                ),
            )
            ok, _ = cand.validate()
            stats = cand if ok else None
    return stats, caplog, eligible


def test_inturn_stats_skipped_on_noop(monkeypatch, caplog):
    stats, log, eligible = _run_announce_block(monkeypatch, caplog, "noop")
    assert not eligible
    assert stats is None
    assert "COMPACTION_STATS" not in log.text


def test_inturn_stats_built_on_compacted(monkeypatch, caplog):
    stats, _, eligible = _run_announce_block(monkeypatch, caplog, "compacted")
    assert eligible
    assert stats is not None  # regression: real compactions still build stats


def test_inturn_stats_built_on_conditional_real_compression(monkeypatch, caplog):
    """Greptile #177 coverage: a CONDITIONAL status ('sanitized') with an ACTUAL
    compression (post < pre) is render-eligible and DOES build stats through the
    integration path — not just the unconditional 'compacted' case."""
    # 12 messages compress to a 4-message tail → post < pre for real
    msgs = [{"role": "user" if i % 2 else "assistant", "content": f"m{i} " + "w" * 40}
            for i in range(12)]
    comp = list(msgs[-4:])
    stats, _, eligible = _run_announce_block(
        monkeypatch, caplog, "sanitized", messages=msgs, compressed=comp,
    )
    assert eligible, "conditional status with post<pre must be render-eligible"
    assert stats is not None


def test_inturn_stats_conditional_noop_not_built(monkeypatch, caplog):
    """A CONDITIONAL status whose 'compression' is a no-op (post == pre) is NOT
    render-eligible → no stats, no marker."""
    stats, log, eligible = _run_announce_block(monkeypatch, caplog, "sanitized")
    assert not eligible
    assert stats is None
    assert "COMPACTION_STATS" not in log.text


def test_nonlcm_stats_not_suppressed_by_gate(monkeypatch, caplog):
    """Greptile #177: the built-in (non-LCM) compressor path is UNCHANGED — the LCM
    render gate must not suppress it even for a status the LCM allow-list would deny
    ('noop'). Non-LCM is always eligible; its announce gating is elsewhere."""
    stats, _, eligible = _run_announce_block(monkeypatch, caplog, "noop", engine="builtin")
    assert eligible, "non-LCM must remain eligible regardless of status"
    assert stats is not None


# ────────────────────────── formatter stats=None tolerance ──────────────────────────

def test_formatter_tolerates_stats_none_on_render_eligible():
    line = _format_compaction_announce(
        engine_name="lcm",
        status="compacted",
        old_session_id="a",
        new_session_id="b",
        old_messages=100,
        new_messages=20,
        pre_tokens=50_000,
        post_tokens=9_000,
        model="m",
        provider="p",
        stats=None,
    )
    assert line, "render-eligible + stats=None must render the two-line form"
    assert "Messages:" in line or "→" in line


# ────────────────────────── D-4: marker self-identification ──────────────────────────

def test_marker_carries_session_and_src_flag(caplog):
    class _Agent:
        session_id = "20260702_120000_abcdef"

    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(_Agent(), "COMPACTION_STATS_TAG_MISSING in-turn")
    rec = [r for r in caplog.records if "COMPACTION_STATS_TAG_MISSING" in r.getMessage()]
    assert rec, "marker not emitted"
    msg = rec[0].getMessage()
    assert "session=20260702_120000_abcdef" in msg
    # running under pytest → PYTEST_CURRENT_TEST is set → src=test present
    assert os.environ.get("PYTEST_CURRENT_TEST")
    assert "src=test" in msg


def test_marker_session_dash_when_missing(caplog):
    class _Agent:
        session_id = None

    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(_Agent(), "COMPACTION_STATS_BUILD_FAILED in-turn")
    rec = [r for r in caplog.records if "COMPACTION_STATS_BUILD_FAILED" in r.getMessage()]
    assert rec and "session=-" in rec[0].getMessage()


def test_marker_throttle_key_unaffected_by_suffix(caplog):
    """The (cause, session) dedupe key must key on the ORIGINAL first-two tokens,
    not be defeated by the appended session/src fields."""
    class _Agent:
        session_id = "S1"

    a = _Agent()
    with caplog.at_level(logging.WARNING):
        _warn_compaction_stats_once(a, "COMPACTION_STATS_TAG_MISSING in-turn")
        _warn_compaction_stats_once(a, "COMPACTION_STATS_TAG_MISSING in-turn")
    hits = [r for r in caplog.records if "COMPACTION_STATS_TAG_MISSING" in r.getMessage()]
    assert len(hits) == 1, "throttle defeated"


def test_src_test_flag_fires_in_fork_test_path(caplog):
    """Pass-3 RC4: the measured polluter (test_compression_concurrent_fork.py) emits
    markers from threading.Thread workers inside the pytest process — NOT subprocesses —
    so PYTEST_CURRENT_TEST is inherited by the emitter. Prove src=test appears on a
    marker emitted from a worker thread (the actual pollution shape)."""
    import threading

    class _Agent:
        session_id = "PARENT_TEST_SESSION"

    with caplog.at_level(logging.WARNING):
        t = threading.Thread(
            target=_warn_compaction_stats_once,
            args=(_Agent(), "COMPACTION_STATS_APPROX_ATTRIBUTION in-turn degraded (x); two-line"),
            name="fork_worker",
        )
        t.start()
        t.join()
    rec = [r for r in caplog.records if "APPROX_ATTRIBUTION" in r.getMessage()]
    assert rec, "marker not emitted from thread"
    msg = rec[0].getMessage()
    assert "session=PARENT_TEST_SESSION" in msg
    assert "src=test" in msg

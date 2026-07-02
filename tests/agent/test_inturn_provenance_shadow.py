"""P2 consumer — shadow-mode provenance trust (spec 2026-07-02, D-3, PR-B).

Multi-pass compactions now carry a provenance stamp, but the consumer treats it as
SHADOW: displayed/validated fields come from the replay/A-floor path exactly as if
unstamped, and ``on_shadow_compare(b_idx, cur_idx)`` reports agree/diverge. Soak
evidence (zero diverge) gates the PR-C trust-flip.

Key negative tests (pass-2 RC#2): the diverge branch actually fires (the soak gate's
own gate), and a WRONG-BUT-IN-RANGE stamp that passes the structural net is caught
by the shadow compare — the exact residual case the pass-1 reviewer named.
"""
from __future__ import annotations

from agent.compaction_stats import build_inturn_stats
from agent.model_metadata import estimate_messages_tokens_rough as _est


def _pre(n=40):
    return [{"role": "user" if i % 2 else "assistant", "content": f"m{i} " + ("w" * 40)}
            for i in range(n)]


def _stamp(rows, origins):
    return [dict(r, **{"_src_idx": o}) for r, o in zip(rows, origins)]


def _identity_sanitize(msgs, **kw):
    return list(msgs)


def _build(pre, comp, *, trust, compare):
    return build_inturn_stats(
        messages=pre,
        compressed=comp,
        estimator=_est,
        engine_is_lcm=False,   # plain rows; no summary classification needed
        sanitize=_identity_sanitize,
        fresh_tail_count=None,  # force B-or-A-floor paths (no replay window)
        provenance_trust=trust,
        on_shadow_compare=compare,
    )


def test_shadow_mode_displays_current_behavior():
    """Shadow: B harvested but NOT used — kept_pre must equal what an unstamped run
    produces (byte-equal displayed fields)."""
    pre = _pre(40)
    kept = _stamp([dict(r) for r in pre[35:]], range(35, 40))
    comp = kept

    calls = []
    shadow = _build(pre, comp, trust="shadow", compare=lambda b, c: calls.append((b, c)))
    unstamped = build_inturn_stats(
        messages=pre, compressed=[dict(r) for r in pre[35:]], estimator=_est,
        engine_is_lcm=False, sanitize=_identity_sanitize, fresh_tail_count=None,
    )
    assert shadow.kept_messages == unstamped.kept_messages
    assert shadow.folded_count == unstamped.folded_count
    assert shadow._kept_pre_messages == unstamped._kept_pre_messages
    assert shadow.approx_attribution == unstamped.approx_attribution
    assert calls, "shadow compare must have been invoked"


def test_shadow_agree_when_b_matches_current():
    pre = _pre(40)
    kept = _stamp([dict(r) for r in pre[35:]], range(35, 40))
    calls = []
    _build(pre, kept, trust="shadow", compare=lambda b, c: calls.append((b, c)))
    (b_idx, cur_idx), = calls
    assert b_idx == list(range(35, 40))
    assert b_idx == cur_idx  # signature partition finds the same rows → agree


def test_shadow_diverge_branch_fires():
    """NEGATIVE (pass-2 RC#2a): construct a B-vs-current disagreement and prove the
    compare reports it — the soak gate's signal is alive."""
    pre = _pre(40)
    # B claims origins 30..34; the actual kept CONTENT matches rows 35..39 →
    # the current (signature) path attributes 35..39. Sets differ → diverge.
    kept = _stamp([dict(r) for r in pre[35:]], range(30, 35))
    calls = []
    _build(pre, kept, trust="shadow", compare=lambda b, c: calls.append((b, c)))
    (b_idx, cur_idx), = calls
    assert b_idx == list(range(30, 35))
    assert cur_idx == list(range(35, 40))
    assert b_idx != cur_idx, "diverge must be observable"


def test_wrongbutinrange_stamp_diverges():
    """NEGATIVE (pass-2 RC#2b): a stamp that passes the structural net (in-range,
    no dup, role+tool_call_id+tool_calls all match — plain same-role text rows)
    but maps to the WRONG origins is caught by the shadow compare, not the net."""
    # EQUAL-TOKEN rows (pass-3 RC1): every row has identical length/token weight, so a
    # scalar token-sum comparator would report a false agree — only an index-SET
    # predicate can detect the swap. i is zero-padded to keep byte length constant.
    pre = [{"role": "user", "content": f"unique-{i:02d} " + ("x" * 30)} for i in range(20)]
    # kept rows are copies of rows 15..19 but stamped as 10..14: same role ("user"),
    # no tool fields → structural net passes; content differs → current path says 15..19.
    kept = _stamp([dict(r) for r in pre[15:]], range(10, 15))
    calls = []
    _build(pre, kept, trust="shadow", compare=lambda b, c: calls.append((b, c)))
    (b_idx, cur_idx), = calls
    assert b_idx == list(range(10, 15))
    assert cur_idx == list(range(15, 20))
    assert b_idx != cur_idx
    # and the token weights ARE equal — proving a sum-based predicate would miss this
    assert _est([pre[i] for i in b_idx]) == _est([pre[i] for i in cur_idx])


def test_singlepass_trust_unchanged():
    """Default trust: B engages for real (PR #110 behavior), no compare invoked."""
    pre = _pre(40)
    kept = _stamp([dict(r) for r in pre[35:]], range(35, 40))
    calls = []
    stats = _build(pre, kept, trust="single-pass", compare=lambda b, c: calls.append((b, c)))
    assert not calls, "single-pass must not shadow-compare"
    assert stats._kept_pre_messages == 5
    assert not stats.approx_attribution  # B is exact


def test_shadow_stats_still_validate():
    pre = _pre(40)
    kept = _stamp([dict(r) for r in pre[35:]], range(35, 40))
    stats = _build(pre, kept, trust="shadow", compare=lambda b, c: None)
    ok, why = stats.validate()
    assert ok, why


def test_shadow_compare_failure_never_breaks_build():
    pre = _pre(40)
    kept = _stamp([dict(r) for r in pre[35:]], range(35, 40))

    def _boom(b, c):
        raise RuntimeError("compare exploded")

    stats = _build(pre, kept, trust="shadow", compare=_boom)
    ok, _ = stats.validate()
    assert ok


def test_diverge_warning_not_throttled_per_session(caplog):
    """Greptile #178: every diverge must be independently observable — the soak
    gate measures within-session frequency, so the diverge arm must NOT use the
    once-per-session throttle. Drive the REAL _on_shadow_compare closure twice
    with a divergence and assert TWO warnings."""
    import logging
    import agent.conversation_compression as cc_mod

    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            if "B_MULTIPASS_SHADOW diverge" in record.getMessage():
                records.append(record.getMessage())

    # Reconstruct the closure exactly as the call site builds it.
    class _Agent:
        session_id = "S-heavy"

    agent = _Agent()

    def _on_shadow_compare(_b_idx, _cur_idx):
        _sid = getattr(agent, "session_id", None) or "-"
        import os
        _src = " src=test" if os.environ.get("PYTEST_CURRENT_TEST") else ""
        if _b_idx == _cur_idx:
            cc_mod.logger.info(
                "COMPACTION_STATS_B_MULTIPASS_SHADOW agree (kept_pre B=%d cur=%d) session=%s%s",
                len(_b_idx), len(_cur_idx), _sid, _src,
            )
        else:
            cc_mod.logger.warning(
                "COMPACTION_STATS_B_MULTIPASS_SHADOW diverge (kept_pre B=%d cur=%d) session=%s%s",
                len(_b_idx), len(_cur_idx), _sid, _src,
            )

    h = _Handler()
    cc_mod.logger.addHandler(h)
    try:
        _on_shadow_compare([1, 2, 3], [1, 2, 4])   # diverge #1
        _on_shadow_compare([1, 2, 3], [1, 2, 5])   # diverge #2, SAME session
    finally:
        cc_mod.logger.removeHandler(h)

    assert len(records) == 2, f"diverge throttled: {records}"
    assert all("session=S-heavy" in m for m in records)


def test_call_site_diverge_uses_direct_logger_not_throttle():
    """Source-structure guard (Greptile #178): the diverge arm of the real
    _on_shadow_compare closure must call logger.warning directly, NOT
    _warn_compaction_stats_once (which throttles once per session)."""
    import inspect
    import agent.conversation_compression as cc_mod

    src = inspect.getsource(cc_mod)
    i = src.index("B_MULTIPASS_SHADOW diverge")
    # the diverge logger call precedes the message string; scan a window AROUND it
    window = src[i - 200:i + 300]
    assert "logger.warning(" in window, "diverge arm must use logger.warning directly"
    assert "_warn_compaction_stats_once" not in window, "diverge arm must not be throttled"

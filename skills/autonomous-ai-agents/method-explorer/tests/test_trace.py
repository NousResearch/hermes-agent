"""Meta-test for trace.py — container-free, deterministic, no tokens.

Gives the shared visibility code teeth: the diagnosis labels must fire on the right
receipts (a non-adjacent re-expand flagged, a fail→later-progress called a recovery), and
`diagnose_run` must agree with helpers.terminal_state. Pins trace.py to the lean schema.

Run:  python3 run.py test_trace
"""
import io

import trace as tr
from helpers import terminal_state


def _row(node, chosen, verdict, **kw):
    r = {"node": node, "q": f"q-{node}", "chosen": chosen, "expected": "x",
         "verdict": verdict, "evidence": "e", "next": "n"}
    r.update(kw)
    return r


def test_diagnose_cycle_labels():
    assert "success" in tr.diagnose_cycle(_row("S3", "delta", "success")).lower()
    assert "tombstone" in tr.diagnose_cycle(_row("S1", "alfa", "fail")).lower()
    # first progress after prior failures = a recovery
    prior = [_row("S1", "alfa", "fail"), _row("S1b", "bravo", "fail")]
    assert "recover" in tr.diagnose_cycle(_row("S2", "charlie", "progress"), prior).lower()
    # plain progress (no prior failure) is not a "recovery"
    plain = tr.diagnose_cycle(_row("S1", "alfa", "progress"), []).lower()
    assert plain.strip().endswith("progress") and "recover" not in plain


def test_reexpand_is_flagged():
    # alfa dead (c1) -> bravo (c2) -> RETURN to alfa (c3, non-adjacent) => invariant violation
    prior = [_row("S1", "alfa", "fail"), _row("S1b", "bravo", "progress")]
    label = tr.diagnose_cycle(_row("S1", "alfa", "fail"), prior)
    assert "RE-EXPAND" in label and "VIOLATION" in label
    # an adjacent rung-0 retry (alfa immediately after alfa) is NOT a re-expand
    assert "RE-EXPAND" not in tr.diagnose_cycle(_row("S1", "alfa", "fail"),
                                                [_row("S1", "alfa", "fail")])


def test_diagnose_run_agrees_with_terminal_state():
    rows = [_row("S1", "alfa", "fail"), _row("S2", "charlie", "progress"),
            _row("S3", "delta", "success")]
    tree = "# Plan-Tree: t   STATE: SUCCESS\nNODES\nFRONTIER: (empty)"
    summary = tr.diagnose_run(rows, tree)
    assert f"classified={terminal_state(rows, tree)}" in summary
    assert "STATE=SUCCESS" in summary and "3 cycles" in summary


def test_show_trace_renders_without_crash():
    rows = [_row("S1", "alfa", "fail"), _row("S2", "charlie", "success")]
    tree = ("# Plan-Tree: t   STATE: SUCCESS\n- S1 alfa ✝ down\n"
            "- S2 charlie ✓ ok\nFRONTIER: (empty)")
    buf = io.StringIO()
    tr.show_trace(rows, tree, intent="do the thing", methods="  alfa=primary",
                  final_words="done", out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert all(k in text for k in ("INPUT", "THINKING", "PLAN-TREE", "OUTCOME"))
    assert "cycle 1" in text and "cycle 2" in text
    # empty rows must render gracefully, not crash
    buf2 = io.StringIO()
    tr.show_trace([], "", out=lambda s: buf2.write(s + "\n"))
    assert "no journal" in buf2.getvalue().lower()

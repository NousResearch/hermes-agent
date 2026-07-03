"""Shared, human-readable trace rendering for method-explorer runs.

Turns a run's artifacts (the lean journal rows + the plan-tree) into a readable
INPUT → THINKING (diagnosed) → PLAN-TREE → OUTCOME trace. Used by demo.py and by
`run.py --show` so ANY agent test can print its input/output with the thinking
diagnosed along the way. Pure formatting — reuses the harness classifiers in helpers,
never re-implements them.
"""
import re

from helpers import dead_set, is_fail, is_succ, terminal_state


def _wrap(label, text, width=92):
    text = " ".join(str(text or "").split())
    if len(text) <= width:
        return f"     {label:9}: {text}"
    return f"     {label:9}: {text[:width]}…"


def diagnose_cycle(row, prior_rows=None):
    """One-line 'what this cycle's thinking did', derived from the receipts (not prose)."""
    prior_rows = prior_rows or []
    v = row.get("verdict")
    chosen = str(row.get("chosen") or "").strip().lower()
    if is_succ(v) and str(v).strip().lower() == "success":
        return "✓ success — intent met"
    if is_succ(v):  # progress
        recovered = (any(is_fail(r.get("verdict")) for r in prior_rows)
                     and not any(is_succ(r.get("verdict")) for r in prior_rows))
        return ("↻ recovered — backtracked to a working branch after failure(s)"
                if recovered else "→ progress")
    if is_fail(v):
        dead = dead_set(prior_rows)
        prev = str(prior_rows[-1].get("chosen") or "").strip().lower() if prior_rows else None
        if chosen and chosen in dead and chosen != prev:
            return "⚠ RE-EXPAND of a tombstoned method — INVARIANT VIOLATION"
        return "✗ tombstone — branch dead, regenerate (backtrack/relax)"
    return "· (no verdict recorded)"


def format_cycle(row, idx, prior_rows=None):
    """The per-cycle block: verdict + diagnosis header, then the lean decision record."""
    v = row.get("verdict", "")
    mark = "✓" if is_succ(v) else ("✗" if is_fail(v) else "·")
    out = [f"  cycle {idx}  [{row.get('node', '?')}]   {mark} verdict={v}   "
           f"« {diagnose_cycle(row, prior_rows)} »"]
    for label, key in (("question", "q"), ("chose", "chosen"), ("expected", "expected"),
                       ("evidence", "evidence"), ("next", "next")):
        out.append(_wrap(label, row.get(key)))
    return "\n".join(out)


def diagnose_run(rows, plan_tree_text=""):
    """Terminal summary line — reuses helpers.terminal_state for the classification."""
    m = re.search(r"STATE:\s*(\S+)", plan_tree_text or "")
    state = m.group(1) if m else "?"
    n_fail = sum(1 for r in rows if is_fail(r.get("verdict")))
    n_prog = sum(1 for r in rows if is_succ(r.get("verdict")))
    return (f"plan-tree STATE={state}  ·  {len(rows)} cycles "
            f"({n_fail} dead, {n_prog} progress/success)  ·  classified={terminal_state(rows, plan_tree_text)}")


def show_trace(rows, plan_tree_text, intent=None, methods=None, final_words=None, out=print):
    """Render the whole INPUT → THINKING(diagnosed) → PLAN-TREE → OUTCOME trace.

    `methods` is a preformatted multi-line string (each demo/test knows its own input).
    """
    if intent:
        out(f"INPUT · intent : {intent}")
    if methods:
        out("INPUT · methods given:")
        out(methods)
    if not rows:
        out("\n(no journal — the run produced nothing to trace)")
        return
    out("\nTHINKING — one decision record per cycle (predict → act → reconcile), diagnosed:")
    for i, r in enumerate(rows, 1):
        out("\n" + format_cycle(r, i, rows[: i - 1]))
    out("\nPLAN-TREE it built (✝ dead · ✓ done · ▶ active · ○ open):")
    for ln in (plan_tree_text or "").splitlines():
        if ln.strip():
            out("    " + ln)
    out("\nOUTCOME · " + diagnose_run(rows, plan_tree_text))
    if final_words:
        out(f"OUTCOME · model's final words: …{' '.join(str(final_words).split())[-160:]}")

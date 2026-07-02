#!/usr/bin/env python3
"""harvest.py — pure parser: a resilient-planner run's artifacts → ledger records.

Turns one cycle's failure conditions into evidence for the next clarify round:
  - plan-tree ✝ dead nodes  → kind=dead-end  "Tried <method>: failed — <reason>"
  - EXHAUSTION-STOP / GUARD-HALT headers → one kind=fact record each
  - journal fail-verdict evidence is merged into each dead record's meta by node id

The plan-tree has NO decision-log section (markers subsume it); per-cycle history lives in
journal.jsonl, which drive.py archives to journal.tick{N}.jsonl between ticks — load_run()
unions them (ticks in numeric order, then the live journal).

Stdlib only; no file writes, no env, no LLM. Grammar per resilient-planner SKILL.md:
  # Plan-Tree: <slug>   STATE: active | SUCCESS | EXHAUSTION-STOP | GUARD-HALT
  - <id>  <method/tag>  [(parent <id>)]  <glyph ✝✓▶○>  [reason/receipt]
  FRONTIER: <method>, ...            (or "(empty ...)")
"""

import hashlib
import json
import os
import re

_STATE_RE = re.compile(r"STATE:\s*([A-Za-z][A-Za-z-]*)", re.IGNORECASE)
_DEAD_RE = re.compile(r"^\s*-\s+(\S+)\s+(.*?)\s*✝\s*(.*)$")
_DONE_RE = re.compile(r"^\s*-\s+(\S+)\s+(.*?)\s*✓\s*(.*)$")
_FRONTIER_RE = re.compile(r"^\s*FRONTIER:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_TICK_RE = re.compile(r"^journal\.tick(\d+)\.jsonl$")


def fp(text):
    """Anti-flap fingerprint: case/whitespace/punctuation-insensitive identity hash."""
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def parse_state(text):
    """Canonical STATE token, normalized by prefix like drive.py. None if unrecognizable."""
    m = _STATE_RE.search(text or "")
    if not m:
        return None
    tok = m.group(1).strip().upper()
    if tok.startswith("ACTIVE"):
        return "active"
    if tok.startswith("SUCCESS"):
        return "SUCCESS"
    if tok.startswith("EXHAUST"):
        return "EXHAUSTION-STOP"
    if tok.startswith("GUARD"):
        return "GUARD-HALT"
    return None


def parse_plan_tree(text):
    """Plan-tree text → {state, guard_text, frontier, dead, done}.

    guard_text: the paragraph starting at a body line beginning "GUARD-HALT:" (below the H1),
    up to the next blank line or the INTENT: line. None when absent.
    frontier: list of untried method labels; "(empty ...)" → [].
    dead/done: [{"id", "method", "reason"|"receipt"}].
    """
    text = text or ""
    lines = text.splitlines()

    guard_lines, in_guard = [], False
    for ln in lines:
        s = ln.strip()
        if not in_guard and s.startswith("GUARD-HALT:") and not s.startswith("#"):
            in_guard = True
            guard_lines.append(s[len("GUARD-HALT:"):].strip())
            continue
        if in_guard:
            if not s or s.startswith("INTENT:"):
                break
            guard_lines.append(s)

    frontier = []
    m = _FRONTIER_RE.search(text)
    if m:
        val = m.group(1).strip()
        if val and not val.startswith("(empty"):
            val = re.sub(r"\([^)]*\)\s*$", "", val).strip()  # drop trailing "(N untried ...)" note
            frontier = [p.strip() for p in val.split(",") if p.strip()]

    dead, done = [], []
    for ln in lines:
        dm = _DEAD_RE.match(ln)
        if dm:
            dead.append({"id": dm.group(1), "method": dm.group(2).strip(),
                         "reason": dm.group(3).strip()})
            continue
        cm = _DONE_RE.match(ln)
        if cm:
            done.append({"id": cm.group(1), "method": cm.group(2).strip(),
                         "receipt": cm.group(3).strip()})

    return {"state": parse_state(text),
            "guard_text": " ".join(guard_lines) if guard_lines else None,
            "frontier": frontier, "dead": dead, "done": done}


def parse_journal(lines):
    """JSON-per-line, tolerant: unparseable or non-dict lines are skipped."""
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def load_run(plans_dir, slug):
    """(plan-tree text or None, journal records) for one planner run.

    Journal = journal.tick{N}.jsonl in numeric order (drive.py archives between ticks),
    then the live journal.jsonl.
    """
    run_dir = os.path.join(plans_dir, slug)
    try:
        with open(os.path.join(run_dir, "plan-tree.md"), encoding="utf-8") as fh:
            tree = fh.read()
    except (FileNotFoundError, NotADirectoryError):
        tree = None

    paths = []
    try:
        ticks = []
        for name in os.listdir(run_dir):
            m = _TICK_RE.match(name)
            if m:
                ticks.append((int(m.group(1)), name))
        paths = [os.path.join(run_dir, name) for _, name in sorted(ticks)]
    except (FileNotFoundError, NotADirectoryError):
        pass
    paths.append(os.path.join(run_dir, "journal.jsonl"))

    records = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                records.extend(parse_journal(fh))
        except (FileNotFoundError, NotADirectoryError):
            continue
    return tree, records


def extract_fork(tree_text):
    """GUARD-HALT only: the human fork the planner refused to coin-flip, as a question."""
    parsed = parse_plan_tree(tree_text)
    if parsed["state"] != "GUARD-HALT":
        return None
    frontier = ", ".join(parsed["frontier"]) or "(none listed)"
    guard = parsed["guard_text"] or "(no guard note)"
    return (f"The planner halted on budget with these branches still open: {frontier}. "
            f"Guard note: {guard} "
            f"Which branch (if any) should be preferred, or what constraint rules them out?")


def harvest(tree_text, journal, cycle):
    """One cycle's artifacts → ledger records (source=harvest).

    Dead-end fp keys on the METHOD LABEL, not the reason: a method dying twice with a
    freshly-phrased reason is the flap this guard exists to catch. Total for any state
    (SUCCESS/active yield dead-end records only).
    """
    parsed = parse_plan_tree(tree_text)
    fails = {}
    for rec in journal:
        if rec.get("verdict") == "fail" and rec.get("node") and rec.get("evidence"):
            fails.setdefault(rec["node"], []).append(rec["evidence"])

    records = []
    for d in parsed["dead"]:
        meta = {"node": d["id"], "reason": d["reason"]}
        if d["id"] in fails:
            meta["journal_evidence"] = fails[d["id"]]
        records.append({"cycle": cycle, "source": "harvest", "kind": "dead-end",
                        "text": f"Tried {d['method']}: failed — {d['reason']}",
                        "fp": fp(d["method"]), "meta": meta})

    if parsed["state"] == "EXHAUSTION-STOP":
        text = ("Planner exhausted every method for this intent under the stated constraints "
                "(frontier empty, all soft constraints relaxed).")
        records.append({"cycle": cycle, "source": "harvest", "kind": "fact",
                        "text": text, "fp": fp(text), "meta": {}})
    elif parsed["state"] == "GUARD-HALT":
        text = (f"Planner halted on a budget guard with branches still open "
                f"(frontier: {', '.join(parsed['frontier']) or 'unknown'}). "
                f"Guard note: {parsed['guard_text'] or '(none)'}")
        records.append({"cycle": cycle, "source": "harvest", "kind": "fact",
                        "text": text, "fp": fp(text), "meta": {}})

    return records

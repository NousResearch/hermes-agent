#!/usr/bin/env python3
"""journey.py — the consolidated decision record: Node = (evidence, options, taken).

The engine's journal.jsonl is the RAW activity log (every step, for durability/replay).
This module folds a run into the CONSOLIDATED record — journey.json — a CHAIN of
decision nodes, each holding exactly three primitives:

  evidence — what is NEW at this node (a delta, never a repeat). Dead-end records here
             ARE the failed paths from earlier nodes (harvest already folds a failed
             task into a dead-end — a failed branch is how evidence gets produced, not
             a separate category); facts are the steps that worked. Each record keeps
             a `from` pointer to the task that produced it, so the decision TREE is
             reconstructible from the chain without storing a tree.
  options  — the moves visible at this node, exactly two states: taken / not_taken,
             each with its contemporaneous why / why_not. What HAPPENED to a taken
             option is not stored on the option — it shows up as evidence at a later
             node. A mid-cycle replan's abandoned tail is a not_taken option
             ("continue-old-tail", why_not = the staleness reason) at the replan node.
  chose    — which option was taken, with the planner's rationale.

Evidence is stored ONLY as per-node deltas: "known at node k" = the union of evidence
at nodes <= k (positional — no ledger array, no watermark indexes to dereference), and
the flat ledger is derivable by concatenation (derive_ledger). The same property makes
the RENDERING LLM-optimal: document order = evidence order, so a model reading a
rendered journey top-to-bottom has, at any node, read exactly what the system knew
there. The primary consumer is an LLM ingesting this as prompt context (the hindsight
judge, future planning oneshots, outer agents); humans are secondary readers.

render_journey(j, level) degrades gracefully: FULL (everything + a Mermaid tail for
humans) / COMPACT (retry/delegate nodes collapse to one line, evidence bodies truncate
harder — what the hindsight judge consumes) / SPINE (one line per node). What never
degrades away: the chain, taken/not_taken, and the fps — the citation skeleton.

Hindsight support: validate_hindsight() checks a judge's claims cite real node keys /
fps; stamp_tiers() classifies each avoidable-branch claim PURELY POSITIONALLY —
(a) genuinely-avoidable: a not_taken option at S_k whose enabling evidence appears at
a node <= k; (b) blind-spot: evidence was known but the option was never recorded;
(c) honest-exploration: the enabling evidence first appears AFTER S_k (the path had to
be run to learn it) → unavoidable. No LLM re-judges an LLM here.

Stdlib only; no file IO, no env, no LLM. fp() is a deliberate copy of planfile.fp /
harvest.fp (pinned behaviorally by tests) — keep them in lockstep.
"""

import hashlib
import re

TEXT_CAP = 400          # FULL-render cap on any free-text line
COMPACT_TEXT_CAP = 120  # COMPACT-render cap (prompt material for the hindsight judge)
MAX_ALTERNATIVES = 3    # recorded not_taken options per decision (prompt-side cap too)
LEVELS = ("SPINE", "COMPACT", "FULL")
TIERS = ("genuinely-avoidable", "blind-spot", "honest-exploration")
OPTIMALITY = ("near-optimal", "acceptable", "sloppy")


def fp(text):
    """Anti-flap fingerprint: case/whitespace/punctuation-insensitive identity hash."""
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def _cap(text, n=TEXT_CAP):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= n else t[:n - 1] + "…"


def _alternatives(plan):
    """Tolerant read of a plan's OPTIONAL `alternatives` field — capture is best-effort,
    so malformed entries are dropped silently, never a validation failure."""
    alts = plan.get("alternatives")
    if not isinstance(alts, list):
        return []
    out = []
    for a in alts[:MAX_ALTERNATIVES]:
        if isinstance(a, dict) and isinstance(a.get("method"), str) and a["method"].strip():
            out.append({"method": a["method"].strip(),
                        "taken": False, "why_not": _cap(a.get("why_not_now") or
                                                        a.get("why_not") or "")})
    return out


def _task_brief(t):
    return {"id": t.get("id"), "method": t.get("method"),
            "success_criterion": t.get("success_criterion", ""),
            "depends_on": list(t.get("depends_on") or [])}


def _tail_label(tasks):
    return " → ".join(t.get("method", "?") for t in tasks) or "(no tasks)"


def plan_event(at, kind, cycle, plan, standing, ledger_len,
               superseded=None, stale_reason=None):
    """One decision event from a validated plan/replan object (or LEVEL 1's
    "replan-failed" sentinel). `superseded` (the old remaining tail, replans only)
    becomes a "continue-old-tail" option: taken=False when a new tail replaced it,
    taken=True when the replan came back needs_decision/exhausted/failed and the
    original tail therefore continued (or the cycle stopped)."""
    disposition = plan.get("disposition")
    options = []
    if disposition == "tasks":
        options.append({"method": _tail_label(plan.get("tasks") or []), "taken": True,
                        "why": _cap(plan.get("rationale") or ""),
                        "tasks": [_task_brief(t) for t in plan.get("tasks") or []]})
        chose = {"disposition": "tasks",
                 "method": options[0]["method"], "why": options[0]["why"]}
    else:
        label = {"needs_decision": "halt: ask a human",
                 "exhausted": "halt: declare exhaustion"}.get(
                     disposition, f"halt: {disposition}")
        why = _cap(plan.get("question") or plan.get("error")
                   or plan.get("rationale") or "")
        # LEVEL 0: any halt verdict is the taken move. Mid-cycle (superseded given):
        # only "exhausted" stops the tail — a forked/failed replan means the original
        # tail continues, so the halt option was NOT taken.
        taken_halt = superseded is None or disposition == "exhausted"
        options.append({"method": label, "taken": taken_halt, "why": why})
        chose = {"disposition": disposition, "method": label, "why": why}
    if superseded is not None:
        old = {"method": f"continue-old-tail: {_tail_label(superseded)}",
               "taken": disposition not in ("tasks", "exhausted"),
               "tasks": [_task_brief(t) for t in superseded]}
        old["why" if old["taken"] else "why_not"] = _cap(
            stale_reason if not old["taken"]
            else f"replan came back {disposition}; original tail continues "
                 f"(stale signal was: {stale_reason})")
        if old["taken"]:
            chose = {"disposition": disposition, "method": old["method"],
                     "why": old["why"]}
        options.append(old)
    options += _alternatives(plan)
    return {"at": at, "kind": kind, "cycle": cycle, "standing": dict(standing or {}),
            "ledger_len": ledger_len, "options": options, "chose": chose}


def retry_event(at, cycle, task, attempt, result, ledger_len):
    """LEVEL 2's per-retry decision: the driver chose to re-attempt the SAME method with
    the scoped clarify's insight rather than accept the dead end. The attempt's outcome
    rides on `chose` (it never folds to the ledger individually — only the task's FINAL
    verdict does), so the record stays honest without inventing evidence records."""
    return {"at": at, "kind": "retry", "cycle": cycle,
            "standing": {"task": task["id"], "attempt": attempt},
            "ledger_len": ledger_len,
            "options": [
                {"method": f"retry: {task['method']}", "taken": True,
                 "why": "local retry budget remained; scoped clarify ran first"},
                {"method": "accept-dead-end", "taken": False,
                 "why_not": "local retry budget remained"}],
            "chose": {"disposition": "retry", "method": f"retry: {task['method']}",
                      "outcome": result["verdict"],
                      "why": _cap(result.get("evidence", ""))}}


def delegate_event(at, cycle, task, deleg, ledger_len):
    """LEVEL 2's exhaustion escalation: the cheap gate chose between a scoped
    method-explorer sub-run and folding the dead end. The sub-run keeps its own
    journey under its own slug — linked by reference, never inlined."""
    attempted = bool(deleg.get("attempted"))
    why = _cap((deleg.get("gate") or {}).get("why", ""))
    options = [
        {"method": "delegate-to-method-explorer", "taken": attempted,
         ("why" if attempted else "why_not"): why,
         "sub_run": deleg.get("slug")},
        {"method": "fold-dead-end", "taken": not attempted,
         ("why" if not attempted else "why_not"): why}]
    chose = {"disposition": "delegate" if attempted else "fold-dead-end",
             "method": options[0 if attempted else 1]["method"], "why": why}
    if attempted:
        chose["outcome"] = deleg.get("status", deleg.get("disposition", "?"))
    return {"at": at, "kind": "delegate", "cycle": cycle,
            "standing": {"task": task["id"]}, "ledger_len": ledger_len,
            "options": options, "chose": chose}


def _evidence(rec):
    e = {"kind": rec["kind"], "text": rec["text"], "fp": rec["fp"],
         "source": rec.get("source", "")}
    meta = rec.get("meta") or {}
    src = meta.get("task") or meta.get("learning_from")
    if src:
        e["from"] = src
    if meta.get("learning_from"):
        e["via"] = "learning"  # a learning is a fact ABOUT the run, not a completed step
    return e


def fold_journey(slug, verdict, detail, receipts, trace, ledger):
    """trace (ordered decision events from plan_event/retry_event/delegate_event) +
    the run's ledger → the journey dict. Evidence lands as per-node deltas cut at each
    event's ledger_len watermark (kept monotonic defensively); whatever arrives after
    the last decision (the final tasks' results) lands on a synthetic terminal node.
    The watermark's clamp to [prev, len(ledger)] is a deliberate never-fail posture: a
    trace with decreasing, corrupted, or out-of-range ledger_len values never crashes
    the fold. Taken options' tasks are annotated primarily by evidence task identity,
    with fingerprint fallback only for legacy identity-free tasks, so a rendered node
    shows how its choice fared without the reader chasing the later evidence."""
    dead_fps = {r["fp"] for r in ledger if r["kind"] == "dead-end"}
    anonymous_ok_fps = {r["fp"] for r in ledger if r["kind"] == "fact"
                        and not (r.get("meta") or {}).get("task")
                        and not (r.get("meta") or {}).get("learning_from")}
    anonymous_dead_fps = {r["fp"] for r in ledger if r["kind"] == "dead-end"
                          and not (r.get("meta") or {}).get("task")}
    worked_ids = {(r.get("cycle"), meta.get("task"))
                  for r in ledger if r["kind"] == "fact"
                  for meta in [r.get("meta") or {}]
                  if meta.get("task") and not meta.get("learning_from")}
    dead_ids = {(r.get("cycle"), meta.get("task"))
                for r in ledger if r["kind"] == "dead-end"
                for meta in [r.get("meta") or {}] if meta.get("task")}
    empty_fp = fp("")
    nodes, raw_deltas, prev = [], [], 0
    for i, ev in enumerate(trace):
        try:
            watermark = int(ev.get("ledger_len", prev))
        except (TypeError, ValueError, OverflowError):
            watermark = prev
        known = max(prev, min(watermark, len(ledger)))
        options = []
        for original in ev.get("options") or []:
            option = dict(original)
            if "tasks" in option:
                option["tasks"] = []
                for original_task in original.get("tasks") or ():
                    task = dict(original_task)
                    if "depends_on" in task:
                        task["depends_on"] = list(task.get("depends_on") or [])
                    option["tasks"].append(task)
            options.append(option)
        for o in options:
            for t in o.get("tasks") or ():
                m = t.get("method")
                tid = t.get("id")
                task_coord = (ev.get("cycle"), tid)
                if task_coord in worked_ids:
                    outcome = "worked"
                elif task_coord in dead_ids:
                    outcome = "failed"
                elif fp(m) != empty_fp:
                    outcome = ("worked" if fp("ok " + (m or "")) in anonymous_ok_fps
                               else "failed" if fp(m) in anonymous_dead_fps
                               else "not-run")
                else:
                    outcome = "not-run"
                t["outcome"] = outcome
        raw_delta = ledger[prev:known]
        nodes.append({"key": f"S{i}", "at": ev["at"], "kind": ev["kind"],
                      "cycle": ev.get("cycle"), "standing": ev.get("standing") or {},
                      "evidence": [_evidence(r) for r in raw_delta],
                      "options": options, "chose": ev.get("chose")})
        raw_deltas.append(raw_delta)
        prev = known
    raw_delta = ledger[prev:]
    nodes.append({"key": f"S{len(trace)}", "at": "terminal", "kind": "terminal",
                  "cycle": receipts.get("cycles"), "standing": {},
                  "evidence": [_evidence(r) for r in raw_delta],
                  "options": [], "chose": None})
    raw_deltas.append(raw_delta)
    # Resolve evidence `from` pointers (task ids) to DECISION COORDINATES —
    # "S<k>:<method>" of the taken option that planned that task — keeping the bare id
    # as from_task. Tolerant: an unresolvable id (e.g. a delegation sub-run's record)
    # keeps the task id. Tree edges are now reconstructible without scanning options.
    # Within a cycle this map is deliberately last-wins: a replan splice can reuse a
    # task id while superseding its original tail, whose earlier planning never ran.
    coord = {}
    for n in nodes:
        for o in n["options"]:
            if o.get("taken"):
                for t in o.get("tasks") or ():
                    coord[(n.get("cycle"), t.get("id"))] = f"{n['key']}:{t.get('method')}"
    for n, raw_delta in zip(nodes, raw_deltas):
        for e, raw in zip(n["evidence"], raw_delta):
            if "from" in e:
                e["from_task"] = e["from"]
                raw_cycle, task_id = raw.get("cycle"), e["from"]
                resolved = coord.get((raw_cycle, task_id))
                if resolved is None:
                    for candidate in reversed(nodes):
                        try:
                            eligible = candidate.get("cycle") <= raw_cycle
                        except TypeError:
                            eligible = False
                        if not eligible:
                            continue
                        match = next((t for o in candidate["options"] if o.get("taken")
                                      for t in o.get("tasks") or ()
                                      if t.get("id") == task_id), None)
                        if match is not None:
                            resolved = f"{candidate['key']}:{match.get('method')}"
                            break
                e["from"] = resolved or task_id
    path = [f"{n['key']}:{n['chose']['method']}" for n in nodes
            if n.get("chose") and n["chose"].get("method")]
    path.append(nodes[-1]["key"])
    # success_path: the worked tasks in execution order — the machine-readable answer to
    # "which steps constitute the win" (non-empty even on failure when a head worked).
    # Repeated task identities can legitimately fail, retry, and later work; worked wins
    # during annotation and dedupe keeps the last matching execution occurrence.
    sp = [{"node": n["key"], "task": t.get("id"), "method": t.get("method")}
          for n in nodes for o in n["options"] if o.get("taken")
          for t in o.get("tasks") or () if t.get("outcome") == "worked"]
    seen_sp, success_path = set(), []
    for entry in reversed(sp):
        key = (entry["task"], entry["method"])
        if key not in seen_sp:
            seen_sp.add(key)
            success_path.append(entry)
    success_path.reverse()
    options_recorded = sum(len(n["options"]) for n in nodes)
    options_taken = sum(1 for n in nodes for o in n["options"] if o.get("taken"))
    return {"schema": 1, "slug": slug, "verdict": verdict, "detail": detail,
            "receipts": dict(receipts or {}), "nodes": nodes, "path": path,
            "success_path": success_path,
            "exploration": {"options_recorded": options_recorded,
                            "options_taken": options_taken,
                            "dead_ends": len(dead_fps),
                            "ratio": round(len(dead_fps) / max(1, options_taken), 2)},
            "hindsight": None}


def degenerate(slug, verdict, detail, receipts, method, why, evidence_text):
    """A one-node journey for the routes with no loop (solve trivial/single_method) —
    one schema for every consumer, however small the run. The synthetic task + the
    "ok <method>" fp namespace (harvest's worked convention) make success_path derive
    naturally instead of being a special case."""
    trace = [{"at": "solve", "kind": "plan", "cycle": 0, "standing": {},
              "ledger_len": 1,
              "options": [{"method": method, "taken": True, "why": _cap(why),
                           "tasks": [{"id": "t0", "method": method,
                                      "success_criterion": "", "depends_on": []}]}],
              "chose": {"disposition": "tasks", "method": method, "why": _cap(why)}}]
    ledger = [{"cycle": 0, "source": "solve", "kind": "fact",
               "text": _cap(evidence_text), "fp": fp("ok " + method),
               "meta": {"task": "t0"}}]
    return fold_journey(slug, verdict, detail, receipts, trace, ledger)


def derive_ledger(journey):
    """The flat evidence ledger, reconstructed by concatenating the per-node deltas —
    the storage IS the deltas; this is the code-consumer convenience view."""
    return [e for n in journey["nodes"] for e in n["evidence"]]


# ── hindsight support (the judge's claims are validated and tiered by CODE) ──────────────────

def validate_hindsight(obj, journey):
    """Violation strings (empty = valid) for a hindsight judge's emission: the
    optimality enum, and the CITATION CONTRACT — every branch claim must name a real
    node key (S<n> or its `at` step key) and any enabling_evidence_fp must resolve in
    the journey's own evidence. An uncited claim is unusable for tier stamping, so it
    is a validation failure (echoed back through the retry channel), not a soft warn."""
    if not isinstance(obj, dict):
        return ["hindsight must be a JSON object"]
    v = []
    if obj.get("optimality") not in OPTIMALITY:
        v.append(f'"optimality" must be one of {list(OPTIMALITY)}')
    hp = obj.get("hindsight_path", [])
    if not isinstance(hp, list):
        v.append('"hindsight_path" must be a list')
    else:
        for i, step in enumerate(hp):
            if not isinstance(step, dict):
                v.append(f"hindsight_path[{i}] must be an object")
            elif not isinstance(step.get("method"), str) or not step["method"].strip():
                v.append(f"hindsight_path[{i}].method must be a non-empty string")
    keys = {n["key"] for n in journey["nodes"]} | {n["at"] for n in journey["nodes"]}
    fps = {e["fp"] for e in derive_ledger(journey)}
    for field in ("avoidable_branches", "unavoidable_branches"):
        claims = obj.get(field, [])
        if not isinstance(claims, list):
            v.append(f'"{field}" must be a list')
            continue
        for i, c in enumerate(claims):
            if not isinstance(c, dict):
                v.append(f"{field}[{i}] must be an object")
                continue
            node = c.get("node")
            if not isinstance(node, str):
                v.append(f"{field}[{i}].node must be a string")
            elif node not in keys:
                v.append(f"{field}[{i}].node {c.get('node')!r} names no node in the "
                         f"journey (valid keys: S0..S{len(journey['nodes']) - 1})")
            if "option" in c and not isinstance(c["option"], str):
                v.append(f"{field}[{i}].option must be a string")
            cited = c.get("enabling_evidence_fp")
            if field == "avoidable_branches":
                if not isinstance(cited, str):
                    v.append(f"{field}[{i}].enabling_evidence_fp must be a string")
                elif cited not in fps:
                    v.append(f"{field}[{i}].enabling_evidence_fp {cited!r} resolves to "
                             f"no evidence fp in the journey — cite one exactly as "
                             f"rendered")
    pl = obj.get("promoted_learnings", [])
    if not isinstance(pl, list) or not all(isinstance(x, str) for x in pl):
        v.append('"promoted_learnings" must be a list of strings')
    return v


def _node_index(journey, key):
    # A key/at collision is impossible by construction: `at` uses lowercase task ids
    # under cycle-prefixed paths (c0/plan, c0/t/t1/retry1), never bare S<digits>.
    for i, n in enumerate(journey["nodes"]):
        if key in (n["key"], n["at"]):
            return i
    return None


def _fp_birth_node(journey, fpv):
    """Index of the node whose evidence delta carries fpv (None if absent) — evidence
    is positional, so 'known at S_k' is simply birth <= k."""
    for i, n in enumerate(journey["nodes"]):
        if any(e["fp"] == fpv for e in n["evidence"]):
            return i
    return None


def _option_matches(claimed, method):
    """Exact normalized identity only: tiering is authoritative/advisory, so a judge
    paraphrase is conservatively a blind spot rather than a fuzzy avoidability claim;
    its prompt already requests verbatim labels. Empty normalization proves nothing."""
    if not isinstance(claimed, str) or not claimed or not isinstance(method, str) or not method:
        return False
    claimed_fp, method_fp, empty_fp = fp(claimed), fp(method), fp("")
    return claimed_fp != empty_fp and method_fp != empty_fp and claimed_fp == method_fp


def stamp_tiers(journey, hindsight):
    """Pure-code tier stamping over the chain (see module docstring). Mutates and
    returns `hindsight`: each avoidable-branch claim gains {tier, seen_at_the_time};
    an option recorded at or before the cited node counts as seen at the time — this
    cumulative horizon is deliberate because a known option does not become unknown.
    honest-exploration claims are the judge over-reaching — dead ends whose disproof
    required running them are the system working as designed."""
    for c in hindsight.get("avoidable_branches") or []:
        k = _node_index(journey, c.get("node"))
        birth = _fp_birth_node(journey, c.get("enabling_evidence_fp"))
        if k is None or birth is None or birth > k:
            c["tier"], c["seen_at_the_time"] = "honest-exploration", False
            continue
        recorded = any(_option_matches(c.get("option") or "", o.get("method") or "")
                       for n in journey["nodes"][:k + 1] for o in n["options"]
                       if not o.get("taken"))
        c["tier"] = "genuinely-avoidable" if recorded else "blind-spot"
        c["seen_at_the_time"] = recorded
    return hindsight


# ── rendering — LLM context first (document order = evidence order) ──────────────────────────

_GLYPH_OK, _GLYPH_DEAD, _GLYPH_NOT = "✓", "✗", "○"  # ✓ ✗ ○


def _option_glyph(o):
    if not o.get("taken"):
        return _GLYPH_NOT
    tasks = o.get("tasks") or []
    return _GLYPH_DEAD if any(t.get("outcome") == "failed" for t in tasks) else _GLYPH_OK


def _option_line(o, cap):
    why = o.get("why") if o.get("taken") else o.get("why_not")
    bits = [f"{_option_glyph(o)} {_cap(o.get('method', '?'), cap)}"]
    if why:
        bits.append(_cap(why, cap))
    if o.get("sub_run"):
        bits.append(f"sub-run: {_cap(o['sub_run'], cap)}")
    return _cap(" — ".join(bits), cap)


def _evidence_line(e, cap):
    frm = ""
    if e.get("from"):
        label = "learning from" if e.get("via") == "learning" else "from"
        frm = f" ({label} {_cap(e['from'], cap)})"
    return _cap(f"- [{e['kind']}·fp {e['fp']}] {_cap(e['text'], cap)}{frm}", cap)


def _hindsight_section(hs, cap):
    if not hs or hs.get("skipped"):
        return ["## Hindsight (advisory)", "",
                f"hindsight unavailable — "
                f"{_cap((hs or {}).get('skipped', 'not run'), cap)}", ""]
    lines = ["## Hindsight (advisory)", "", f"optimality: {hs.get('optimality')}"]
    if hs.get("hindsight_path"):
        lines.append(_cap("shorter path, with hindsight: " + " → ".join(
            _cap(p.get("method", "?") if isinstance(p, dict) else str(p), cap)
            for p in hs["hindsight_path"]), cap))
    for field, title in (("avoidable_branches", "avoidable"),
                         ("unavoidable_branches", "unavoidable")):
        for c in hs.get(field) or []:
            tier = f" [{c['tier']}]" if c.get("tier") else ""
            why = c.get("why") or c.get("why_necessary") or ""
            lines.append(_cap(f"- {title}{tier}: {c.get('node')} "
                              f"{_cap(c.get('option', ''), cap)} — "
                              f"{_cap(why, cap)}", cap))
    for l in hs.get("promoted_learnings") or []:
        lines.append(f"- learned: {_cap(l, cap)}")
    lines.append("")
    return lines


def _standing_line(n):
    s = n.get("standing") or {}
    bits = [f"cycle {n['cycle']}" if n.get("cycle") is not None else None]
    for k in ("elapsed", "budget_remaining", "share"):
        if s.get(k) is not None:
            bits.append(f"{k.replace('_', ' ')} {s[k]}s")
    for k in ("task", "attempt", "after"):
        if s.get(k) is not None:
            bits.append(f"{k} {s[k]}")
    if s.get("done"):
        bits.append(f"done: {', '.join(s['done'])}")
    if s.get("failed"):
        bits.append(f"failed: {', '.join(s['failed'])}")
    return " · ".join(b for b in bits if b)


def _mermaid_label(text):
    """Mermaid quoted-label payload: cap before calling, then allow only inert text
    characters so model-controlled syntax cannot terminate or reshape a node."""
    return re.sub(r"[^A-Za-z0-9 ·→:_./-]", " ", str(text or ""))


def _mermaid(journey):
    lines = ["```mermaid", "flowchart TD"]
    spine = [n for n in journey["nodes"]]
    for i, n in enumerate(spine):
        label = n["at"] if n["kind"] != "terminal" else f"terminal: {journey['verdict']}"
        label = _mermaid_label(_cap(f'{n["key"]} {label}', 60))
        lines.append(f'  {n["key"]}["{label}"]')
        if i:
            lines.append(f"  {spine[i - 1]['key']} --> {n['key']}")
        for j, o in enumerate(n["options"]):
            if o.get("taken"):
                continue
            sid = f"{n['key']}o{j}"
            label = _mermaid_label(_cap(f'{_GLYPH_NOT} {o.get("method", "?")}', 60))
            lines.append(f'  {sid}(["{label}"])')
            lines.append(f"  {n['key']} -.-> {sid}")
    lines.append("```")
    return lines


def render_journey(journey, level="FULL"):
    """The journey as markdown, LLM-context-first: abstract up top (primacy), one block
    per node in decision order with its evidence DELTA (reading order = evidence
    horizon), a one-line recap at the bottom (recency). Tiers degrade gracefully —
    see the module docstring; the citation skeleton (chain, taken/not_taken, fps)
    survives every level. Pure: same journey → same text, at every level."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    cap = TEXT_CAP if level == "FULL" else COMPACT_TEXT_CAP
    r = journey.get("receipts") or {}
    taken_methods = [p.split(":", 1)[1] for p in journey["path"] if ":" in p]
    head = [f"# journey: {journey['slug']} — {journey['verdict'].upper()}",
            # the chain of TAKEN decisions (failed choices included) — the evidence
            # sections below distinguish what actually worked
            "decisions: " + (" → ".join(_cap(m, 60) for m in taken_methods) or "(none)"),
            "receipts: " + " · ".join(f"{k}={v}" for k, v in sorted(r.items())
                                      if v is not None)
            + f" · exploration {journey['exploration']['dead_ends']} dead ends / "
              f"{journey['exploration']['options_taken']} decisions",
            ""]
    if level == "SPINE":
        lines = head
        for n in journey["nodes"]:
            opts = " | ".join(f"{_option_glyph(o)} {_cap(o.get('method', '?'), 50)}"
                              for o in n["options"]) or "(terminal)"
            fps = (" [" + " ".join(e["fp"] for e in n["evidence"]) + "]"
                   if n["evidence"] else "")  # the citation skeleton survives every level
            lines.append(f"{n['key']} {n['at']} ({n['kind']}): {opts}{fps}")
        lines += ["", f"RECAP: {journey['verdict']} — {_cap(journey.get('detail', ''))}"]
        return "\n".join(lines) + "\n"

    lines = head
    if journey["verdict"] == "success":
        lines.append("## The path that worked")
        by_fp = {e["fp"]: e for n in journey["nodes"] for e in n["evidence"]}
        for s in journey.get("success_path") or ():
            done = by_fp.get(fp("ok " + (s["method"] or "")))  # harvest's worked fp
            lines.append(f"- {s['node']} · {_cap(s['method'], cap)}"
                         + (f" — {_cap(done['text'], cap)}" if done else ""))
        if not journey.get("success_path"):
            lines.append("- (see decision chain)")
    else:
        lines.append("## Where it stopped")
        lines.append(f"- stop: {_cap(journey.get('detail', journey['verdict']), cap)}")
        tail = journey["nodes"][-1]["evidence"]
        lines += [_evidence_line(e, cap) for e in tail if e["kind"] in ("dead-end", "gap")]
    lines.append("")

    lines.append("## How we decided")
    for n in journey["nodes"]:
        collapse = level == "COMPACT" and n["kind"] in ("retry", "delegate")
        if collapse:
            outcome = (n.get("chose") or {}).get("outcome", "")
            not_taken = [_cap(o.get("method", "?"), cap) for o in n["options"]
                         if not o.get("taken")]
            lines.append(f"- {n['key']} {n['at']} ({n['kind']}): "
                         f"{_cap((n.get('chose') or {}).get('method', '?'), cap)}"
                         + (f" → {_cap(outcome, cap)}" if outcome else "")
                         + (f" · not: {', '.join(not_taken)}" if not_taken else "")
                         + (f" · +{len(n['evidence'])} evidence "
                            f"[{' '.join(e['fp'] for e in n['evidence'])}]"
                            if n["evidence"] else ""))
            continue
        lines.append(f"### {n['key']} · {n['at']} ({n['kind']})")
        standing = _standing_line(n)
        if standing:
            lines.append(f"standing: {standing}")
        if n["evidence"]:
            lines.append("new evidence:")
            lines += [_evidence_line(e, cap) for e in n["evidence"]]
        if n["options"]:
            lines.append("options (as recorded at the time — not all possible options):")
            lines += [_cap(f"- {_option_line(o, cap)}", cap) for o in n["options"]]
        if n.get("chose"):
            c = n["chose"]
            outcome = f" → {_cap(c['outcome'], cap)}" if c.get("outcome") else ""
            choice = _cap(c.get("method", c.get("disposition", "?")), cap)
            lines.append(f"chose: {choice}{outcome}")
        lines.append("")

    lines += _hindsight_section(journey.get("hindsight"), cap)
    lines.append(f"RECAP: {journey['verdict']} — decisions: "
                 + (" → ".join(_cap(m, 60) for m in taken_methods) or "(none)")
                 + f" — {_cap(journey.get('detail', ''), cap)}")
    if level == "FULL":
        lines += ["", "## Map (human garnish — the node blocks above are the record)"]
        lines += _mermaid(journey)
    return "\n".join(lines) + "\n"

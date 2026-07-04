#!/usr/bin/env python3
"""iterate.py — the Investigator: build context to convergence, then respond.

A dedicated investigator skill, SEPARATE from the `next-best-questions` ranker. It *calls* the ranker
to get the next-best questions, then answers them with a full Hermes agent and folds each distilled
fact into one continuously-growing context (append-only). The loop:

    tombstones = []                                  # answered facts + known gaps
    for round in range(max_rounds):
        evidence = [t.evidence for t in tombstones]  # the shared growing context
        ranked   = infogain.run(problem, evidence)   # rank given everything known so far
        above    = [q in ranked if value >= floor and not already answered][:K]   # top-K BY RANK
        if not above: stop "converged"
        for q in above: tombstones += grounded_answer(q)   # `ask` skill, full agency, distilled
    final = respond(problem, evidence)

Design notes:
- Selection is **top-K by rank** (sidesteps the absolute-threshold problem; floor only ends the loop).
- The grounded ANSWERER is the `ask` skill (`dispatch_single`: full Hermes agent, isolated context) —
  only the distilled fact returns, so the loop context stays lean. Runs INSIDE the hermes container.
- **Capability ladder** (`--capability`, default `act`): full agency by default (all tools, unattended);
  `experiment`/`read` only down-scope for caution. See CAPABILITIES below + references/investigator.md.
- NOT_FOUND is recorded as a plain gap (no revival machinery yet — YAGNI).
- `answerer`/`responder` are injectable so the loop logic is testable on the host with a mock
  (no hermes / no model calls): `--dry-run`.
- **Durability** (`--run-dir`): each tombstone is appended to `<run_dir>/tombstones.jsonl` as it
  lands, and reloaded on re-run — an interrupted investigation resumes instead of re-researching
  (artifact-based resume, the drive.py pattern; no engine dependency). The answerer additionally
  captures each answer as `<run_dir>/answer-<fp>.json` (artifact-beats-stdout; see answerer.py).
  Without `--run-dir` behavior is exactly the old in-memory run.

Usage (inside the container, from the user's project dir):
    python3 scripts/iterate.py --problem "Add authentication to my web app"
    python3 scripts/iterate.py --problem "..." --capability read     # down-scope to read-only
    python3 scripts/iterate.py --problem "..." --run-dir $HERMES_HOME/state/inv-myrun
    python3 scripts/iterate.py --problem "..." --validate top --k 2   # end-to-end test (#21)
On the host (loop logic only): python3 scripts/iterate.py --problem "..." --dry-run

Depends on the next-best-questions ranker (resolved via INFOGAIN_SCRIPTS_DIR / HERMES_HOME) and the
`ask` skill's model_utils (resolved via ASK_SCRIPTS_DIR inside scripts/answerer.py).
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
# This skill depends on the next-best-questions ranker (formerly "information-gain" — the
# INFOGAIN_* env prefix and infogain.py module name are kept). Resolve its scripts dir, trying
# the new skill name first and falling back to the old one for pre-rename installs.
# INFOGAIN_SCRIPTS_DIR overrides for tests / non-standard installs.
_NBQ_CANDIDATES = [os.path.join(_HOME, "skills", "autonomous-ai-agents", name, "scripts")
                   for name in ("next-best-questions", "information-gain")]
_INFOGAIN = os.environ.get("INFOGAIN_SCRIPTS_DIR") or next(
    (p for p in _NBQ_CANDIDATES if os.path.isdir(p)), _NBQ_CANDIDATES[0])
_INFOGAIN_ERR = ("investigator requires the next-best-questions skill (infogain.py) to rank "
                 f"questions. Looked in {_INFOGAIN!r}. Set INFOGAIN_SCRIPTS_DIR or HERMES_HOME.")
sys.path.insert(0, _INFOGAIN)
if _HERE not in sys.path:  # answerer.py lives beside this file
    sys.path.insert(0, _HERE)

try:
    import infogain  # noqa: E402  — the next-best-questions ranker
    _HAVE_INFOGAIN = True
except ImportError as _ie:  # graceful: import-safe without it; rank() raises instead
    _HAVE_INFOGAIN = False

# The answerer/responder seam (ask-skill dispatch, stdout salvage, answer artifacts) lives in
# answerer.py; re-export its names so existing by-name users (tests, external callers) keep
# working against this module.
from answerer import (  # noqa: E402
    _ASK_ERR, _HAVE_ASK, _extract, _strip_suggestion, dispatch_single, fp,
    grounded_answer, judgment_call, qtext_of, read_answer_artifact, refine_prompt, resolve_alias,
    respond, triage_batch,
)

# Capability ladder. DEFAULT = act (full agency, unattended) — today's behavior. The other levels
# only DOWN-scope: they set the answerer's toolsets and inject a restriction directive. (Toolset
# read-only granularity is best-effort/instruction-level in v1; act is unaffected.)
CAPABILITIES = {
    # artifact_write: may the answerer agent be INSTRUCTED to write its answer artifact?
    # Under `read` the directive says "do NOT modify files" — instructing a file write in
    # the same prompt is incoherent, so read keeps the pure stdout path (the tombstone
    # journal is written by THIS process and is unaffected).
    "act": {"toolsets": "file,web,terminal", "directive": "", "artifact_write": True},
    "experiment": {"toolsets": "file,web,terminal", "artifact_write": True,
                   "directive": "You may run REVERSIBLE experiments (prefer a scratch/worktree dir) to "
                                "find out, but do NOT make irreversible or production changes."},
    "read": {"toolsets": "file,web", "artifact_write": False,
             "directive": "READ-ONLY: inspect, search, and read only. Do NOT modify files, run mutating "
                          "commands, or take any action with side effects. If answering requires an "
                          "action, reply NOT_FOUND: needs <action> (capability restricted)."},
}

DEFAULTS = {
    "k": 6, "max_rounds": 3, "floor": 0.12, "key_gap_threshold": 0.40,
    "output": "response", "stakes_aware_respond": False,
    "triage": False, "triage_model": "fast", "triage_provider": "ollama-glm",
    "triage_timeout": 60, "judge_model": "deepseek", "judge_provider": "ollama-glm",
    "judge_timeout": 120, "max_assumes": 6,
    "answer_model": "glm", "answer_provider": "ollama-glm",
    "answer_toolsets": "file,web,terminal", "answer_directive": "",  # = act (default)
    "answer_timeout": 300, "answer_max_turns": None,
    # Where the grounded answerer researches. None = inherit the caller's cwd (the user's project
    # in a real session). Pin it when the wrapper runs detached from the project — a live test showed
    # the answerer otherwise researches the install dir and misses the actual project.
    "answer_cwd": None,
    "responder_model": "glm", "responder_provider": "ollama-glm", "responder_timeout": 300,
    "responder_toolsets": "", "responder_cwd": None,  # responder usually synthesizes from facts (no tools)
    # Durability: journal tombstones (and capture answer artifacts) under this dir; None = in-memory.
    "run_dir": None, "answer_artifact_write": True,
}


def apply_capability(cfg, level):
    """Set answer_toolsets + answer_directive + answer_artifact_write from a capability
    level (act|experiment|read)."""
    cap = CAPABILITIES.get(level, CAPABILITIES["act"])
    cfg["answer_toolsets"] = cap["toolsets"]
    cfg["answer_directive"] = cap["directive"]
    cfg["answer_artifact_write"] = cap.get("artifact_write", True)
    return cfg


# ── tombstone journal (artifact-based resume — the drive.py pattern, no engine) ─────

JOURNAL = "tombstones.jsonl"
REFINED_PROMPT_FILE = "refined-prompt.md"


def _append_journal(run_dir, rec):
    """Plain append (not atomic — a torn tail line costs one re-asked question, by design)."""
    with open(os.path.join(run_dir, JOURNAL), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def _load_journal(run_dir, problem):
    """(tombstones, answered_fps) resumed from run_dir's journal.

    Line 1 is a header {kind: header, problem_fp}; a missing/mismatched header means the
    dir holds a DIFFERENT problem's run — rotate the journal to .stale, clear the answer
    artifacts, and start fresh. Parse is tolerant: unparseable lines are skipped."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, JOURNAL)
    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
    except (FileNotFoundError, NotADirectoryError):
        pass

    header = records[0] if records and records[0].get("kind") == "header" else None
    if header is None or header.get("problem_fp") != fp(problem):
        if records:  # stale run for a different problem — keep it for audit, start fresh
            os.replace(path, path + ".stale")
            for name in os.listdir(run_dir):
                if name.startswith("answer-") and name.endswith(".json"):
                    os.remove(os.path.join(run_dir, name))
            refined_path = os.path.join(run_dir, REFINED_PROMPT_FILE)
            if os.path.exists(refined_path):
                os.remove(refined_path)
        _append_journal(run_dir, {"schema": 1, "kind": "header", "problem_fp": fp(problem)})
        return [], set()
    tombs = [r for r in records[1:]
             if r.get("question") and r.get("status") in ("ANSWERED", "NOT_FOUND")
             and r.get("evidence") is not None]
    return tombs, {fp(t["question"]) for t in tombs}


def _rank_cfg(triage=False):
    if not _HAVE_INFOGAIN:  # tests monkeypatch rank(); the cfg is then unused
        cfg = {}
    else:
        cfg = dict(infogain.DEFAULTS)
        cfg.update(infogain.MODES.get("focus", {}))
        cfg["mode"], cfg["max_rounds"] = "focus", 1
    if triage:
        cfg["auto_derive"] = "on"
    return cfg


def rank(problem, evidence, rank_cfg):
    """Ranked candidate questions (all_scored, value-desc) given the growing context."""
    if not _HAVE_INFOGAIN:
        raise RuntimeError(_INFOGAIN_ERR)
    res = infogain.run(problem, rank_cfg, evidence=evidence)
    ranked = sorted(res.get("all_scored", []), key=lambda r: r.get("value", 0.0), reverse=True)
    return ranked


def _tombstone(q, found, text, via="research", rationale=None):
    qt = q.get("question", "")
    answers = q.get("answers", None)
    stakes_values = []
    if isinstance(answers, list):
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            stakes = answer.get("stakes", None)
            if isinstance(stakes, (int, float)) and not isinstance(stakes, bool):
                stakes_values.append(stakes)
    metadata = {
        "value": q.get("value", None),
        "stakes": max(stakes_values) if stakes_values else None,
        "recommendation": q.get("recommendation", None),
    }
    if found:
        tomb = {"question": qt, "status": "ANSWERED", "fact": text,
                "evidence": f"{qt} -> {text}", "via": via, **metadata}
    else:
        tomb = {"question": qt, "status": "NOT_FOUND", "fact": text,
                "evidence": f"{qt} -> (known gap: {text})", "via": via, **metadata}
    if rationale is not None:
        tomb["rationale"] = rationale
    return tomb


# ── the loop ──────────────────────────────────────────────────────────────────

# The converged stop_reason vocabulary. Exported: relentless-solve's information-dry
# rule keys on it (stop_is_converged prefers this constant when the module is loaded
# and falls back to the "converged" substring on replays — the two must agree).
STOP_CONVERGED = "converged"


def iterate(problem, cfg=None, answerer=None, responder=None, progress=None, seed_evidence=None,
            triager=None, judge=None, refiner=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    cfg["k"] = max(1, int(cfg["k"]))
    answerer = answerer or grounded_answer
    responder = responder or respond
    refiner = refiner or refine_prompt
    triager = triager or triage_batch
    judge = judge or judgment_call
    progress = progress or (lambda m: None)
    rank_cfg = _rank_cfg(bool(cfg.get("triage")))
    seeds = [s for s in (seed_evidence or []) if s.strip()]  # caller-known facts; never tombstoned

    run_dir = cfg.get("run_dir")
    if run_dir:
        tombstones, answered = _load_journal(run_dir, problem)
        if tombstones:
            progress(f"resumed {len(tombstones)} tombstone(s) from {run_dir}")
    else:
        tombstones, answered = [], set()
    n_resumed = len(tombstones)
    assumes_used = sum(1 for t in tombstones if t.get("via") == "assumed")

    # rounds/k_capped count THIS invocation only — a resumed run may converge in round 1.
    rounds, stop_reason, k_capped = 0, None, False
    next_questions = []  # the final round's above-floor-but-unattempted leftovers
    for rnd in range(cfg["max_rounds"]):
        rounds = rnd + 1
        evidence = seeds + [t.get("evidence", "") for t in tombstones]
        ranked = rank(problem, evidence, rank_cfg)
        for r in ranked:
            question = r.get("question", "")
            derived_answer = r.get("derived_answer")
            if (r.get("recommendation") == "DERIVED"
                    and isinstance(derived_answer, str) and derived_answer
                    and fp(question) not in answered):
                tomb = _tombstone(r, True, derived_answer, via="derived")
                tomb["evidence"] = f"{question} -> {derived_answer} (derived during analysis)"
                tombstones.append(tomb)
                if run_dir:
                    _append_journal(run_dir, tomb)
                answered.add(fp(question))
        evidence = seeds + [t.get("evidence", "") for t in tombstones]  # context grows immediately
        above = [r for r in ranked
                 if r.get("value", 0.0) >= cfg["floor"] and fp(r.get("question", "")) not in answered]
        seen_above, deduped_above = set(), []
        for r in above:
            question_fp = fp(qtext_of(r))
            if question_fp not in seen_above:
                seen_above.add(question_fp)
                deduped_above.append(r)
        above = deduped_above
        if not above:
            stop_reason = f"{STOP_CONVERGED} (no question above floor)"
            next_questions = []  # converged: nothing above floor remains unattempted
            break
        if len(above) > cfg["k"]:
            k_capped = True  # more worthwhile questions than K — the cap rate-limited this round
        top = above[: cfg["k"]]
        # What the K-cap rate-limited THIS round, EVSI-ranked — surfaced (additive key,
        # no extra rank() call) so a caller like relentless scope can say "answer these,
        # in this order, to sharpen the scope". Values predate this round's answers.
        next_questions = [{"question": r.get("question", ""),
                           "value": round(float(r.get("value", 0.0)), 4)}
                          for r in above[cfg["k"]:]]
        progress(f"round {rounds}: {len(above)} above floor, researching top {len(top)} "
                 f"(best value={top[0].get('value', 0):.2f})")
        routes = triager(problem, top, evidence, cfg) if cfg["triage"] else {}
        for q in top:
            route = routes.get(fp(qtext_of(q)), "FINDABLE")
            if route == "JUDGMENT" and assumes_used < cfg["max_assumes"]:
                ok, decision, rationale = judge(qtext_of(q), problem, evidence, cfg)
                if ok:
                    tomb = _tombstone(q, True, decision, via="assumed", rationale=rationale)
                    tomb["evidence"] = f"{qtext_of(q)} -> {decision} (assumed: {rationale})"
                    assumes_used += 1
                    found = True
                else:
                    found, text = answerer(q, problem, evidence, cfg)
                    tomb = _tombstone(q, found, text)
            else:
                found, text = answerer(q, problem, evidence, cfg)
                tomb = _tombstone(q, found, text)
            tombstones.append(tomb)
            answered.add(fp(qtext_of(q)))
            if run_dir:
                _append_journal(run_dir, tomb)
            evidence = seeds + [t.get("evidence", "") for t in tombstones]  # context grows immediately
            progress(f"  {'✓' if found else '∅'} {q.get('question', '')[:60]}")
    else:
        stop_reason = "max_rounds reached"

    evidence_final = seeds + [t.get("evidence", "") for t in tombstones]
    gaps = [t for t in tombstones if t["status"] == "NOT_FOUND"]

    def gap_value(tomb):
        value = tomb.get("value", None)
        return value if value is not None else 0.0

    ranked_gaps = sorted(gaps, key=gap_value, reverse=True)
    surfaced_gaps = [t for t in ranked_gaps if gap_value(t) >= cfg["key_gap_threshold"]]
    if ranked_gaps and gap_value(ranked_gaps[0]) < cfg["key_gap_threshold"]:
        surfaced_gaps.append(ranked_gaps[0])
    unresolved_key_questions = []
    seen_gap_questions = set()
    for tomb in sorted(surfaced_gaps, key=gap_value, reverse=True):
        question = tomb.get("question", None)
        if question in seen_gap_questions:
            continue
        seen_gap_questions.add(question)
        unresolved_key_questions.append({
            "question": question,
            "value": tomb.get("value", None),
            "stakes": tomb.get("stakes", None),
            "gap_reason": tomb.get("fact", None),
        })
    output_mode = cfg.get("output", "response")
    refined_prompt = None
    final = None
    if output_mode in ("prompt", "both"):
        refined_prompt = refiner(problem, evidence_final, cfg)
    if output_mode == "both":
        responder_cfg = ({**cfg, "tombstones": tombstones,
                          "unresolved_key_questions": unresolved_key_questions}
                         if cfg.get("stakes_aware_respond") else cfg)
        responder_problem = (problem if isinstance(refined_prompt, str)
                             and refined_prompt.startswith("(no refined prompt:")
                             else refined_prompt)
        final = responder(responder_problem, evidence_final, responder_cfg)
    elif output_mode != "prompt":
        responder_cfg = ({**cfg, "tombstones": tombstones,
                          "unresolved_key_questions": unresolved_key_questions}
                         if cfg.get("stakes_aware_respond") else cfg)
        final = responder(problem, evidence_final, responder_cfg)
    if run_dir:
        refined_path = os.path.join(run_dir, REFINED_PROMPT_FILE)
        if refined_prompt is not None:
            with open(refined_path, "w", encoding="utf-8") as fh:
                fh.write(refined_prompt)
        elif os.path.exists(refined_path):
            os.remove(refined_path)
    artificial_cap = k_capped or bool(next_questions)
    return {
        "problem": problem, "final": final, "refined_prompt": refined_prompt,
        "tombstones": tombstones,
        "rounds": rounds, "stop_reason": stop_reason,
        "k_capped": k_capped, "artificial_cap_bound": artificial_cap,
        "n_resumed": n_resumed, "run_dir": run_dir,
        "n_answered": sum(1 for t in tombstones if t["status"] == "ANSWERED"),
        "n_gaps": sum(1 for t in tombstones if t["status"] == "NOT_FOUND"),
        "n_derived": sum(1 for t in tombstones if t.get("via") == "derived"),
        "n_assumed": sum(1 for t in tombstones if t.get("via") == "assumed"),
        "assumptions": [{"question": t["question"], "decision": t["fact"],
                         "rationale": t.get("rationale", "")}
                        for t in tombstones if t.get("via") == "assumed"],
        "next_questions": next_questions,
        "unresolved_key_questions": unresolved_key_questions,
    }


def validate_selection(problem, which, k, cfg=None, answerer=None, responder=None, progress=None):
    """End-to-end test (#21): respond after answering a selection of round-1 ranked questions.

    which: "baseline" (answer none — respond directly) | "top" (top-k) | "bottom" (bottom-k).
    Single round, so the three are directly comparable for a blind A/B judge.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    answerer = answerer or grounded_answer
    responder = responder or respond
    progress = progress or (lambda m: None)
    if which == "baseline" or k <= 0:
        sel = []
    else:
        ranked = rank(problem, [], _rank_cfg())
        sel = ranked[:k] if which == "top" else list(reversed(ranked[-k:]))
    progress(f"{which}-{k}: " + (" | ".join(q.get("question", "")[:40] for q in sel) or "(no clarification)"))
    tombstones, evidence = [], []
    for q in sel:
        found, text = answerer(q, problem, evidence, cfg)
        tombstones.append(_tombstone(q, found, text))
        evidence = [t["evidence"] for t in tombstones]
    return {"which": which, "k": k, "selected": [q.get("question") for q in sel],
            "values": [round(q.get("value", 0), 3) for q in sel],
            "final": responder(problem, evidence, cfg), "tombstones": tombstones}


# ── mock answerer/responder for host-side loop testing (no hermes) ────────────

def _mock_answerer(q, problem, evidence, cfg):
    return True, f"[mock fact for: {q.get('question', '')[:40]}]"


def _mock_responder(problem, evidence, cfg):
    return f"[mock response over {len(evidence)} established fact(s)]"


def _mock_refiner(problem, evidence, cfg):
    return f"[mock refined prompt over {len(evidence)} established fact(s)]"


def _mock_triager(problem, questions, evidence, cfg):
    return {fp(qtext_of(question)): ("FINDABLE" if i % 2 == 0 else "JUDGMENT")
            for i, question in enumerate(questions)}


def _mock_judge(question, problem, evidence, cfg):
    return (True, "use the standard/default option",
            "reversible and least-surprising choice")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True)
    p.add_argument("--k", type=int, default=DEFAULTS["k"])
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--floor", type=float, default=DEFAULTS["floor"])
    p.add_argument("--key-gap-threshold", type=float, default=None)
    p.add_argument("--capability", choices=["act", "experiment", "read"], default="act",
                   help="full agency (act, default) | reversible experiments | read-only.")
    p.add_argument("--validate", choices=["baseline", "top", "bottom"],
                   help="end-to-end test: respond after answering baseline/top-k/bottom-k.")
    p.add_argument("--dry-run", action="store_true", help="mock answerer/responder (host, no hermes).")
    p.add_argument("--triage", choices=["on", "off"], default=None)
    p.add_argument("--stakes-aware-respond", choices=["on", "off"], default=None)
    p.add_argument("--output", choices=["prompt", "response", "both"], default=None)
    p.add_argument("--triage-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--max-assumes", type=int, default=None)
    p.add_argument("--evidence-file",
                   help="seed facts, one per line (# comments skipped), folded into evidence before "
                        "round 1. Ignored by --validate.")
    p.add_argument("--run-dir",
                   help="journal tombstones (and capture answer artifacts) under this dir; re-running "
                        "with the same dir+problem resumes instead of re-researching. Ignored by "
                        "--validate.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.k < 1:
        p.error("--k must be at least 1")

    def _int_env(name, cli_val, default):
        if cli_val is not None:            # CLI flag wins
            return cli_val
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw.strip())
        except ValueError:
            p.error(f"{name} must be an integer (got {raw!r})")   # names var; stderr; exit 2

    def _float_env(name, cli_val, default):
        if cli_val is not None:            # CLI flag wins
            return cli_val
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw.strip())
        except ValueError:
            p.error(f"{name} must be a float (got {raw!r})")   # names var; stderr; exit 2

    seeds = []
    if args.evidence_file:
        with open(args.evidence_file, encoding="utf-8") as fh:
            seeds = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]

    triage_setting = (args.triage if args.triage is not None
                      else os.environ.get("INVESTIGATOR_TRIAGE", "on"))
    stakes_aware_setting = (args.stakes_aware_respond if args.stakes_aware_respond is not None
                            else os.environ.get("INVESTIGATOR_STAKES_AWARE_RESPOND", "off"))
    output_setting = (args.output if args.output is not None
                      else os.environ.get("INVESTIGATOR_OUTPUT", "prompt"))
    if output_setting not in {"prompt", "response", "both"}:
        p.error(f"INVESTIGATOR_OUTPUT must be prompt, response, or both (got {output_setting!r})")
    triage = triage_setting.lower() == "on"
    stakes_aware_respond = stakes_aware_setting.lower() == "on"
    triage_model = (args.triage_model or os.environ.get("INVESTIGATOR_TRIAGE_MODEL")
                    or DEFAULTS["triage_model"])
    judge_model = (args.judge_model or os.environ.get("INVESTIGATOR_JUDGE_MODEL")
                   or DEFAULTS["judge_model"])
    max_rounds = _int_env("INVESTIGATOR_MAX_ROUNDS", args.max_rounds, DEFAULTS["max_rounds"])
    if max_rounds < 1:
        p.error("max_rounds must be at least 1")
    max_assumes = _int_env("INVESTIGATOR_MAX_ASSUMES", args.max_assumes, DEFAULTS["max_assumes"])
    key_gap_threshold = _float_env("INVESTIGATOR_KEY_GAP_THRESHOLD", args.key_gap_threshold,
                                   DEFAULTS["key_gap_threshold"])
    if max_assumes < 0:
        p.error("max_assumes must be at least 0")
    if not 0.0 <= key_gap_threshold <= 1.0:
        p.error("key_gap_threshold must be between 0.0 and 1.0")
    if not 0.0 <= args.floor <= 1.0:
        p.error("--floor must be between 0.0 and 1.0")
    cfg = apply_capability({"k": args.k, "max_rounds": max_rounds, "floor": args.floor,
                            "key_gap_threshold": key_gap_threshold,
                            "run_dir": args.run_dir, "triage": triage,
                            "stakes_aware_respond": stakes_aware_respond,
                            "output": output_setting,
                            "triage_model": triage_model, "judge_model": judge_model,
                            "max_assumes": max_assumes},
                           args.capability)
    answerer = _mock_answerer if args.dry_run else None
    responder = _mock_responder if args.dry_run else None
    refiner = _mock_refiner if args.dry_run else None
    triager = _mock_triager if args.dry_run else None
    judge = _mock_judge if args.dry_run else None
    if not args.dry_run and not _HAVE_ASK:
        print(f"live runs need the `ask` skill (model_utils): {_ASK_ERR}", file=sys.stderr)
        return 2
    if not args.dry_run and not _HAVE_INFOGAIN:
        print(_INFOGAIN_ERR, file=sys.stderr)
        return 2
    prog = lambda m: print(f"… {m}", file=sys.stderr, flush=True)

    t0 = time.time()
    if args.validate:
        out = validate_selection(args.problem, args.validate, args.k, cfg, answerer, responder, prog)
    else:
        out = iterate(args.problem, cfg, answerer, responder, prog, seed_evidence=seeds,
                      triager=triager, judge=judge, refiner=refiner)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["capability"] = args.capability

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"\n=== stop: {out.get('stop_reason', out.get('which'))} · cap={args.capability} · "
              f"rounds={out.get('rounds', 1)} · answered={out.get('n_answered', '-')} "
              f"gaps={out.get('n_gaps', '-')} · {out['elapsed_s']}s ===\n")
        for t in out["tombstones"]:
            print(f"  [{t['status']}] {t['question'][:70]}\n      {t['fact'][:120]}")
        if out.get("refined_prompt") is not None:
            if out.get("run_dir"):
                print(f"\n--- REFINED PROMPT ---\nWritten to "
                      f"{os.path.join(out['run_dir'], REFINED_PROMPT_FILE)}")
            else:
                print(f"\n--- REFINED PROMPT ---\n{out['refined_prompt']}")
        if out.get("final") is not None:
            print(f"\n--- FINAL RESPONSE ---\n{out['final']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""relentless.py — clarify → plan → execute per task → harvest → repeat, until the intent holds.

The deterministic L3 loop over two existing skills (the loop makes no LLM calls; the
`solve` entry's routing gate makes exactly one, receipted in gate.json):
  CLARIFY  — investigator iterate.py (in-process): rank next-best questions by EVSI, research
             the top-K with a full Hermes agent, fold answers/gaps back as tombstones.
  PLAN     — task-decomposer (one `hermes -z` oneshot): the immutable intent + rendered ledger
             become one validated plan.json — ordered oneshot-sized tasks with success
             criteria, or an honest needs_decision / exhausted verdict.
  EXECUTE  — one `hermes -z` oneshot per task; the executor writes result-<id>.json
             {verdict, evidence} judged against the task's success criterion (a missing
             artifact is a failure — disk beats stdout).
  HARVEST  — harvest.py (pure fold): per-task verdicts become evidence for the next
             clarify/replan round.

EXECUTE is itself two nested levels (run_intent_path): LEVEL 2 (run_task_with_local_retry)
gives a failed task a bounded number of local reattempts, each preceded by a clarify
SCOPED to why that one task failed; LEVEL 1 evaluates a pure-code staleness gate
(stale_tail) after every task and, only when it trips, requests a mid-cycle partial
replan (request_partial_replan) of the untouched tail — never overwriting c<N>/plan.json.
Both levels escalate to the unchanged cycle-boundary behavior (next full clarify+replan)
on exhaustion; the human --gate suspend point lives only at the top of the cycle loop.

The evidence LEDGER is the only shared state; the original prompt is immutable (intent) and
each cycle re-renders prompt+ledger for a fresh plan at `relentless/<slug>/c<N>/`. Stop
conditions: SUCCESS (all plan tasks verified worked — the final task is always intent
verification) | information-dry (a full cycle yields zero fresh facts — anti-flap) |
max_cycles | wallclock (cycle-boundary check + a mid-cycle deadline backstop under the
budget cascade; per-oneshot timeouts cap within-cycle).

Written as a resumable-script flow: each phase is a memoized ctx.step, so a crash replays
completed phases from the journal — the memoized plan step keeps per-task step keys
(c<N>/t/<id>, plus c<N>/t/<id>/retry<K>[/clarify] and c<N>/replan/after-<id>)
replay-deterministic, so execution resumes at the first un-journaled task/retry/replan.
Strict-replay rules honored: branch conditions derive only from the immutable input and
memoized step results; clock reads are steps. The only ctx.ask is the planner's
needs_decision fork under --gate (default is assume-and-note: record the open fork as a
gap and let the next clarify round rank it) — a mid-cycle needs_decision from a partial
replan ALWAYS assume-and-notes, regardless of --gate (see run_intent_path).

Runs INSIDE the hermes container (iterate.py needs the ask skill's model_utils). Host-side
use is tests-only (fakes). State:
  ${HERMES_HOME}/relentless/<slug>/   prompt-c<N>.md · ledger.jsonl · report.md · flow/ (engine)
                                      journey.json (the consolidated decision record —
                                      journey.py; report.md is its pure FULL render)
                                      retro.json (post-success hindsight, success+full only)
                              c<N>/   plan.json (receipt) · result-<id>.json per task

Usage:
  relentless.py run    --slug S (--prompt TEXT | --prompt-file F) --answer-cwd DIR [options]
  relentless.py resume --slug S --answer TEXT [--key K]
Exit codes are the resumable-script engine's (0 completed — read result.outcome; 10 suspended
under --gate; 1/2/3 failures). See SKILL.md.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_AA = os.path.abspath(os.path.join(_HERE, "..", ".."))  # skills/autonomous-ai-agents
sys.path.insert(0, _HERE)

import harvest  # noqa: E402
import journey as journeylib  # noqa: E402  (the consolidated decision record — journey.py)
import knowledge  # noqa: E402  (global tier v1 — see knowledge.py + ARCHITECTURE.md)
import retro_envelope  # noqa: E402  (the post-success hindsight oneshot contract)

_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))

# Knowledge-plane context for THIS process, set by the cmd handlers at CLI time (never
# inside flow source — flow_hash must not change). write_report() promotes through it;
# run_clarify() seeds through inp (both derive from the same CLI flags). `enabled`
# mirrors --knowledge on|off (off = HERMETIC: no seeding, no promotion).
_KNOWLEDGE_CTX = {"enabled": True, "project": None, "slug": None}


def set_knowledge_ctx(enabled, project, slug):
    _KNOWLEDGE_CTX.update(enabled=enabled, project=project, slug=slug)
PLANS_DIR = os.path.join(_HOME, "plans")
# method-explorer (fka resilient-planner) drive.py — old env var and old deployed
# dir name stay accepted so pre-rename deployments keep working.
_DRIVE_CANDIDATES = (
    os.path.join(_HOME, "skills", "method-explorer", "scripts", "drive.py"),
    os.path.join(_HOME, "skills", "resilient-planner", "scripts", "drive.py"))
DRIVE_PY = (os.environ.get("METHOD_EXPLORER_DRIVE")
            or os.environ.get("RESILIENT_DRIVE")
            or next((p for p in _DRIVE_CANDIDATES if os.path.exists(p)),
                    _DRIVE_CANDIDATES[0]))
_ENGINE_DIR = os.environ.get("RESUMABLE_ENGINE_DIR") or os.path.join(
    _HOME, "skills", "resumable-script", "scripts")

_ONESHOT_MOD = None


def _oneshot():
    """Lazy: the bare `hermes -z` dispatch primitive shared with method-explorer's
    drive.py (scripts/oneshot.py) — lives alongside the resumable-script engine, same
    resolution as _ENGINE_DIR (env override or deployed, no sibling check: resumable-
    script isn't always mirrored as a sibling, e.g. in the hermes-agent tree)."""
    global _ONESHOT_MOD
    if _ONESHOT_MOD is None:
        if _ENGINE_DIR not in sys.path:
            sys.path.insert(0, _ENGINE_DIR)
        import oneshot  # noqa: E402
        _ONESHOT_MOD = oneshot
    return _ONESHOT_MOD

DEFAULTS = {
    "max_cycles": 5, "wallclock": 4 * 3600, "k": 6, "inv_rounds": 3, "floor": 0.12,
    "capability": "act", "answer_cwd": None, "gate": False,
    "plan_timeout": 300, "task_timeout": 600,
    # LEVEL 2 (per-task local retry): bounded reattempts + the scoped clarify's own,
    # deliberately tighter k/rounds (this runs per retry, not once per cycle).
    "local_retry_budget": 2, "local_k": 2, "local_inv_rounds": 1,
    # drive config survives for the solve `single_method` route only (method-explorer).
    "drive": {"max_ticks": 12, "per_tick_timeout": 900, "wallclock": 3600},
    # LEVEL 2's exhaustion escalation: a scoped method-explorer sub-run for ONE task,
    # far smaller than the whole-intent `drive` config above — see run_task_delegation.
    "task_drive": {"max_ticks": 4, "per_tick_timeout": 300, "wallclock": 900},
}

# Cascade clamps for the full route's per-cycle oneshot budgets (see relentless_flow).
PLAN_TO_FLOOR, PLAN_TO_CAP = 60, 300
TASK_TO_FLOOR, TASK_TO_CAP = 120, 900
REPLAN_TO_FLOOR, REPLAN_TO_CAP = 60, 240  # LEVEL 1's mid-cycle partial-replan oneshot
PLAN_ATTEMPTS = 3  # planning oneshots per cycle before the flow fails (resumable)
STALE_MIN_OVERLAP = 2  # LEVEL 1 staleness gate: significant-word overlap threshold
MAX_REPLANS_PER_CYCLE = 3  # anti-flap cap on mid-cycle partial replans
MAX_RP_DELEGATIONS_PER_CYCLE = 1  # cap on LEVEL 2's method-explorer sub-runs per cycle
RP_DELEGATION_GATE_TIMEOUT = 180  # cheap classifier, same cost class as the routing gate
# The post-success hindsight oneshot (retro/judge): success + full route only, spends
# only LEFTOVER budget, and can never un-succeed a run (any failure → a skip receipt).
HINDSIGHT_TO_FLOOR, HINDSIGHT_TO_CAP = 60, 180
RETRO_ATTEMPTS = 2  # one violation-echo retry, then drop — advisory work stays cheap

# ── solve: gate → route → run (the one-argument entry) ────────────────────────────────────────
# Interface principle: everything a caller could pass is either DERIVED from the intent
# (slug, route), a deployment CONVENTION (paths), or ADAPTIVE at runtime (budget shares) —
# the honest residue is {intent, budget, risk, gate}. Every derived/adaptive choice leaves
# a receipt (gate.json + report header): an invisible default is a bug.

CONTAINER = os.environ.get("HERMES_CONTAINER", "hermes")
HERMES_BIN = os.environ.get("HERMES_BIN", "/opt/hermes/bin/hermes")
GATE_ROUTES = ("trivial", "single_method", "full")
GATE_TIMEOUT = 180
SOLVE_BUDGET = 1800  # seconds — the ONE number; routes subdivide it

_SLUG_STOP = frozenset(
    "a an and are be for from how i in into is it me my of on or our over please that the "
    "this to under use using we what when which with your".split())


def derive_slug(intent):
    """Deterministic slug from the intent's key nouns (method-explorer's slug rule):
    lowercase kebab-case, <=4 significant words, no dates/counters/randomness — the same
    prompt must land on the same ${HERMES_HOME}/relentless/<slug>/ across re-invocations."""
    words = re.findall(r"[a-z0-9]+", intent.lower())
    keep = [w for w in words if w not in _SLUG_STOP and not w.isdigit()][:4]
    return "-".join(keep) or "task"


def extract_json_object(text):
    """First balanced {...} that parses as JSON, tolerant of surrounding prose/fences
    (oneshot has no native JSON mode — port of hermes_cli's _extract_json_object idea)."""
    for m in re.finditer(r"\{", text or ""):
        depth, start = 0, m.start()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        else:
            break
    return None


def gate_prompt(intent, risk):
    riskline = ("" if risk == "act" else
                f"Risk posture: {risk} (read = the task must not modify anything).\n")
    return (
        "Classify this task for routing. Reply with ONLY one compact JSON object, no prose: "
        '{"route":"trivial"|"single_method"|"full","why":"<one line>"}.\n'
        "trivial = answerable in one short response; no tools or planning needed. "
        "single_method = real work but ONE clear method; no plausible alternatives to search. "
        "full = failure-prone, multi-step, and/or has plausible alternative methods — plans "
        "the work into tasks, executes each with bounded local retry, and can replan mid-run "
        "if a task's outcome invalidates the remaining plan. If unsure, say full.\n"
        + riskline + f"TASK: {intent}"
    )


def run_oneshot(prompt, timeout=GATE_TIMEOUT):
    """One bare `hermes -z` in the container (host-side docker exec; stdout = the final
    response only). Injectable for tests. Dispatches via the shared oneshot module (see
    _oneshot()) — the same primitive method-explorer's drive.py uses in-container."""
    return _oneshot().run_docker_exec(prompt, timeout, CONTAINER, hermes_bin=HERMES_BIN,
                                      pad=60).stdout.strip()


def classify(intent, risk, oneshot=None):
    """Gate verdict {route, why, source}. ANY failure → full: misrouting a trivial prompt
    wastes a few calls; misrouting a hard task to passthrough silently under-serves it."""
    try:
        out = (oneshot or run_oneshot)(gate_prompt(intent, risk))
        obj = extract_json_object(out) or {}
        route = obj.get("route")
        if route in GATE_ROUTES:
            return {"route": route, "why": str(obj.get("why", ""))[:200], "source": "model"}
        return {"route": "full", "source": "default",
                "why": f"unrecognized gate verdict {route!r} -> full"}
    except Exception as e:  # gate must never crash solve — it can only choose a route
        return {"route": "full", "source": "default",
                "why": f"gate error ({e.__class__.__name__}) -> full"}


# ── injectable phase helpers (tests monkeypatch these, like drive.py's DI wiring) ─────────────

_INVESTIGATOR_MOD = None


def _investigator():
    """Lazy: import the investigator (which pulls the next-best-questions ranker) only
    when the live clarify phase actually runs — module import stays stdlib+harvest for
    standalone tests. Resolution: env override → same-repo sibling → deployed
    hermes-agent layout. The ranker dir is pinned before importing iterate when a
    sibling exists (iterate's own default assumes the deployed layout); the skill was
    renamed information-gain → next-best-questions, so try the new name first."""
    global _INVESTIGATOR_MOD
    if _INVESTIGATOR_MOD is None:
        if not os.environ.get("INFOGAIN_SCRIPTS_DIR"):
            for name in ("next-best-questions", "information-gain"):
                d = os.path.join(_AA, name, "scripts")
                if os.path.isdir(d):
                    os.environ["INFOGAIN_SCRIPTS_DIR"] = d
                    break
        candidates = [os.environ.get("INVESTIGATOR_SCRIPTS_DIR"),
                      os.path.join(_AA, "investigator", "scripts"),
                      os.path.join(_HOME, "skills", "autonomous-ai-agents",
                                   "investigator", "scripts")]
        inv_dir = next((d for d in candidates
                        if d and os.path.exists(os.path.join(d, "iterate.py"))), None)
        if inv_dir is None:
            raise SystemExit(f"investigator skill not found (looked in "
                             f"{[c for c in candidates if c]!r}); set "
                             f"INVESTIGATOR_SCRIPTS_DIR or sync the investigator skill.")
        if inv_dir not in sys.path:
            sys.path.insert(0, inv_dir)
        import iterate  # noqa: E402
        _INVESTIGATOR_MOD = iterate
    return _INVESTIGATOR_MOD


def run_clarify(problem, seeds, inp, run_dir=None):
    """Investigator round trimmed to what the loop consumes (the engine journals results).
    run_dir gives the investigator its per-cycle artifact journal (tombstones.jsonl +
    answer artifacts) so a crash mid-clarify resumes instead of re-researching; older
    investigators simply ignore the key.

    Global-tier seeding (knowledge plane, topology B) happens HERE — outside the flow's
    hashed source: same-project records from global.jsonl join the seeds as provenance-
    prefixed evidence texts ONLY (never the ledger — a prior run's dead-end must never
    become a binding dead_fp in this one). `--knowledge off` (inp["knowledge"]) skips it;
    replays never re-read (clarify results come memoized from the journal).

    `next_questions` (EVSI-ranked, above-floor, never attempted — additive key from
    newer investigators; older ones degrade to []) passes through for scope mode's
    package; relentless_flow ignores it."""
    inv = _investigator()
    if inp.get("knowledge", "on") != "off":
        seeds = list(seeds) + knowledge.seed_texts(inp.get("project"))
    cfg = inv.apply_capability(
        {"k": inp["k"], "max_rounds": inp["inv_rounds"], "floor": inp["floor"],
         "answer_cwd": inp["answer_cwd"], "responder_cwd": inp["answer_cwd"],
         "run_dir": run_dir},
        inp["capability"])
    out = inv.iterate(problem, cfg, seed_evidence=seeds)
    return {"tombstones": out["tombstones"], "stop_reason": out["stop_reason"],
            "n_answered": out["n_answered"], "n_gaps": out["n_gaps"],
            "next_questions": out.get("next_questions", [])}


def stop_is_converged(stop_reason):
    """Did clarify stop because EVSI converged (vs a round/budget cap)? Prefer the
    investigator's own STOP_CONVERGED constant when the module is loaded (live runs);
    fall back to the substring for older investigators and for replays, where clarify
    steps come from the journal and the module is never imported. InvestigatorContract
    pins `"converged" in STOP_CONVERGED`, so the two paths cannot disagree."""
    s = stop_reason or ""
    const = getattr(_INVESTIGATOR_MOD, "STOP_CONVERGED", None) if _INVESTIGATOR_MOD else None
    return (const in s) if const else ("converged" in s)


def run_drive(slug, prompt_path, dcfg):
    """Drive one method-explorer run to a terminal STATE (solve `single_method` route
    only — the full loop plans via the task-decomposer skill). Any parseable --json result is
    a successful step (the status travels in the result); raise only on unparseable
    output/timeout."""
    cmd = [sys.executable, DRIVE_PY, "--slug", slug, "--prompt-file", prompt_path,
           "--in-container", "--json",
           "--max-ticks", str(dcfg["max_ticks"]),
           "--per-tick-timeout", str(dcfg["per_tick_timeout"]),
           "--wallclock", str(dcfg["wallclock"])]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=dcfg["wallclock"] + 300)
    lines = [ln for ln in (p.stdout or "").strip().splitlines() if ln.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"drive.py produced no parseable JSON (rc={p.returncode}): "
                           f"{(p.stderr or '')[-500:]}") from e
    return {**result, "slug": slug}


_DECOMPOSER_MODS = None


def _decomposer():
    """Lazy: the task-decomposer skill OWNS the plan-as-data contract (scripts/planfile.py +
    scripts/envelope.py) — load both from the same dir so schema and prompt can't drift.
    Loaded via importlib under private names: method-explorer also ships an `envelope`
    module and both may be on sys.path. Resolution: env override → same-repo sibling →
    deployed $HERMES_HOME layout. Returns (planfile, envelope)."""
    global _DECOMPOSER_MODS
    if _DECOMPOSER_MODS is None:
        import importlib.util
        candidates = [os.environ.get("TASK_DECOMPOSER_DIR"),
                      os.path.abspath(os.path.join(_HERE, "..", "..",
                                                   "task-decomposer", "scripts")),
                      os.path.join(_HOME, "skills", "task-decomposer", "scripts")]
        for d in candidates:
            if d and os.path.exists(os.path.join(d, "planfile.py")):
                mods = []
                for name in ("planfile", "envelope"):
                    spec = importlib.util.spec_from_file_location(
                        f"taskdecomposer_{name}", os.path.join(d, f"{name}.py"))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    mods.append(mod)
                _DECOMPOSER_MODS = tuple(mods)
                break
        else:
            raise SystemExit(f"task-decomposer skill not found (looked in "
                             f"{[c for c in candidates if c]!r}); set TASK_DECOMPOSER_DIR "
                             f"or sync the task-decomposer skill alongside this one.")
    return _DECOMPOSER_MODS


_REPORTER_MOD = None


def _reporter():
    """task-decomposer's report.py (the completion contract) — loaded from the SAME
    directory _decomposer() resolved, so schema and report cannot drift apart. Lazy: only
    the --dod path ever touches it."""
    global _REPORTER_MOD
    if _REPORTER_MOD is None:
        import importlib.util
        planfile, _ = _decomposer()
        path = os.path.join(os.path.dirname(planfile.__file__), "report.py")
        spec = importlib.util.spec_from_file_location("taskdecomposer_report", path)
        _REPORTER_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_REPORTER_MOD)
    return _REPORTER_MOD


_SPEC_MOD = None


def _spec():
    """Lazy: the define-done skill OWNS the dod.md grammar (scripts/spec.py). Only
    loaded when a run carries --dod; the dod itself always travels as parse_dod()'s
    dict, never as a cross-skill import inside the contract modules. Resolution
    mirrors _decomposer(): env override → same-repo sibling → deployed $HERMES_HOME."""
    global _SPEC_MOD
    if _SPEC_MOD is None:
        import importlib.util
        candidates = [os.environ.get("DEFINE_DONE_DIR"),
                      os.path.abspath(os.path.join(_HERE, "..", "..",
                                                   "define-done", "scripts")),
                      os.path.join(_HOME, "skills", "define-done", "scripts")]
        for d in candidates:
            if d and os.path.exists(os.path.join(d, "spec.py")):
                spec = importlib.util.spec_from_file_location(
                    "definedone_spec", os.path.join(d, "spec.py"))
                _SPEC_MOD = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_SPEC_MOD)
                break
        else:
            raise SystemExit(f"define-done skill not found (looked in "
                             f"{[c for c in candidates if c]!r}); set DEFINE_DONE_DIR "
                             f"or sync the define-done skill alongside this one.")
    return _SPEC_MOD


def invoke_hermes(prompt, timeout):
    """One bare `hermes -z` (stdout = the final response only; '' on timeout — artifacts
    on disk still count, the drive.py lesson). Dispatches via the shared oneshot module
    (see _oneshot()) — direct subprocess when the hermes binary is local (in-container,
    the loop's normal habitat, same primitive method-explorer's drive.py uses), docker
    exec otherwise (host-side use is tests-only; tests monkeypatch this)."""
    if os.path.exists(HERMES_BIN):
        p = _oneshot().run_direct(prompt, timeout, hermes_bin=HERMES_BIN, pad=60)
    else:
        p = _oneshot().run_docker_exec(prompt, timeout, CONTAINER, hermes_bin=HERMES_BIN,
                                       pad=60)
    return (p.stdout or "").strip()


def _plan_attempt_loop(prompt_fn, out_path, timeout, stamp_fn, validate_kwargs, label,
                       extra_checks=None):
    """Shared bounded-retry loop (PLAN_ATTEMPTS strikes, violation-echo, .rejN archiving)
    behind BOTH request_plan (LEVEL 0's whole-cycle plan) and request_partial_replan
    (LEVEL 1's mid-cycle tail-only replan), so the two call sites' retry contract cannot
    drift apart. prompt_fn(violations_or_None) builds each attempt's prompt — None on the
    first attempt (no prior violations to echo). extra_checks(obj) → more violations in
    the same shape (dead-method / dod-coverage), echoed through the same retry channel."""
    planfile, _ = _decomposer()
    violations = ["(no attempt made)"]
    for attempt in range(PLAN_ATTEMPTS):
        prompt = prompt_fn(violations if attempt else None)
        stdout = invoke_hermes(prompt, timeout)
        obj = planfile.load(out_path) or extract_json_object(stdout)
        if obj is None:
            violations = ["no plan.json artifact and no parseable JSON object in the reply"]
        else:
            stamp_fn(obj)  # stamp identity; models echo stale values
            violations = planfile.validate(obj, **validate_kwargs)
            if not violations and extra_checks:
                violations = extra_checks(obj)
        if not violations:
            planfile.dump(obj, out_path)  # canonical artifact even on the stdout fallback
            return obj
        if os.path.exists(out_path):
            os.replace(out_path, f"{out_path}.rej{attempt}")
    raise RuntimeError(f"{label} produced no valid plan after {PLAN_ATTEMPTS} attempts: "
                       f"{violations}")


def _binding_checks(planfile, dodctx, dead_fps):
    """The two validation-time bindings absorbed from the taskmap grammar: never
    re-propose a dead method; when a dod fronts the intent, every unmet requirement
    must be served (or the plan must honestly be needs_decision/exhausted)."""
    def check(obj):
        v = planfile.dead_violations(obj, dead_fps)
        if dodctx:
            v += planfile.coverage_violations(obj, dodctx["unmet"],
                                              known_ids=dodctx["known"])
        return v
    return check


def request_plan(slug_dir, slug, cycle, rendered_body, timeout, dodctx=None,
                 dead_fps=()):
    """One validated plan.json from a task-decomposer oneshot, with bounded retries that echo
    the violations back. The artifact beats stdout (a timeout can kill stdout after the
    file write landed); rejected attempts are archived beside it for the audit. Raise only
    after PLAN_ATTEMPTS strikes — the flow fails visibly and stays resumable.
    dodctx ({parsed, unmet, known}, or None) makes coverage binding; dead_fps makes the
    dead-method rule binding — both echoed through the same retry channel."""
    planfile, penv = _decomposer()
    cycle_dir = os.path.join(slug_dir, f"c{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)
    out_path = planfile.plan_path(cycle_dir)

    def prompt_fn(violations):
        base = penv.plan_prompt(rendered_body, out_path,
                                dod_ids=(dodctx or {}).get("unmet"))
        return base + penv.retry_suffix(violations) if violations else base

    return _plan_attempt_loop(prompt_fn, out_path, timeout,
                              stamp_fn=lambda o: o.update(slug=slug, cycle=cycle),
                              validate_kwargs={}, label="task-decomposer",
                              extra_checks=_binding_checks(planfile, dodctx, dead_fps))


def render_partial(prompt, ledger, done_tasks, done_results, dod_section=None):
    """LEVEL 1's mid-cycle replan body: render() (intent + ledger, UNCHANGED) plus a
    "Completed this cycle" section — mechanical task-attempt detail, appended exactly
    like render()'s own evidence sections. Never touches `prompt` itself — the intent
    stays declarative regardless of how much mid-cycle mechanics accumulate below it."""
    verdicts = {r["id"]: r for r in done_results}
    lines = ["## Completed this cycle so far (do NOT re-plan; do NOT reuse these ids)"]
    for t in done_tasks:
        r = verdicts.get(t["id"], {})
        lines.append(f"- {t['id']} ({t['method']}): {r.get('verdict', '?')} — "
                     f"{r.get('evidence', '')}")
    return (render(prompt, ledger, dod_section=dod_section)
            + "\n\n" + "\n".join(lines) + "\n")


def request_partial_replan(slug_dir, slug, cycle, seq, rendered_body, done_ids, timeout,
                           dodctx=None, dead_fps=()):
    """LEVEL 1's mid-cycle plan request: its own artifact (c<N>/replan-<seq>.json) — it
    NEVER overwrites c<N>/plan.json, which stays the original cycle plan for audit. The
    model supplies only the NEW tail (forbidden_ids blocks reusing a completed task's
    id); the caller splices tasks[:i+1] + this result's tasks in code, so the model is
    never asked to reproduce already-completed tasks verbatim. dodctx here carries the
    TAIL's unmet ids (the caller subtracts what the completed head already served)."""
    planfile, penv = _decomposer()
    cycle_dir = os.path.join(slug_dir, f"c{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)  # matches request_plan's defensive posture
    out_path = os.path.join(cycle_dir, f"replan-{seq}.json")

    def prompt_fn(violations):
        base = penv.partial_replan_prompt(rendered_body, out_path, done_ids,
                                          dod_ids=(dodctx or {}).get("unmet"))
        return base + penv.retry_suffix(violations) if violations else base

    return _plan_attempt_loop(
        prompt_fn, out_path, timeout,
        stamp_fn=lambda o: o.update(slug=slug, cycle=cycle, replan_seq=seq),
        validate_kwargs={"forbidden_ids": set(done_ids)},
        label=f"partial-replan(seq={seq})",
        extra_checks=_binding_checks(planfile, dodctx, dead_fps))


def _attempt_partial_replan(slug_dir, slug, cycle, seq, rendered_body, done_ids, timeout,
                            dodctx=None, dead_fps=()):
    """A failed OPTIMIZATION (the staleness gate's mid-cycle replan) must never sink a
    cycle that could otherwise finish its original tail — wrap request_partial_replan's
    exception into a MEMOIZABLE sentinel instead of letting it raise. Distinct from a
    genuine model-declared "exhausted"/"needs_decision" verdict, which IS returned as-is."""
    try:
        return request_partial_replan(slug_dir, slug, cycle, seq, rendered_body, done_ids,
                                      timeout, dodctx=dodctx, dead_fps=dead_fps)
    except Exception as e:
        return {"disposition": "replan-failed", "error": f"{e.__class__.__name__}: {e}"}


LEARNINGS_MAX_COUNT = 5      # per task attempt — a runaway-ledger backstop, not a target
LEARNINGS_MAX_CHARS = 500    # matches the existing evidence[:500] truncation convention


def task_prompt(task, cycle_dir, suffix=None, capability=None):
    """The executor oneshot for ONE task attempt: do the work, judge strictly against the
    success criterion, leave the verdict as an artifact (never trust stdout to survive a
    timeout). `suffix` (a LEVEL 2 local-retry attempt label, e.g. "retry1") keeps each
    attempt's artifact on disk separately for audit: result-<id>.json for the first
    attempt (byte-identical to today), result-<id>-<suffix>.json for a reattempt."""
    _decomposer()  # fail early if the contract modules are missing
    base = task["id"] if not suffix else f"{task['id']}-{suffix}"
    rpath = os.path.join(cycle_dir, f"result-{base}.json")
    read_only = ("HARD CONSTRAINT: read-only — observe and verify only; do not modify "
                 "any file, config, or external state.\n" if capability == "read" else "")
    return (
        "Execute ONE task, then verify it.\n"
        + read_only
        + f"TASK: {task['description']}\n"
        f"SUCCESS CRITERION: {task['success_criterion']}\n"
        "Do the work, then JUDGE the outcome STRICTLY against the success criterion — "
        "verify by observation, not by assumption. Whether it worked or failed, write "
        f"EXACTLY one JSON object to {rpath} AND echo the same JSON as your final message: "
        '{"verdict": "worked"|"failed"|"needs_split", '
        '"evidence": "<one line: what you observed vs the criterion>", '
        '"split": ["<subtask>", ...], '
        '"learnings": ["<optional: any OTHER fact worth remembering, beyond the pass/fail '
        'verdict itself>"]}. '
        "Write the file even on failure — a missing file is treated as failure.\n"
        '"needs_split" is for ONE case only: the task cannot be completed as ONE method '
        "and must be split into smaller tasks — then \"split\" (required for this "
        "verdict) names the pieces. Do NOT use it merely because the task is hard or "
        "failed; a doable-but-failed task is \"failed\".\n"
        '"learnings" is OPTIONAL — most tasks report none. When you DO include one, make '
        "it a self-contained mini post-mortem someone with no other context could act on: "
        "name the SYSTEMS/MATERIALS involved, WHAT and WHY you expected to succeed (your "
        "hypothesis going in), WHAT actually happened, and WHY it succeeded or failed.\n"
        "  THIN (not enough to act on): \"the API call failed\".\n"
        "  RICH (self-contained): \"Called the billing API's /v2/charges endpoint "
        "expecting a synchronous 200, since the docs describe it as blocking — it "
        "actually returns 202 and processes async via a webhook, so callers must poll or "
        "subscribe rather than read the response body directly.\"\n"
    )


def run_task(task, cycle_dir, timeout, suffix=None, capability=None):
    """One task attempt → {id, method, verdict, evidence, learnings}. The verdict is read
    from the result artifact by CODE (no LLM judge); artifact beats stdout, but a
    parseable echoed JSON object in stdout is now read as a FALLBACK when the artifact is
    missing (e.g. a timeout after the model replied but before the file write landed) —
    the same artifact-then-stdout posture request_plan already uses; only a missing OR
    malformed result in both places is a failure. `learnings` are OPTIONAL incidental
    facts (capped at LEARNINGS_MAX_COUNT, truncated at LEARNINGS_MAX_CHARS each) — folded
    into the ledger by harvest.py alongside (not instead of) the primary worked/failed
    record, whether this attempt succeeded or failed."""
    stdout = invoke_hermes(task_prompt(task, cycle_dir, suffix,
                                       capability=capability), timeout)
    base = task["id"] if not suffix else f"{task['id']}-{suffix}"
    rpath = os.path.join(cycle_dir, f"result-{base}.json")
    verdict, evidence, learnings = "failed", "no verdict artifact (timeout or malformed output)", []
    try:
        with open(rpath, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, ValueError):
        obj = extract_json_object(stdout)
    split = []
    if isinstance(obj, dict):
        if obj.get("verdict") in ("worked", "failed", "needs_split"):
            verdict = obj["verdict"]
            evidence = (str(obj.get("evidence", "")).strip()[:500]
                        or "(no evidence reported)")
            raw = obj.get("learnings")
            if isinstance(raw, list):
                learnings = [str(x).strip()[:LEARNINGS_MAX_CHARS] for x in raw
                            if str(x).strip()][:LEARNINGS_MAX_COUNT]
            if verdict == "needs_split":
                raw_split = obj.get("split")
                if isinstance(raw_split, list):
                    split = [str(x).strip()[:200] for x in raw_split if str(x).strip()][:8]
                if not split:  # the split list is the verdict's whole point
                    verdict = "failed"
                    evidence = f"needs_split without a split list — treated as failed ({evidence})"
        else:
            evidence = "malformed verdict artifact"
    return {"id": task["id"], "method": task["method"],
            "verdict": verdict, "evidence": evidence, "learnings": learnings,
            "split": split}


# ── LEVEL 2 — bounded local recovery of ONE task ───────────────────────────────────────────────
# Owns exactly one decision: does THIS task get another local shot before its failure is
# allowed to escalate to LEVEL 1's existing (unchanged) cycle-boundary behavior.

def scoped_clarify_problem(task, result):
    """LEVEL 2's clarify problem — built from TASK FIELDS ONLY (method/description/
    success_criterion/evidence), never inp["prompt"]. This is what keeps a local retry's
    investigation scoped to "why did THIS task fail" instead of drifting into
    re-litigating the intent."""
    return (
        "A task attempt failed; investigate the SPECIFIC cause of THIS failure and what "
        "would make it succeed (not the broader goal).\n"
        f"TASK METHOD: {task['method']}\nTASK: {task['description']}\n"
        f"SUCCESS CRITERION: {task['success_criterion']}\n"
        f"OBSERVED FAILURE: {result['evidence']}"
    )


def _run_scoped(problem, inp, run_dir, ledger):
    """Shared by run_scoped_clarify (LEVEL 2) and run_replan_clarify (LEVEL 1): both are
    the SAME investigator primitive (run_clarify) with the SAME tighter local_k/
    local_inv_rounds config (cheap — these fire per retry/replan attempt, not once per
    cycle) and the SAME whole-ledger seeding (single blackboard). The two callers differ
    only in how `problem` gets built (one task's failure vs. fresh evidence + a remaining
    task list) — kept as separate named functions below for that call-site clarity and
    their distinct LEVEL-anchored docstrings, not because the dispatch itself differs."""
    scoped_inp = {**inp, "k": inp.get("local_k", DEFAULTS["local_k"]),
                 "inv_rounds": inp.get("local_inv_rounds", DEFAULTS["local_inv_rounds"])}
    return run_clarify(problem, [r["text"] for r in ledger], scoped_inp, run_dir=run_dir)


def run_scoped_clarify(task, result, inp, run_dir, ledger):
    """LEVEL 2's clarify call — a problem scoped to this one task's failure. Its own
    run_dir means a crash mid-scoped-clarify resumes this one narrow investigation via
    investigator's own tombstones.jsonl journal, same mechanism LEVEL 0's clarify already
    relies on. See _run_scoped for the shared dispatch."""
    return _run_scoped(scoped_clarify_problem(task, result), inp, run_dir, ledger)


# ── LEVEL 2's exhaustion escalation — a scoped method-explorer sub-run for ONE task ──────────
# Fires only when local retry is exhausted (not deadline-starved) — the same diagnose→
# retry→backtrack→cap policy method-explorer already owns, deliberately parallel (see
# relentless-solve's SKILL.md design notes), invoked here at task grain instead of
# re-implemented. Never a hard dependency: any failure to reach method-explorer falls
# back to LEVEL 1's existing partial-replan path, unchanged.

def task_delegation_intent(task, result, retry_meta):
    """The `intent` handed to method-explorer — TASK FIELDS ONLY (method/description/
    success_criterion/intent_link), plus the already-tried method named as a known-dead
    approach so the search doesn't re-choose it. Never touches inp["prompt"] — same
    intent/mechanics separation as scoped_clarify_problem."""
    learned = "; ".join(retry_meta.get("scoped_texts", []) + retry_meta.get("task_learnings", []))
    return (
        f"{task['description']}\n"
        f"SUCCESS CRITERION: {task['success_criterion']}\n"
        f"WHY THIS MATTERS: {task.get('intent_link', '')}\n"
        f"DO NOT RETRY the method '{task['method']}' — it already failed: "
        f"{result['evidence']}"
        + (f"\nALREADY LEARNED (from local investigation): {learned}" if learned else "")
    )


def rp_delegation_gate(task, result, retry_meta, inp, oneshot=None):
    """Cheap one-call classifier (same try/except-degrades-safely pattern as classify()):
    does this exhausted task look like it has PLAUSIBLE ALTERNATIVE METHODS worth a real
    search, or does the failure look environmental/unfixable by a different approach
    (permissions, missing infra — no method would help)? ANY failure (parse error,
    exception, timeout) -> False — never escalate into an expensive sub-run on ambiguity.
    Returns {"alt_methods_plausible": bool, "why": str}."""
    prompt = (
        "Classify whether a failed task has PLAUSIBLE ALTERNATIVE METHODS worth "
        "searching, or whether the failure looks environmental/unfixable by a different "
        "approach. Reply with ONLY one compact JSON object, no prose: "
        '{"alt_methods_plausible": true|false, "why": "<one line>"}.\n'
        f"TASK METHOD (already tried, failed): {task['method']}\n"
        f"TASK: {task['description']}\n"
        f"WHY IT FAILED: {result['evidence']}\n"
        "WHAT LOCAL RETRIES ALREADY LEARNED: "
        + ("; ".join(retry_meta.get("scoped_texts", []) + retry_meta.get("task_learnings", []))
           or "(nothing new)")
    )
    try:
        out = (oneshot or invoke_hermes)(prompt, RP_DELEGATION_GATE_TIMEOUT)
        obj = extract_json_object(out) or {}
        return {"alt_methods_plausible": obj.get("alt_methods_plausible") is True,
                "why": str(obj.get("why", ""))[:200]}
    except Exception as e:
        return {"alt_methods_plausible": False,
                "why": f"gate error ({e.__class__.__name__}) -> no delegation"}


def run_task_delegation(slug, task, result, retry_meta, cycle_dir, dcfg, risk, drive=None):
    """Hands ONE task off to a scoped method-explorer sub-run: builds
    `<slug>-<cycle-basename>-<task-id>` as its own plans/ slug (never collides with the
    whole-intent `<slug>-single` route or across cycles), writes the prompt to this
    cycle's own scratch (`c<N>/rp-<task-id>/prompt.md`, matching the `clarify-scoped/...`
    convention), and drives it to a terminal STATE via the SAME run_drive primitive
    solve_single already uses. Returns the raw drive status dict ({status, detail, ...})."""
    rp_slug = f"{slug}-{os.path.basename(cycle_dir)}-{task['id']}"
    extra = ("HARD CONSTRAINT: read-only — do not modify any external state.\n"
             if risk == "read" else "")
    intent = task_delegation_intent(task, result, retry_meta)
    prompt = _envelope().real_prompt(intent, rp_slug, PLANS_DIR, extra=extra)
    rp_dir = os.path.join(cycle_dir, f"rp-{task['id']}")
    os.makedirs(rp_dir, exist_ok=True)
    ppath = os.path.join(rp_dir, "prompt.md")
    _atomic_write(ppath, prompt)
    return (drive or run_drive)(rp_slug, ppath, dcfg)


def _attempt_task_delegation(slug, task, result, retry_meta, cycle_dir, dcfg, risk,
                             drive=None):
    """Wraps run_task_delegation like _attempt_partial_replan wraps
    request_partial_replan: a missing method-explorer sibling (_envelope() raises
    SystemExit — checked explicitly since it's a BaseException, not caught by a bare
    `except Exception`) or any runtime failure becomes a MEMOIZABLE sentinel instead of
    raising. A failed or unavailable delegation must never sink a cycle that could
    otherwise finish via LEVEL 1's existing partial-replan path."""
    try:
        _envelope()
    except SystemExit as e:
        return {"disposition": "delegation-unavailable", "error": str(e)}
    try:
        return {"disposition": "delegated",
               **run_task_delegation(slug, task, result, retry_meta, cycle_dir, dcfg,
                                     risk, drive=drive)}
    except Exception as e:
        return {"disposition": "delegation-failed", "error": f"{e.__class__.__name__}: {e}"}


def _maybe_delegate_task(slug, task, result, retry_meta, cycle_dir, dcfg, risk, inp,
                         oneshot=None, drive=None):
    """The single memoized unit for LEVEL 2's exhaustion escalation: gate-check, then
    (conditionally) attempt the scoped delegation. Kept as one function so the caller's
    ctx.step wraps exactly one black-box operation, matching the rest of the codebase's
    memoization granularity (one step = one decision + its consequence)."""
    gate = rp_delegation_gate(task, result, retry_meta, inp, oneshot=oneshot)
    if not gate["alt_methods_plausible"]:
        return {"attempted": False, "gate": gate}
    outcome = _attempt_task_delegation(slug, task, result, retry_meta, cycle_dir, dcfg,
                                       risk, drive=drive)
    return {"attempted": True, "gate": gate, **outcome}


def run_task_with_local_retry(ctx, cycle, task, cycle_dir, task_to, ledger, seen, inp,
                              deadline=None, allow_delegation=False, trace=None):
    """LEVEL 2 — bounded reattempt loop for ONE task. On failure: run a scoped clarify,
    fold any fresh fact straight into the shared ledger (so LEVEL 1's staleness gate sees
    it immediately), reattempt the SAME task id. Bounded by inp["local_retry_budget"] — a
    plain integer comparison against an immutable config value, so this cannot loop
    forever by construction. On exhaustion, the last attempt's {verdict:"failed", ...} is
    returned UNCHANGED for the caller to fold via harvest.harvest_tasks exactly as today's
    single-attempt failure — escalation behavior stays byte-identical to today, UNLESS
    `allow_delegation` is set and a cheap gate judges the failure has plausible
    alternative methods, in which case a scoped method-explorer sub-run gets one shot
    at the task before escalation (see _maybe_delegate_task).

    A task that succeeds on its FIRST attempt uses the exact same step key (c<N>/t/<id>)
    and result artifact (result-<id>.json) as before this feature existed, and — since
    `deadline` is only consulted right before an actual retry attempt — issues NO new
    ctx.step call either: a task-that-just-works costs nothing extra.

    `deadline` (a memoized wall-clock value, cascade-only) bounds retries against
    aggregate wallclock: checked ONLY right before spending time on a retry (not on the
    first, already-budgeted attempt), so a slow model can't chain unbounded retries past
    the cycle's fair share. The SAME `deadline_hit` flag also gates delegation below — no
    point starting an expensive sub-run with no wallclock left.

    Returns (result, retry_meta) where retry_meta =
        {"attempts": int, "fresh_local": int, "exhausted": bool, "scoped_texts": [str],
         "task_learnings": [str], "delegated": bool}.
    "exhausted" is True whether the budget ran out or the deadline did — both mean no
    more LOCAL recovery is available for this task in this cycle (a SUCCESSFUL delegation
    flips this back to False). "task_learnings" accumulates EVERY attempt's `learnings`
    (failed attempts included, not just the final one) — a failed attempt can still
    surface something worth remembering that isn't captured by its dead-end evidence text.
    """
    c = f"c{cycle}"
    budget = inp.get("local_retry_budget", DEFAULTS["local_retry_budget"])
    result = ctx.step(
        f"{c}/t/{task['id']}",
        lambda t=task: run_task(t, cycle_dir, task_to,
                                capability=inp.get("capability")))
    fresh_local, scoped_texts, attempt, deadline_hit = 0, [], 0, False
    task_learnings = list(result.get("learnings") or [])
    # needs_split short-circuits LEVEL 2 entirely: neither a local retry nor a
    # method-explorer delegation fixes GRANULARITY — the task goes straight back to
    # LEVEL 1, which folds the split hint and forces a partial replan. (`exhausted`
    # below stays False for it — attempt is 0 — so delegation never fires either.)
    while result["verdict"] not in ("worked", "needs_split") and attempt < budget:
        if deadline is not None:
            now = ctx.step(f"{c}/t/{task['id']}/retry{attempt + 1}/clock",
                           lambda: time.time())
            if now > deadline:
                deadline_hit = True
                break
        attempt += 1
        run_dir = os.path.join(cycle_dir, "clarify-scoped", f"{task['id']}-retry{attempt}")
        scoped = ctx.step(
            f"{c}/t/{task['id']}/retry{attempt}/clarify",
            lambda t=task, r=result, rd=run_dir:
                run_scoped_clarify(t, r, inp, rd, ledger))
        fresh_local += fold_clarify(scoped["tombstones"], cycle, ledger, seen,
                                    source="scoped-clarify")
        scoped_texts += [t["evidence"] for t in scoped["tombstones"]]
        result = ctx.step(
            f"{c}/t/{task['id']}/retry{attempt}",
            lambda t=task, a=attempt: run_task(
                t, cycle_dir, task_to, suffix=f"retry{a}",
                capability=inp.get("capability")))
        task_learnings += result.get("learnings") or []
        if trace is not None:
            trace.append(journeylib.retry_event(
                f"{c}/t/{task['id']}/retry{attempt}", cycle, task, attempt, result,
                len(ledger)))

    exhausted = (budget > 0 and result["verdict"] not in ("worked", "needs_split")
                and (attempt == budget or deadline_hit))
    delegated = False
    if exhausted and not deadline_hit and allow_delegation:
        retry_meta_so_far = {"scoped_texts": scoped_texts, "task_learnings": task_learnings}
        dcfg = inp.get("task_drive", DEFAULTS["task_drive"])
        deleg = ctx.step(
            f"{c}/t/{task['id']}/rp-delegate",
            lambda t=task, r=result, rm=retry_meta_so_far:
                _maybe_delegate_task(inp["slug"], t, r, rm, cycle_dir, dcfg,
                                     inp["capability"], inp))
        if trace is not None:
            trace.append(journeylib.delegate_event(
                f"{c}/t/{task['id']}/rp-delegate", cycle, task, deleg, len(ledger)))
        if deleg["attempted"] and deleg.get("status") == "SUCCESS":
            delegated = True
            alt = deleg.get("detail") or "an alternate method"
            result = {"id": task["id"], "method": task["method"], "verdict": "worked",
                      "evidence": f"method-explorer delegation succeeded: {alt}"}
            task_learnings.append(
                f"Task '{task['method']}' originally failed, but a method-explorer "
                f"delegation found a working alternative: {alt}")
            exhausted = False
        elif deleg["attempted"]:
            fresh_local += fold_one(
                ledger, seen, cycle, "harvest", "dead-end",
                f"method-explorer delegation for task '{task['method']}' ended "
                f"{deleg.get('status', deleg.get('disposition', 'without success'))}: "
                f"{deleg.get('detail', deleg.get('error', ''))}")
            # result/exhausted UNCHANGED — LEVEL 1's existing partial-replan path fires
            # exactly as today, just with a richer dead-set.

    return result, {"attempts": attempt, "fresh_local": fresh_local, "exhausted": exhausted,
                    "scoped_texts": scoped_texts, "task_learnings": task_learnings,
                    "delegated": delegated}


# ── LEVEL 1 — task sequencing, staleness gate, mid-cycle partial replan ───────────────────────
# Owns exactly one decision: given the evidence gained so far THIS cycle, does the
# remaining to-do list still hold, or should its untouched tail be reforged.

def _significant_words(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _SLUG_STOP and not w.isdigit() and len(w) > 2}


def stale_tail(result, retry_meta, remaining, ledger, min_overlap=STALE_MIN_OVERLAP):
    """LEVEL 1's staleness gate — pure code, no LLM, no I/O. Any ONE trigger is enough:

    A. DEAD-METHOD REUSE — a remaining task's method fingerprint (harvest.fp, the
       existing anti-flap function) now matches a dead-end just recorded: the tail is
       provably planning to reuse a method already proven dead. Exact, no false positives.
    B. VOCABULARY BLEED — the fresh evidence text (this task's result evidence, any
       LEVEL 2 scoped-clarify facts, AND any `learnings` the attempt(s) reported —
       incidental facts fire this trigger even on a SUCCESSFUL task) shares
       >= min_overlap stopword-filtered significant words with a remaining task's
       method+description+success_criterion+intent_link. A deliberately blunt, cheap
       proxy for "this new fact is ABOUT a queued task."
    C. RETRY EXHAUSTION — this task burned its full local-retry budget and still failed.
       Unconditional: local recovery already failed, which is itself strong enough signal.

    False positives cost one extra oneshot (hard-capped by MAX_REPLANS_PER_CYCLE); false
    negatives fall through to LEVEL 0's existing next-cycle clarify+replan safety net.
    Returns (is_stale, reason) — reason is None when not stale.
    """
    dead_fps = {r["fp"] for r in ledger if r["kind"] == "dead-end"}
    if any(harvest.fp(t["method"]) in dead_fps for t in remaining):
        return True, "dead-method-reuse"
    fresh_words = _significant_words(" ".join(
        [result["evidence"]] + retry_meta["scoped_texts"] + retry_meta["task_learnings"]))
    if fresh_words:
        for t in remaining:
            tail_words = _significant_words(
                f"{t['method']} {t['description']} {t['success_criterion']} "
                f"{t.get('intent_link', '')}")
            if len(fresh_words & tail_words) >= min_overlap:
                return True, "vocabulary-bleed"
    if retry_meta["exhausted"]:
        return True, "retry-exhaustion"
    return False, None


def replan_clarify_problem(fresh_texts, remaining):
    """LEVEL 1's pre-replan clarify problem — scoped to the fresh evidence that just
    tripped the staleness gate (not literally "why did this task fail" like LEVEL 2's
    problem — this is "given what was just learned, what's still worth asking before
    reforging the remaining plan"). Built from evidence/learnings texts + the remaining
    tasks' own fields only — never touches inp["prompt"] directly (same intent/mechanics
    separation as scoped_clarify_problem)."""
    evidence = "; ".join(t for t in fresh_texts if t) or "(no specific new evidence)"
    tail = "; ".join(f"{t['id']} ({t['method']})" for t in remaining)
    return (
        "Fresh evidence suggests the REMAINING task plan may need revision before it is "
        "reforged.\n"
        f"FRESH EVIDENCE: {evidence}\n"
        f"REMAINING TASKS ABOUT TO BE REPLANNED: {tail}\n"
        "What open questions, if answered now, would materially improve the remaining "
        "plan? If nothing is worth asking, say so."
    )


def run_replan_clarify(fresh_texts, remaining, inp, run_dir, ledger):
    """LEVEL 1's pre-replan clarify — a problem scoped to the fresh evidence that tripped
    the staleness gate plus the remaining task list. Seeded with the WHOLE ledger (single
    blackboard — includes any `learnings` already folded from this task, so the replan is
    genuinely informed by them, not just by the raw task_learnings list passed in). Its
    own run_dir gives it the same tombstones.jsonl resume durability as every other
    clarify call. See _run_scoped for the shared dispatch."""
    return _run_scoped(replan_clarify_problem(fresh_texts, remaining), inp, run_dir, ledger)


def run_intent_path(ctx, cycle, plan, cycle_dir, task_to, replan_to, ledger, seen, inp,
                    slug_dir, deadline=None, dodctx=None, dead_fps=(), trace=None):
    """LEVEL 1 — walks ONE cycle's to-do list task by task, delegating each attempt to
    LEVEL 2 (run_task_with_local_retry). Owns: dependency-skip, per-task harvest fold, the
    STALENESS GATE (evaluated after EVERY task), and — only when the gate fires — a
    bounded partial replan of the untouched tail. Mapping intent to task always leverages
    whatever is already in the ledger (none, the first time) and, right before that
    mapping happens, runs a next-best-questions clarify pass to refine it — LEVEL 0 does
    this via run_clarify, LEVEL 2 via run_scoped_clarify, and LEVEL 1's partial replan via
    run_replan_clarify below, so no "map intent to task" moment in the loop skips it.
    `deadline` (a memoized wall-clock value, cascade-only) is the mid-cycle wallclock
    backstop against aggregate overshoot from chained retries/replans — consulted only at
    those two "extra work" boundaries (inside LEVEL 2's retry loop, and right before a
    replan attempt below), never before a task's own first attempt (already bounded by
    task_to) — so a cycle where nothing fails or replans issues no new clock reads at all.

    needs_decision from a partial replan ALWAYS assume-and-notes — it never calls
    ctx.ask, even under --gate. The human-suspend gate lives only at the top of the cycle
    loop (LEVEL 0); a second suspend point nested inside task execution would need
    continuation/resume semantics that don't exist today. If the fork persists, it
    resurfaces at the NEXT cycle boundary, where --gate already works.

    Returns (results, fresh_harv): the final per-task verdicts (dependency-skips included)
    and the total fresh-ledger-record count (task verdicts + LEVEL 2's scoped-clarify
    facts + any replan-fork gap facts).
    """
    c = f"c{cycle}"
    results, failed, fresh_harv, replan_seq, delegations_used = [], set(), 0, 0, 0
    tasks = list(plan["tasks"])
    i = 0
    while i < len(tasks):
        task = tasks[i]
        if failed.intersection(task.get("depends_on") or ()):
            results.append({"id": task["id"], "method": task["method"],
                            "verdict": "skipped", "evidence": "dependency failed"})
            i += 1
            continue

        r, retry_meta = run_task_with_local_retry(
            ctx, cycle, task, cycle_dir, task_to, ledger, seen, inp, deadline=deadline,
            allow_delegation=delegations_used < MAX_RP_DELEGATIONS_PER_CYCLE,
            trace=trace)
        if retry_meta["delegated"]:
            delegations_used += 1
        # Merge in EVERY attempt's learnings (not a mutation — r may be a cached dict
        # returned straight from a memoized ctx.step; a new dict keeps that cache clean).
        if retry_meta["task_learnings"]:
            r = {**r, "learnings": retry_meta["task_learnings"]}
        results.append(r)
        fresh_harv += retry_meta["fresh_local"]
        if r["verdict"] != "worked":
            failed.add(task["id"])
        fresh_harv += fold_records(ledger, seen,
                                   harvest.harvest_tasks({"tasks": tasks}, [r], cycle))

        remaining = tasks[i + 1:]
        if remaining and replan_seq < MAX_REPLANS_PER_CYCLE:
            # needs_split IS the staleness signal — the executor itself declared the
            # plan's granularity wrong, so skip the heuristic gate and replan the tail
            # now. The split hint reaches the replan prompt via the ledger (harvest
            # folds it as a fact; render_partial renders facts).
            if r["verdict"] == "needs_split":
                stale, _reason = True, "needs-split"
            else:
                stale, _reason = stale_tail(r, retry_meta, remaining, ledger)
            past_deadline = False
            if stale and deadline is not None:
                rnow = ctx.step(f"{c}/replan/after-{task['id']}/clock", lambda: time.time())
                past_deadline = rnow > deadline
            if stale and past_deadline:
                fresh_harv += fold_one(
                    ledger, seen, cycle, "assumption", "gap",
                    f"Staleness noted near task {task['id']} but the cycle deadline was "
                    "reached; continuing with the original remaining tasks.")
            elif stale:
                replan_seq += 1
                # Map intent to task always leverages whatever's already known (the
                # ledger) and, right before that mapping, refines it via a next-best-
                # questions clarify pass — the SAME thing LEVEL 0 does before its plan
                # and LEVEL 2 does before its retry, so LEVEL 1's replan doesn't skip it.
                fresh_texts = ([r["evidence"]] + retry_meta["scoped_texts"]
                               + retry_meta["task_learnings"])
                clarify_run_dir = os.path.join(
                    cycle_dir, "clarify-scoped", f"replan{replan_seq}-after-{task['id']}")
                pre_replan = ctx.step(
                    f"{c}/replan/after-{task['id']}/clarify",
                    lambda ft=fresh_texts, rem=remaining, rd=clarify_run_dir:
                        run_replan_clarify(ft, rem, inp, rd, ledger))
                fresh_harv += fold_clarify(pre_replan["tombstones"], cycle, ledger, seen,
                                           source="replan-clarify")
                done_ids = {t["id"] for t in tasks[:i + 1]}
                tail_dodctx = dodctx
                if dodctx:
                    # the tail need not re-cover what the completed head already served
                    worked = {r["id"] for r in results if r["verdict"] == "worked"}
                    head_served = {rid for t in tasks[:i + 1] if t["id"] in worked
                                   for rid in t.get("serves") or ()}
                    tail_dodctx = {**dodctx,
                                   "unmet": [rid for rid in dodctx["unmet"]
                                             if rid not in head_served]}
                rendered = render_partial(inp["prompt"], ledger, tasks[:i + 1], results,
                                          dod_section=dodctx["section"] if dodctx else None)
                replan = ctx.step(
                    f"{c}/replan/after-{task['id']}",
                    lambda body=rendered, done=done_ids, seq=replan_seq, dc=tail_dodctx:
                        _attempt_partial_replan(slug_dir, inp["slug"], cycle, seq, body,
                                                done, replan_to, dodctx=dc,
                                                dead_fps=dead_fps))
                if trace is not None:
                    trace.append(journeylib.plan_event(
                        f"{c}/replan/after-{task['id']}", "replan", cycle, replan,
                        {"after": task["id"], "done": sorted(done_ids),
                         "failed": sorted(failed)},
                        len(ledger), superseded=remaining, stale_reason=_reason))
                if replan["disposition"] == "tasks":
                    tasks = tasks[:i + 1] + replan["tasks"]
                elif replan["disposition"] == "exhausted":
                    fresh_harv += fold_one(
                        ledger, seen, cycle, "harvest", "fact",
                        "Partial replan declared exhaustion mid-cycle: no method outside "
                        "the dead-end list remains for the untouched tail.")
                    break
                elif replan["disposition"] == "needs_decision":
                    q = replan.get("question") or "(replan asked an unspecified question)"
                    fresh_harv += fold_one(
                        ledger, seen, cycle, "assumption", "gap",
                        f"OPEN FORK (partial replan, unresolved): {q} -> ASSUMED: keep "
                        "the original remaining tasks; the next cycle's clarify should "
                        "rank it.")
                    # tasks left UNCHANGED — never suspends (see docstring)
                else:  # "replan-failed" — a non-fatal technical failure
                    fresh_harv += fold_one(
                        ledger, seen, cycle, "assumption", "gap",
                        f"Partial replan attempt failed technically "
                        f"({replan.get('error', '?')}); continuing with the original "
                        "remaining tasks.")
        i += 1
    return results, fresh_harv


def write_plan_receipt(cycle_dir, plan, results, report=None):
    """Annotate per-task verdicts back into plan.json — the cycle's receipt (scratch:
    read by humans, never by the loop, which folds from the in-memory results). When a
    dod fronts the run, `report` (a computed completion_report dict) lands beside it as
    c<N>/report.json — the cycle's formal completion contract."""
    planfile, _ = _decomposer()
    verdicts = {r["id"]: r["verdict"] for r in results}
    receipt = dict(plan)
    receipt["tasks"] = [{**t, "status": verdicts.get(t["id"], t.get("status", "pending"))}
                        for t in plan.get("tasks") or []]
    path = planfile.plan_path(cycle_dir)
    planfile.dump(receipt, path)
    if report is not None:
        _reporter().save_report(cycle_dir, report)
    return path


def dod_requirements_section(dod_parsed):
    """The unmet requirements as a rendered prompt section — pure fn of parse_dod()'s
    dict (met/waived leaves omitted; they are receipts, not work)."""
    lines = ["## Requirements (definition of done) — unmet, by id"]
    for g in dod_parsed.get("groups", []):
        for it in g.get("items", []):
            if it.get("marker") not in ("✓", "~"):
                lines.append(f"- {it['id']} {it['text']}")
    return "\n".join(lines)


def render(prompt, ledger, dod_section=None):
    """Immutable intent + the evidence ledger, section per kind. Empty sections omitted.
    dod_section (from dod_requirements_section) sits directly under the intent — it is
    part of the WHAT, not of the attempt evidence below it."""
    sections = (("fact", "## Established facts (do not re-derive)"),
                ("gap", "## Known gaps (proceed on the stated assumption)"),
                ("dead-end", "## Dead ends — do NOT re-attempt these methods"))
    parts = [prompt.rstrip()]
    if dod_section:
        parts.append(dod_section)
    for kind, header in sections:
        texts = [r["text"] for r in ledger if r["kind"] == kind]
        if texts:
            parts.append(header + "\n" + "\n".join(f"- {t}" for t in texts))
    return "\n\n".join(parts) + "\n"


_ENVELOPE_MOD = None


def _envelope():
    """method-explorer OWNS its invocation contract (scripts/envelope.py) — import it
    directly instead of keeping a runtime copy that can drift. Lazy + cached. Resolution:
    env override (new name, then RESILIENT_ENVELOPE_DIR for pre-rename deployments) →
    same-repo sibling (also the in-container layout, /opt/data/skills/…; old dir name
    accepted) → deployed $HERMES_HOME layout (both names). Both skills ship together in
    both layouts, so a missing envelope is a broken deployment, not a standalone mode."""
    global _ENVELOPE_MOD
    if _ENVELOPE_MOD is None:
        candidates = [os.environ.get("METHOD_EXPLORER_DIR"),
                      os.environ.get("RESILIENT_ENVELOPE_DIR"),  # back-compat (fka)
                      os.path.abspath(os.path.join(_HERE, "..", "..",
                                                   "method-explorer", "scripts")),
                      os.path.abspath(os.path.join(_HERE, "..", "..",
                                                   "resilient-planner", "scripts")),
                      os.path.join(_HOME, "skills", "method-explorer", "scripts"),
                      os.path.join(_HOME, "skills", "resilient-planner", "scripts")]
        for d in candidates:
            if d and os.path.exists(os.path.join(d, "envelope.py")):
                if d not in sys.path:
                    sys.path.insert(0, d)
                import envelope  # noqa: E402
                _ENVELOPE_MOD = envelope
                break
        else:
            raise SystemExit(f"method-explorer envelope.py not found (looked in "
                             f"{[c for c in candidates if c]!r}); set METHOD_EXPLORER_DIR "
                             f"or sync the method-explorer skill alongside this one.")
    return _ENVELOPE_MOD


def _atomic_write(path, content):
    tmp = path + ".tmp"
    if isinstance(content, bytes):
        with open(tmp, "wb") as fh:
            fh.write(content)
    else:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(content)
    os.replace(tmp, path)


def persist(slug_dir, cycle, rendered, ledger):
    os.makedirs(slug_dir, exist_ok=True)
    prompt_path = os.path.join(slug_dir, f"prompt-c{cycle}.md")
    _atomic_write(prompt_path, rendered)
    _atomic_write(os.path.join(slug_dir, "ledger.jsonl"),
                  "".join(json.dumps(r) + "\n" for r in ledger))
    return {"prompt_path": prompt_path}


def write_report(slug_dir, outcome, ledger, cycles, detail, requirements=None,
                 journey_obj=None, hindsight=None):
    """The run's human/LLM deliverable. With a journey (the normal flow path):
    hindsight (a judged dict or a {"skipped": reason} sentinel — NEVER able to flip the
    outcome) is spliced into the journey's advisory slot, journey.json is re-written
    final, and report.md becomes the journey's FULL render — a pure view of the same
    record, plus the evidence-trail appendix and any dod rollup. Valid hindsight
    learnings also ride into the knowledge-plane promotion below (records only — the
    in-memory ledger is NEVER mutated here: this step doesn't re-run on replay, so a
    live-vs-replay ledger divergence would trip the engine's determinism check)."""
    os.makedirs(slug_dir, exist_ok=True)
    retro_records = []
    synthesized_learning = None
    if (isinstance(hindsight, dict) and "skipped" not in hindsight
            and hindsight.get("hindsight_path")):
        path_entries = hindsight["hindsight_path"]
        methods = [str(entry.get("method", "")).strip() for entry in path_entries]
        promoted = hindsight.setdefault("promoted_learnings", [])
        already_repeated = any(
            method.lower() in str(learning).lower()
            for method in methods if method
            for learning in promoted)
        if not already_repeated:
            why = next((str(entry.get("why_available_earlier", "")).strip()
                        for entry in path_entries
                        if str(entry.get("why_available_earlier", "")).strip()), "")
            if not why:
                why = next((str(branch.get("why", "")).strip()
                            for branch in hindsight.get("avoidable_branches", [])
                            if str(branch.get("why", "")).strip()), "")
            why = why or "a more direct route was available"
            synthesized_learning = (
                f"With hindsight, a shorter route existed: {' → '.join(methods)} — {why}"
                [:LEARNINGS_MAX_CHARS])
            promoted.append(synthesized_learning)
    if journey_obj is not None:
        journey_obj = {**journey_obj, "hindsight": hindsight}
        _atomic_write(os.path.join(slug_dir, "journey.json"),
                      json.dumps(journey_obj, indent=2) + "\n")
        lines = [render_journey_report(journey_obj)]
        promoted = ((hindsight or {}).get("promoted_learnings") or [])
        # A validator-defying overlong list must not push the code-guaranteed synthesis
        # past the promotion cap: retain the first records but reserve one slot for it.
        promotion_learnings = promoted[:LEARNINGS_MAX_COUNT]
        if (synthesized_learning is not None
                and synthesized_learning not in promotion_learnings):
            promotion_learnings = (promotion_learnings[:LEARNINGS_MAX_COUNT - 1]
                                   + [synthesized_learning])
        retro_records = [
            {"cycle": cycles, "source": "retro", "kind": "fact",
             "text": str(l).strip()[:LEARNINGS_MAX_CHARS], "fp": harvest.fp(str(l)),
             "meta": {}}
            for l in promotion_learnings if str(l).strip()]
    else:
        lines = [f"# relentless-solve report", "",
                 f"OUTCOME: {outcome}   CYCLES: {cycles}", f"DETAIL: {detail}", ""]
    if requirements:
        lines.append(f"## Requirements rollup (last completed cycle)")
        lines += [f"- {rid}: {state}" for rid, state in sorted(requirements.items())]
        lines.append("")
    for kind, title in (("fact", "Established facts"), ("gap", "Known gaps / assumptions"),
                        ("dead-end", "Dead ends")):
        recs = [r for r in ledger if r["kind"] == kind]
        if recs:
            lines.append(f"## {title} ({len(recs)})")
            lines += [f"- [c{r['cycle']}·{r['source']}] {r['text']}" for r in recs]
            lines.append("")
    path = os.path.join(slug_dir, "report.md")
    _atomic_write(path, "\n".join(lines))
    # Global-tier promotion (knowledge plane, topology B) — through the module-level ctx
    # set at CLI time, NOT a new parameter: the relentless_flow call site must stay
    # byte-identical (flow_hash). Memoized step → not re-run on replay; append()'s fp
    # dedup backstops even a forced re-run. Promotion failure must never sink a run.
    if _KNOWLEDGE_CTX["enabled"] and _KNOWLEDGE_CTX["slug"]:
        try:
            knowledge.promote(list(ledger) + retro_records, _KNOWLEDGE_CTX["slug"],
                              _KNOWLEDGE_CTX["project"])
        except Exception as e:
            print(f"knowledge promotion failed (non-fatal): {e}", file=sys.stderr)
    return path


def render_journey_report(journey_obj):
    """The report body when a journey exists: the journey's FULL render — report.md is
    a PURE VIEW of journey.json (regenerable offline; the view cannot drift from the
    record). The ledger appendix write_report adds after this is the same data in its
    legacy grouped-by-kind shape, kept for grep-familiarity."""
    return journeylib.render_journey(journey_obj, level="FULL")


# ── the consolidated decision record (journey.json) + post-success hindsight ─────────────────
# journey.py owns the fold/render; these wrappers own the IO and the oneshot, and are
# module-level injectables (tests monkeypatch them like run_clarify/run_task).

def build_journey(slug_dir, slug, verdict, detail, receipts, trace, ledger):
    """Fold the run's trace + ledger into journey.json (hindsight still null — the
    report step splices it after retro/judge). Memoized as its own step so replay is
    byte-identical even though the fold itself is pure."""
    jobj = journeylib.fold_journey(slug, verdict, detail, receipts, trace, ledger)
    os.makedirs(slug_dir, exist_ok=True)
    _atomic_write(os.path.join(slug_dir, "journey.json"),
                  json.dumps(jobj, indent=2) + "\n")
    return jobj


def run_hindsight(jobj, slug_dir, timeout):
    """One hindsight oneshot against the journey's COMPACT render, citation-validated
    by code with one violation-echo retry. INVARIANT: this can never un-succeed a run —
    every failure mode returns a {"skipped": <reason>} sentinel, and the caller only
    ever splices the result into the journey's advisory `hindsight` slot."""
    out_path = os.path.join(slug_dir, "retro.json")
    if os.path.exists(out_path):
        # Artifact beats stdout, so quarantine a previous run's artifact before this
        # run can accidentally treat it as fresh judge output.
        os.replace(out_path, out_path + ".prior")
    compact = journeylib.render_journey(jobj, level="COMPACT")
    violations = None
    for _ in range(RETRO_ATTEMPTS):
        prompt = retro_envelope.hindsight_prompt(compact, out_path)
        if violations:
            prompt += retro_envelope.retry_suffix(violations)
        try:
            stdout = invoke_hermes(prompt, timeout)
        except Exception as e:
            return {"skipped": f"hindsight oneshot error ({e.__class__.__name__})"}
        try:
            with open(out_path, encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            obj = extract_json_object(stdout)
        if not isinstance(obj, dict):
            violations = ["no retro.json artifact and no parseable JSON in the reply"]
            continue
        violations = journeylib.validate_hindsight(obj, jobj)
        if not violations:
            return journeylib.stamp_tiers(jobj, obj)
        if os.path.exists(out_path):
            os.replace(out_path, out_path + ".rej")
    return {"skipped": f"hindsight failed validation after {RETRO_ATTEMPTS} attempts: "
                       f"{violations}"}


# ── pure ledger folds (replay-deterministic: functions of memoized results only) ─────────────

def fold_records(ledger, seen, records):
    fresh = 0
    for r in records:
        if r["fp"] in seen:
            continue
        seen.add(r["fp"])
        ledger.append(r)
        fresh += 1
    return fresh


def fold_clarify(tombstones, cycle, ledger, seen, source="clarify"):
    """source distinguishes LEVEL 0's whole-ledger clarify (default) from LEVEL 2's
    per-task scoped clarify ("scoped-clarify") in the ledger/report — same fp namespace
    (fp on the question) either way, so anti-flap dedup is unaffected by which level
    answered a given question first."""
    records = [{"cycle": cycle, "source": source,
                "kind": "fact" if t["status"] == "ANSWERED" else "gap",
                "text": t["evidence"], "fp": harvest.fp(t["question"]),
                "meta": {"question": t["question"]}} for t in tombstones]
    return fold_records(ledger, seen, records)


def fold_one(ledger, seen, cycle, source, kind, text):
    return fold_records(ledger, seen, [{"cycle": cycle, "source": source, "kind": kind,
                                        "text": text, "fp": harvest.fp(text), "meta": {}}])


# ── the flow ──────────────────────────────────────────────────────────────────────────────────

def relentless_flow(ctx, inp):
    inp = {**DEFAULTS, **(inp or {})}
    ledger, seen = [], set()  # rebuilt deterministically on every (re)play
    trace = []  # decision events for the journey fold — same rebuild discipline as ledger
    slug_dir = os.path.join(_HOME, "relentless", inp["slug"])
    # A definition of done travels as TEXT inside the immutable engine input (read at
    # CLI time), so every replay parses the exact spec the run started with — parse_dod
    # is pure, so this adds no step key and no journal entry. Requires the define-done
    # skill on disk whenever a --dod run plays (live or replay).
    dodctx, last_report = None, None
    if inp.get("dod"):
        spec = _spec()
        parsed = spec.parse_dod(inp["dod"])
        dodctx = {"parsed": parsed, "unmet": spec.unmet(parsed),
                  "known": set(spec.ids(parsed)),
                  "section": dod_requirements_section(parsed)}
    if inp.get("gate_note"):  # solve's gate verdict is the ledger's first receipt
        fold_one(ledger, seen, 0, "gate", "fact", inp["gate_note"])
    t0 = ctx.step("t0", lambda: time.time())
    outcome, detail = "max-cycles", f"max_cycles={inp['max_cycles']} reached without success"
    cycles_run = 0

    for cycle in range(inp["max_cycles"]):
        c = f"c{cycle}"
        now = ctx.step(f"{c}/clock", lambda: time.time())
        if now - t0 > inp["wallclock"]:
            outcome, detail = "wallclock", f"wall-clock budget hit before cycle {cycle}"
            break

        # A — CLARIFY: next-best questions over everything known so far. The run_dir gives
        # the investigator a per-cycle tombstone journal: a crash mid-clarify (the step is
        # not yet memoized) resumes the investigation instead of re-researching.
        inv = ctx.step(f"{c}/clarify",
                       lambda: run_clarify(inp["prompt"], [r["text"] for r in ledger], inp,
                                           run_dir=os.path.join(slug_dir, f"c{cycle}",
                                                                "clarify")))
        fresh_clar = fold_clarify(inv["tombstones"], cycle, ledger, seen)

        # B — PLAN: one task-decomposer oneshot against the re-rendered prompt.
        # Budget cascade (solve): this cycle's share = remaining/cycles-left, recomputed
        # each boundary from the memoized clock — an unspent share flows back automatically.
        # Replay-safe: derives only from immutable inp + the memoized `now`/`t0`/plan steps.
        # Three pools: plan 20%, task attempts 70% (n_tasks * attempts_per_task, LEVEL 2's
        # local retries), replan 10% (LEVEL 1's mid-cycle partial replans, MAX_REPLANS_PER_CYCLE
        # of them). replan_to defaults to plan_to (same posture as plan_to/task_to) when the
        # cascade is off (the bare `run` CLI path).
        plan_to, task_to = inp["plan_timeout"], inp["task_timeout"]
        replan_to, share = plan_to, None
        if inp.get("cascade"):
            share = (inp["wallclock"] - (now - t0)) / max(1, inp["max_cycles"] - cycle)
            plan_to = int(max(PLAN_TO_FLOOR, min(PLAN_TO_CAP, share * 0.2)))
            replan_to = int(max(REPLAN_TO_FLOOR, min(REPLAN_TO_CAP,
                                share * 0.1 / MAX_REPLANS_PER_CYCLE)))
        cycle_dir = os.path.join(slug_dir, f"c{cycle}")
        # k_in: everything the mapper is GIVEN this cycle (the completion report's delta
        # baseline); dead_fps: the binding never-re-propose set. Both pure fns of the
        # deterministically rebuilt ledger — no step keys.
        k_in = frozenset(seen)
        dead_fps = frozenset(r["fp"] for r in ledger if r["kind"] == "dead-end")
        rendered = render(inp["prompt"], ledger,  # pure fn of deterministic state
                          dod_section=dodctx["section"] if dodctx else None)
        ctx.step(f"{c}/render", lambda: persist(slug_dir, cycle, rendered, ledger))
        known_at_plan = len(ledger)  # the planner's evidence horizon (post-clarify)
        plan = ctx.step(f"{c}/plan", lambda: request_plan(slug_dir, inp["slug"], cycle,
                                                          rendered, plan_to,
                                                          dodctx=dodctx,
                                                          dead_fps=dead_fps))
        cycles_run = cycle + 1
        trace.append(journeylib.plan_event(
            f"{c}/plan", "plan", cycle, plan,
            {"elapsed": int(now - t0),
             "budget_remaining": int(inp["wallclock"] - (now - t0)),
             "share": int(share) if share is not None else None,
             "capability": inp["capability"]},
            known_at_plan))

        # E — the planner's human fork: ask under --gate, else assume-and-note (next
        # clarify ranks it). Exhaustion is a fact; re-declaring it is not fresh (anti-flap).
        if plan["disposition"] == "needs_decision":
            q = plan.get("question") or "(planner asked an unspecified question)"
            if inp["gate"]:
                ans = ctx.ask(f"{c}/fork", {"prompt": q, "type": "string"},
                              schema={"type": "string"})
                fresh_harv = fold_one(ledger, seen, cycle, "clarify", "fact",
                                      f"{q} -> {ans}")
            else:
                fresh_harv = fold_one(
                    ledger, seen, cycle, "assumption", "gap",
                    f"OPEN FORK (planner, unresolved): {q} -> ASSUMED: plan around the "
                    f"open fork next cycle; the next clarify round should rank it.")
        elif plan["disposition"] == "exhausted":
            fresh_harv = fold_one(
                ledger, seen, cycle, "harvest", "fact",
                "Planner declared exhaustion: no method outside the dead-end list remains "
                "under the stated constraints.")
        else:
            # C — EXECUTE + HARVEST via LEVEL 1 (run_intent_path): task sequencing, each
            # task's LEVEL 2 local retry, and the mid-cycle staleness gate + partial
            # replan. Task ids come from the MEMOIZED plan step, so a task's FIRST-attempt
            # step key (c<N>/t/<id>) is identical on replay and a crash resumes at the
            # first un-journaled task/retry/replan.
            if share is not None:
                n_tasks = len(plan["tasks"])
                attempts_per_task = 1 + inp.get("local_retry_budget",
                                                DEFAULTS["local_retry_budget"])
                task_to = int(max(TASK_TO_FLOOR, min(TASK_TO_CAP,
                                  share * 0.7 / max(1, n_tasks * attempts_per_task))))
            deadline = (now + share) if share is not None else None
            results, fresh_harv = run_intent_path(ctx, cycle, plan, cycle_dir, task_to,
                                                  replan_to, ledger, seen, inp, slug_dir,
                                                  deadline=deadline, dodctx=dodctx,
                                                  dead_fps=dead_fps, trace=trace)
            if dodctx:  # the completion contract — pure fn of memoized plan/results
                last_report = _reporter().completion_report(
                    plan, results, dod_parsed=dodctx["parsed"],
                    knowledge_in_fps=k_in, cycle=cycle)
            ctx.step(f"{c}/plan-out",
                     lambda: write_plan_receipt(cycle_dir, plan, results,
                                                report=last_report))
            # SUCCESS = every task verified worked; the plan contract makes the final
            # task an intent-verification task, so this equivalence stays in pure code.
            if results and all(r["verdict"] == "worked" for r in results):
                outcome = "success"
                detail = f"all {len(results)} plan tasks verified worked"
                break

        # D — STOP HONESTLY: relentless ≠ flailing; zero new information anywhere → dry
        if fresh_clar == 0 and fresh_harv == 0 and stop_is_converged(inv.get("stop_reason")):
            outcome, detail = "information-dry", (f"cycle {cycle} produced zero fresh facts "
                                                  f"(clarify converged, harvest all seen)")
            break

    # F — the CONSOLIDATED record (journey.json: Node = evidence/options/taken) and, on
    # a successful full (cascade) route with leftover budget, the ADVISORY hindsight
    # judge. Branch conditions derive only from memoized results + immutable input;
    # the judge's every failure mode is a skip sentinel — it can never un-succeed a run.
    receipts = {"route": "full" if inp.get("cascade") else "run",
                "cycles": cycles_run, "max_cycles": inp["max_cycles"],
                "wallclock": inp["wallclock"], "capability": inp["capability"],
                "stop_reason": outcome}
    jobj = ctx.step("retro/journey",
                    lambda: build_journey(slug_dir, inp["slug"], outcome, detail,
                                          receipts, trace, ledger))
    hs = {"skipped": "hindsight runs only on a successful full route"}
    if outcome == "success" and inp.get("cascade"):
        rnow = ctx.step("retro/clock", lambda: time.time())
        remaining = inp["wallclock"] - (rnow - t0)
        if remaining >= HINDSIGHT_TO_FLOOR:
            hs_to = int(max(HINDSIGHT_TO_FLOOR, min(HINDSIGHT_TO_CAP, remaining)))
            hs = ctx.step("retro/judge",
                          lambda: run_hindsight(jobj, slug_dir, hs_to))
        else:
            hs = {"skipped": f"no leftover budget for hindsight "
                             f"({int(remaining)}s remaining)"}
    rep = ctx.step("report",
                   lambda: write_report(slug_dir, outcome, ledger, cycles_run, detail,
                                        requirements=(last_report or {}).get("requirements"),
                                        journey_obj=jobj, hindsight=hs))
    return {"outcome": outcome, "cycles": cycles_run, "detail": detail,
            "n_facts": len(ledger), "report": rep}


# ── scope mode — CLARIFY → PLAN rounds, never EXECUTE (the pipeline prefix as a product) ─────
# The write-contract line (ARCHITECTURE.md): investigation writes KNOWLEDGE, planning
# writes a PROPOSAL, only EXECUTE writes the world. scope stops at the proposal: the
# deliverable is the scope package (facts + task breakdown + open decisions/questions)
# for a human to author from. Its own engine flow (id "relentless-scope") so
# relentless_flow's hashed source stays byte-identical.

SCOPE_DEFAULTS = {"rounds": 2, "scope_budget": 1800}


def _git(cwd, *args, timeout=60):
    """One git command → (rc, stdout). Never raises on git failure: scope's mechanical
    isolation DEGRADES (to directive + no-terminal) rather than sinking the run."""
    try:
        p = subprocess.run(["git", "-C", cwd] + list(args), capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"(git error: {e})"


def git_porcelain(path):
    """Sorted `git status --porcelain` lines, or None when path isn't a git worktree.
    Pure read — scope_flow memoizes it per round for the dirty receipts."""
    rc, out = _git(path, "status", "--porcelain")
    if rc != 0:
        return None
    return sorted(ln for ln in out.splitlines() if ln.strip())


def worktree_receipt(research_dir, baseline_commit):
    """Content-sensitive worktree identity, not merely filenames: replay containment
    must notice commit drift and edits whose porcelain shape happens to stay unchanged."""
    porcelain = git_porcelain(research_dir)
    head_rc, head = _git(research_dir, "rev-parse", "HEAD")
    diff_rc, diff = _git(research_dir, "diff", "--binary", baseline_commit)
    ok = porcelain is not None and head_rc == 0 and diff_rc == 0
    return {"porcelain": porcelain, "head": head.strip() if head_rc == 0 else "",
            "diff_sha256": (hashlib.sha256(diff.encode()).hexdigest()
                            if diff_rc == 0 else ""), "ok": ok}


def receipt_matches(a, b):
    """Compare only the stable receipt contract so diagnostic fields cannot alter
    containment decisions; an explicit failed read always fails closed."""
    if not a or not b or a.get("ok") is False or b.get("ok") is False:
        return False
    return all(a.get(key) == b.get(key)
               for key in ("porcelain", "head", "diff_sha256"))


def setup_research_worktree(answer_cwd, slug):
    """Create a sibling worktree with a content-sensitive baseline receipt. Carrying
    the caller patch is fail-closed because isolated research without the caller's own
    tracked changes would answer a subtly different question."""
    rc, _ = _git(answer_cwd, "rev-parse", "--show-toplevel")
    if rc != 0:
        return answer_cwd, {"isolated": False,
                            "reason": "answer-cwd is not a git repo — proceeding on the "
                                      "read directive + no-terminal toolset only"}
    rc, head = _git(answer_cwd, "rev-parse", "HEAD")
    if rc != 0:
        return answer_cwd, {"isolated": False, "reason": f"git rev-parse HEAD failed: {head}"}
    head = head.strip()
    scope_dir = os.path.join(_HOME, "relentless", slug, "scope")
    wt = os.path.join(scope_dir, "worktree")
    os.makedirs(scope_dir, exist_ok=True)
    moved_aside = None
    baseline_path = os.path.join(scope_dir, "worktree-baseline.json")
    patch_path = os.path.join(scope_dir, "worktree-callerdiff.patch")

    def fail(reason):
        _git(answer_cwd, "worktree", "remove", "--force", wt)
        _git(answer_cwd, "worktree", "prune")
        return answer_cwd, {"isolated": False, "reason": reason,
                            "moved_aside": moved_aside}

    if os.path.isdir(wt):
        prior_baseline = None
        try:
            with open(baseline_path, encoding="utf-8") as fh:
                prior_baseline = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        baseline_commit = ((prior_baseline or {}).get("head")
                           if isinstance(prior_baseline, dict) else None)
        cur = (worktree_receipt(wt, baseline_commit)
               if baseline_commit else None)
        if receipt_matches(cur, prior_baseline):
            reset_rc, reset_out = _git(wt, "reset", "--hard", head)
            clean_rc, clean_out = _git(wt, "clean", "-fd")
            if reset_rc != 0:
                return fail(f"worktree refresh reset failed: {reset_out.strip()[:200]}")
            if clean_rc != 0:
                return fail(f"worktree refresh clean failed: {clean_out.strip()[:200]}")
        else:
            n = 1
            while os.path.exists(f"{wt}.violated-{n}"):
                n += 1
            moved_aside = f"{wt}.violated-{n}"
            os.rename(wt, moved_aside)
            _git(answer_cwd, "worktree", "prune")
    if not os.path.isdir(wt):
        rc, out = _git(answer_cwd, "worktree", "add", "--detach", wt, "HEAD")
        if rc != 0:
            return fail(f"git worktree add failed: {out.strip()[:200]}")
    rc, diff = _git(answer_cwd, "diff", "--binary", "HEAD")
    if rc != 0:
        return fail(f"caller diff capture failed: {diff.strip()[:200]}")
    try:
        _atomic_write(patch_path, diff)
    except OSError as e:
        return fail(f"caller patch write failed: {e}")
    diff_applied = False
    if diff.strip():
        try:
            p = subprocess.run(["git", "-C", wt, "apply"], input=diff,
                               capture_output=True, text=True, timeout=60)
            diff_applied = p.returncode == 0
            if not diff_applied:
                detail = (p.stderr or p.stdout or "git apply failed").strip()[:200]
                return fail(f"caller patch apply failed: {detail}")
        except (OSError, subprocess.SubprocessError) as e:
            return fail(f"caller patch apply failed: {e}")
    rc, baseline_commit = _git(wt, "rev-parse", "HEAD")
    if rc != 0:
        return fail(f"baseline commit capture failed: {baseline_commit.strip()[:200]}")
    baseline_commit = baseline_commit.strip()
    captured = worktree_receipt(wt, baseline_commit)
    if not captured["ok"]:
        return fail("baseline receipt capture failed")
    baseline = {key: captured[key] for key in ("porcelain", "head", "diff_sha256")}
    try:
        _atomic_write(baseline_path, json.dumps(baseline) + "\n")
    except OSError as e:
        return fail(f"baseline receipt write failed: {e}")
    return wt, {"isolated": True, "worktree": wt, "head": head, "baseline": baseline,
                "diff_applied": diff_applied, "moved_aside": moved_aside}


def reset_research_worktree(research_dir):
    """Restore the recorded detached commit and caller patch, never a drifted HEAD,
    so containment produces the exact setup receipt rather than a merely clean tree."""
    artifact_dir = os.path.dirname(research_dir)
    baseline_path = os.path.join(artifact_dir, "worktree-baseline.json")
    patch_path = os.path.join(artifact_dir, "worktree-callerdiff.patch")
    try:
        with open(baseline_path, encoding="utf-8") as fh:
            baseline = json.load(fh)
        with open(patch_path, encoding="utf-8") as fh:
            patch = fh.read()
    except (OSError, json.JSONDecodeError, ValueError):
        fresh = worktree_receipt(research_dir, "")
        return {"ok": False, "steps": [], "receipt": fresh}
    baseline_commit = baseline.get("head", "")
    steps = []
    rc, _ = _git(research_dir, "reset", "--hard", baseline_commit)
    steps.append([f"reset --hard {baseline_commit}", rc])
    rc, _ = _git(research_dir, "clean", "-ffdx")
    steps.append(["clean -ffdx", rc])
    if patch:
        rc, _ = _git(research_dir, "apply", patch_path)
        steps.append(["apply caller patch", rc])
    fresh = worktree_receipt(research_dir, baseline_commit)
    ok = all(rc == 0 for _, rc in steps) and receipt_matches(fresh, baseline)
    return {"ok": ok, "steps": steps, "receipt": fresh}


def archive_violation_evidence(research_dir, baseline_commit, dest_dir):
    """Archive bounded audit evidence without following links or copying an unbounded
    tree; skipped files stay explicit in the manifest without defeating containment."""
    manifest = []
    diff_rc, diff = _git(research_dir, "diff", "--binary", baseline_commit)
    patch_ok = diff_rc == 0
    if patch_ok:
        try:
            os.makedirs(dest_dir, exist_ok=True)
            _atomic_write(os.path.join(dest_dir, "violation.patch"), diff)
        except OSError:
            patch_ok = False
    manifest.append({"path": "violation.patch",
                     "action": "copied" if patch_ok else "skipped",
                     "reason": "captured" if patch_ok else "capture failed",
                     "size": len(diff.encode()) if diff_rc == 0 else 0})
    files_rc, files = _git(research_dir, "ls-files", "--others", "--exclude-standard")
    copied = 0
    if files_rc == 0:
        for relpath in [line for line in files.splitlines() if line]:
            src = os.path.join(research_dir, relpath)
            action, reason, size = "skipped", "unreadable", 0
            try:
                st = os.lstat(src)
                size = st.st_size
                parts = relpath.split(os.sep)
                if os.path.isabs(relpath) or ".." in parts:
                    reason = "path escapes research worktree"
                elif ".git" in parts:
                    reason = "contains .git path component"
                elif not stat.S_ISREG(st.st_mode):
                    reason = "not a regular file"
                elif size > 1024 * 1024:
                    reason = "larger than 1 MiB"
                elif copied >= 20:
                    reason = "copy limit reached"
                else:
                    with open(src, "rb") as fh:
                        content = fh.read()
                    dest = os.path.join(dest_dir, "untracked", relpath)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    _atomic_write(dest, content)
                    action, reason = "copied", "copied"
                    copied += 1
            except OSError as e:
                reason = f"copy failed: {e}"
            manifest.append({"path": relpath, "action": action,
                             "reason": reason, "size": size})
    manifest_ok = True
    try:
        os.makedirs(dest_dir, exist_ok=True)
        _atomic_write(os.path.join(dest_dir, "manifest.json"),
                      json.dumps(manifest, indent=2) + "\n")
    except OSError:
        manifest_ok = False
    return {"ok": patch_ok and manifest_ok, "manifest": manifest}


def teardown_research_worktree(answer_cwd, info):
    """Remove only a worktree whose complete receipt still matches setup; commit or
    content drift is retained even when porcelain alone would look unchanged."""
    if not info.get("isolated"):
        return {"removed": False, "violation": False, "reason": info.get("reason")}
    wt = info["worktree"]
    baseline = info["baseline"]
    cur = worktree_receipt(wt, baseline["head"])
    if receipt_matches(cur, baseline):
        rc, _ = _git(answer_cwd, "worktree", "remove", "--force", wt)
        return {"removed": rc == 0, "violation": False}
    base = set(baseline.get("porcelain") or [])
    return {"removed": False, "violation": True,
            "new_dirt": [ln for ln in (cur.get("porcelain") or [])
                         if ln not in base][:20]}


def clean_ledger(ledger, tainted_rounds):
    """Exclude clarify evidence from violating rounds only where it could steer later
    research or planning; retaining the full ledger preserves the audit trail."""
    tainted = set(tainted_rounds)
    return [record for record in ledger
            if not (record.get("source") == "clarify"
                    and record.get("cycle") in tainted)]


def scope_flow(ctx, inp):
    """CLARIFY → PLAN rounds, never EXECUTE. Per round: grounded read-only clarify
    (investigator under the forced `read` capability, researching in the disposable
    sibling worktree), then one task-decomposer oneshot against the re-rendered
    prompt. Round verdicts: tasks → scoped (done); needs_decision → the question is
    folded as a gap so the NEXT round's clarify ranks and targets it (scope never
    assumes and never suspends — an unresolved decision is scope OUTPUT); exhausted →
    infeasible-as-stated (a valuable scoping outcome); zero fresh facts + converged →
    dry. dead_fps stays empty (no execution → no dead-ends). Emits the scope package
    (scope.md + scope.json) as its final step."""
    inp = {**DEFAULTS, **SCOPE_DEFAULTS, **(inp or {})}
    inp["capability"] = "read"  # forced — scope never crosses the world-write line
    ledger, seen = [], set()  # rebuilt deterministically on every (re)play
    slug_dir = os.path.join(_HOME, "relentless", inp["slug"], "scope")
    dodctx = None
    if inp.get("dod"):
        spec = _spec()
        parsed = spec.parse_dod(inp["dod"])
        dodctx = {"parsed": parsed, "unmet": spec.unmet(parsed),
                  "known": set(spec.ids(parsed)),
                  "section": dod_requirements_section(parsed)}
        if parsed.get("open"):  # the deferred OPEN→clarify seeding, live in scope mode
            fold_one(ledger, seen, 0, "dod", "gap",
                     f"OPEN (from the definition of done): {parsed['open']}")
    for fact in inp.get("seed_facts") or []:  # the human answer-back loop (--fact)
        fold_one(ledger, seen, 0, "human", "fact", fact)
    baseline = inp.get("worktree_baseline")
    t0 = ctx.step("t0", lambda: time.time())
    outcome, detail, rounds_run = "budget", "no round completed within budget", 0
    plan, inv, open_decision, violations = None, None, None, []
    tainted_rounds = []

    for rnd in range(inp["rounds"]):
        s = f"s{rnd}"
        now = ctx.step(f"{s}/clock", lambda: time.time())
        if now - t0 > inp["scope_budget"]:
            outcome, detail = "budget", f"scope budget hit before round {rnd}"
            break
        inv = ctx.step(f"{s}/clarify",
                       lambda r=rnd: run_clarify(inp["prompt"],
                                                 [x["text"] for x in clean_ledger(
                                                     ledger, tainted_rounds)], inp,
                                                 run_dir=os.path.join(slug_dir, f"s{r}",
                                                                      "clarify")))
        fresh = fold_clarify(inv["tombstones"], rnd, ledger, seen)
        if inp.get("research_dir") and baseline is not None:
            st = ctx.step(
                f"{s}/status",
                lambda b=baseline: worktree_receipt(inp["research_dir"], b["head"]))
            if not receipt_matches(st, baseline):
                current_porcelain = st.get("porcelain")
                baseline_porcelain = baseline.get("porcelain")
                if current_porcelain != baseline_porcelain:
                    if current_porcelain is None:
                        new_dirt = ["receipt mismatch: porcelain unavailable"]
                    else:
                        current_set = set(current_porcelain)
                        baseline_set = set(baseline_porcelain or [])
                        new_dirt = ([ln for ln in current_porcelain
                                     if ln not in baseline_set]
                                    + [f"porcelain removed: {ln}"
                                       for ln in (baseline_porcelain or [])
                                       if ln not in current_set])[:20]
                else:
                    if st.get("head") != baseline.get("head"):
                        new_dirt = ["receipt mismatch: head changed"]
                    elif st.get("diff_sha256") != baseline.get("diff_sha256"):
                        new_dirt = ["receipt mismatch: diff_sha256 changed"]
                    else:
                        new_dirt = ["receipt mismatch: receipt read failed"]
                violation = {"round": rnd, "new_dirt": new_dirt, "evidence": None}
                violations.append(violation)
                fold_one(ledger, seen, rnd, "isolation", "gap",
                         "WRITE-CONTRACT VIOLATION: the read-only research worktree "
                         f"receipt mismatched its baseline during round {rnd}: "
                         + "; ".join(new_dirt[:5]))
                tainted_rounds.append(rnd)
                evidence_dir = os.path.join(slug_dir, "violations", f"round-{rnd}")
                violation["evidence"] = evidence_dir
                ev = ctx.step(
                    f"{s}/evidence",
                    lambda r=rnd, b=baseline: archive_violation_evidence(
                        inp["research_dir"], b["head"],
                        os.path.join(slug_dir, "violations", f"round-{r}")))
                if not ev["ok"]:
                    outcome, detail = "containment-failed", (
                        f"evidence capture failed for violating round {rnd}")
                    break
                rr = ctx.step(f"{s}/reset",
                              lambda: reset_research_worktree(inp["research_dir"]))
                if not rr["ok"]:
                    outcome, detail = "containment-failed", (
                        f"worktree reset failed for violating round {rnd}")
                    break
        planning_ledger = clean_ledger(ledger, tainted_rounds)
        rendered = render(inp["prompt"], planning_ledger,
                          dod_section=dodctx["section"] if dodctx else None)
        ctx.step(f"{s}/render",
                 lambda r=rnd, body=rendered: persist(slug_dir, r, body, ledger))
        plan = ctx.step(f"{s}/plan",
                        lambda r=rnd, body=rendered: request_plan(
                            slug_dir, inp["slug"], r, body, inp["plan_timeout"],
                            dodctx=dodctx, dead_fps=frozenset()))
        rounds_run = rnd + 1
        if plan["disposition"] == "tasks":
            open_decision = None
            outcome = "scoped"
            detail = f"breakdown with {len(plan['tasks'])} tasks after {rounds_run} round(s)"
            break
        if plan["disposition"] == "exhausted":
            outcome, detail = "infeasible", (
                "the decomposer declared exhaustion: no viable method under the stated "
                "constraints — see the final plan artifact for its reasoning")
            break
        open_decision = plan.get("question") or "(decomposer asked an unspecified question)"
        fresh += fold_one(ledger, seen, rnd, "decomposer", "gap",
                          f"OPEN DECISION (blocks planning): {open_decision} — scope "
                          "targets it next round; a human answer unblocks the breakdown.")
        outcome = "open-decisions"
        detail = f"planning blocked on a decision after {rounds_run} round(s)"
        if fresh == 0 and stop_is_converged(inv.get("stop_reason")):
            outcome, detail = "dry", (f"round {rnd} produced zero fresh facts and "
                                      "clarify converged — more rounds will not "
                                      "sharpen the scope")
            break

    pkg = ctx.step("package",
                   lambda: write_scope_package(slug_dir, inp, outcome, detail, ledger,
                                               plan, dodctx, open_decision, inv,
                                               violations, rounds_run, tainted_rounds))
    return {"outcome": outcome, "rounds": rounds_run, "detail": detail,
            "n_records": len(ledger),
            "n_facts": sum(1 for record in ledger if record.get("kind") == "fact"),
            "scope_path": pkg["scope_path"], "plan_path": pkg["plan_path"],
            "violations": len(violations), "tainted_rounds": tainted_rounds}


def write_scope_package(slug_dir, inp, outcome, detail, ledger, plan, dodctx,
                        open_decision, inv, violations, rounds, tainted_rounds):
    """The scope deliverable: scope.md (human) + scope.json (the DIRECT layer's
    machine-readable half of the subroutine contract). Distinguishes UNANSWERABLE
    (attempted, NOT_FOUND, with reasons) from NEXT QUESTIONS (EVSI-ranked, above
    floor, never attempted) — they demand different actions from the reader. Also
    promotes the run's facts to the global tier (scope learns for later solve calls)."""
    os.makedirs(slug_dir, exist_ok=True)
    lines = ["# scope package", "",
             f"VERDICT: {outcome}   ROUNDS: {rounds}", f"DETAIL: {detail}", "",
             "## Intent", "", inp["prompt"].rstrip(), ""]
    if dodctx:
        unmet = ", ".join(dodctx["unmet"]) or "(none)"
        lines += ["## What done means", "",
                  f"Definition of done: {len(dodctx['known'])} requirement id(s); "
                  f"unmet at scope time: {unmet}", ""]
    facts = [r for r in ledger if r["kind"] == "fact"]
    if facts:
        lines.append(f"## Facts learned ({len(facts)})")
        for r in facts:
            flagged = r["source"] == "clarify" and r["cycle"] in tainted_rounds
            prefix = "⚠ " if flagged else ""
            suffix = " **(violating round — re-verify)**" if flagged else ""
            lines.append(f"- {prefix}[s{r['cycle']}·{r['source']}] {r['text']}{suffix}")
        lines.append("")
    if plan and plan.get("disposition") == "tasks":
        lines.append("## Proposed task breakdown (the authoring work-list)")
        if tainted_rounds:
            lines.append("Some facts came from a tainted round and should be re-verified.")
        for t in plan["tasks"]:
            lines.append(f"- **{t['id']}** [{t['method']}] {t['description']}")
            lines.append(f"  - done when: {t['success_criterion']}")
        lines.append("")
    if open_decision:
        lines += ["## Open decisions (answer to unblock the breakdown)", "",
                  f"- {open_decision}", ""]
    gaps = [t for t in ((inv or {}).get("tombstones") or [])
            if t.get("status") == "NOT_FOUND"]
    if gaps:
        lines.append("## Unanswerable (attempted, came back NOT_FOUND)")
        lines += [f"- {t.get('question', '')}: {t.get('fact', '')}" for t in gaps]
        lines.append("")
    nq = (inv or {}).get("next_questions") or []
    if nq:
        lines.append("## Next questions (EVSI-ranked — answer these, in this order, "
                     "to sharpen the scope; re-run with --fact)")
        lines += [f"- ({q.get('value', 0):.2f}) {q.get('question', '')}" for q in nq]
        lines.append("")
    if violations:
        lines.append("## WRITE-CONTRACT VIOLATIONS (read-only research modified files)")
        for v in violations:
            lines.append(f"- round {v['round']}: " + "; ".join(v["new_dirt"]))
        lines.append("")
    scope_path = os.path.join(slug_dir, "scope.md")
    _atomic_write(scope_path, "\n".join(lines))
    plan_path = None
    if rounds:
        planfile, _ = _decomposer()
        plan_path = planfile.plan_path(os.path.join(slug_dir, f"c{rounds - 1}"))
    obj = {"slug": inp["slug"], "verdict": outcome, "detail": detail, "rounds": rounds,
           "scope_path": scope_path, "plan_path": plan_path, "n_facts": len(facts),
           "open_decision": open_decision, "next_questions": nq,
           "violations": violations, "tainted_rounds": tainted_rounds}
    _atomic_write(os.path.join(slug_dir, "scope.json"),
                  json.dumps(obj, indent=2) + "\n")
    if _KNOWLEDGE_CTX["enabled"] and _KNOWLEDGE_CTX["slug"]:
        try:
            knowledge.promote(clean_ledger(ledger, tainted_rounds),
                              _KNOWLEDGE_CTX["slug"], _KNOWLEDGE_CTX["project"])
        except Exception as e:
            print(f"knowledge promotion failed (non-fatal): {e}", file=sys.stderr)
    return {"scope_path": scope_path, "plan_path": plan_path}


# ── solve route handlers ──────────────────────────────────────────────────────────────────────

def _solve_report(slug_dir, verdict, budget, spent, stop, body):
    """report.md with the receipted-defaults header — every derived/adaptive choice visible."""
    os.makedirs(slug_dir, exist_ok=True)
    head = (f"# solve report\n\n"
            f"SLUG: {verdict['slug']}   ROUTE: {verdict['route']} ({verdict['source']}: "
            f"{verdict['why']})\nBUDGET: total={budget}s spent={int(spent)}s   "
            f"RISK: {verdict['risk']}\nSTOP: {stop}\n\n")
    path = os.path.join(slug_dir, "report.md")
    _atomic_write(path, head + body)
    return path


def write_solve_json(slug_dir, verdict, outcome, detail, report_path, spent, artifacts,
                     journey=None):
    """B2 — the subroutine result contract: solve.json is the DIRECT layer's (the dev
    loop's) machine-readable API, written by EVERY route beside the human report.md.
    `artifacts` keys are pinned per route (trivial: answer · single_method: plan_tree ·
    full: ledger, last_plan · all routes: journey) so a caller never guesses. See
    SKILL.md for exit codes."""
    artifacts = {**artifacts, "journey": journey}
    obj = {"slug": verdict["slug"], "route": verdict["route"], "outcome": outcome,
           "detail": detail, "report_path": report_path, "spent_s": int(spent),
           "artifacts": artifacts}
    os.makedirs(slug_dir, exist_ok=True)
    _atomic_write(os.path.join(slug_dir, "solve.json"), json.dumps(obj, indent=2) + "\n")
    return obj


def _engine_result(state_dir):
    """Tolerant read of the engine's state.json → the flow's result dict (or None) —
    how the full route's solve.json learns its outcome after run_cli returns."""
    try:
        with open(os.path.join(state_dir, "state.json"), encoding="utf-8") as fh:
            st = json.load(fh)
        r = st.get("result")
        return r if isinstance(r, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _refresh_solve_json_after_resume(slug, state_dir):
    """Best-effort refresh of a full-route solve.json after resume. gate.json restores
    the original verdict receipt and engine state supplies the terminal result; spent_s
    is necessarily 0 because resume cannot recover the original invocation's elapsed
    time. Missing, malformed, or non-full gates are advisory no-ops, never resume errors."""
    slug_dir = os.path.join(_HOME, "relentless", slug)
    gate_path = os.path.join(slug_dir, "gate.json")
    try:
        with open(gate_path, encoding="utf-8") as fh:
            verdict = json.load(fh)
    except FileNotFoundError:
        print(f"solve.json not refreshed: gate.json missing for {slug}", file=sys.stderr)
        return
    except (OSError, json.JSONDecodeError, ValueError):
        print(f"solve.json not refreshed: gate.json is unreadable for {slug}",
              file=sys.stderr)
        return
    if not isinstance(verdict, dict) or verdict.get("route") != "full":
        print(f"solve.json not refreshed: gate route is not full for {slug}",
              file=sys.stderr)
        return

    terminal = _engine_result(state_dir)
    res = terminal or {}
    cyc = res.get("cycles")
    journey_path = os.path.join(slug_dir, "journey.json")
    write_solve_json(
        slug_dir, verdict, res.get("outcome", "engine-rc-0"), res.get("detail", ""),
        res.get("report") or os.path.join(slug_dir, "report.md"), 0,
        {"ledger": os.path.join(slug_dir, "ledger.jsonl"),
         "last_plan": (os.path.join(slug_dir, f"c{cyc - 1}", "plan.json")
                       if cyc else None)},
        journey=(journey_path if terminal and os.path.exists(journey_path) else None))


def _write_degenerate_journey(slug_dir, verdict, outcome, detail, method, evidence):
    """One journey.json schema for EVERY route, however small the run — the loopless
    routes get a one-node chain so downstream consumers never special-case them."""
    jobj = journeylib.degenerate(verdict["slug"], outcome, detail,
                                 {"route": verdict["route"], "cycles": 0,
                                  "stop_reason": outcome},
                                 method, verdict.get("why", ""), evidence)
    os.makedirs(slug_dir, exist_ok=True)
    _atomic_write(os.path.join(slug_dir, "journey.json"),
                  json.dumps(jobj, indent=2) + "\n")


def solve_trivial(intent, slug_dir, verdict, budget, oneshot=None):
    """One bare pass-through answer; no tree, no ledger, no loop machinery."""
    t0 = time.time()
    answer = (oneshot or run_oneshot)(intent, timeout=min(budget, 600))
    stop = "answered" if answer else "empty response (backend no-op)"
    path = _solve_report(slug_dir, verdict, budget, time.time() - t0, stop,
                         f"## Answer\n\n{answer or '(none)'}\n")
    outcome = "answered" if answer else "empty"
    _write_degenerate_journey(slug_dir, verdict, outcome, stop, "direct-answer",
                              answer or "(empty response)")
    write_solve_json(slug_dir, verdict, outcome, stop, path,
                     time.time() - t0, {"answer": answer or ""},
                     journey=os.path.join(slug_dir, "journey.json"))
    print(path)
    return 0 if answer else 1


def solve_single(intent, slug_dir, verdict, budget, risk, drive=None):
    """One method-explorer run (all-but-reserve of the budget); no clarify loop."""
    t0 = time.time()
    slug = f"{verdict['slug']}-single"
    extra = ("HARD CONSTRAINT: read-only — do not modify any external state.\n"
             if risk == "read" else "")
    prompt = _envelope().real_prompt(intent, slug, PLANS_DIR, extra=extra)
    os.makedirs(slug_dir, exist_ok=True)
    ppath = os.path.join(slug_dir, "prompt-single.md")
    _atomic_write(ppath, prompt)
    dcfg = {**DEFAULTS["drive"], "wallclock": max(300, budget - 60)}
    st = (drive or run_drive)(slug, ppath, dcfg)
    ok = st.get("status") == "SUCCESS"
    path = _solve_report(slug_dir, verdict, budget, time.time() - t0,
                         f"drive {st.get('status')}",
                         f"## Outcome\n\n{st.get('detail', '')}\n\nplan-tree: "
                         f"{PLANS_DIR}/{slug}/plan-tree.md\n")
    _write_degenerate_journey(slug_dir, verdict, st.get("status", "?"),
                              st.get("detail", ""), "method-explorer drive",
                              f"drive ended {st.get('status')}: {st.get('detail', '')} "
                              f"(plan tree: {PLANS_DIR}/{slug}/plan-tree.md)")
    write_solve_json(slug_dir, verdict, st.get("status", "?"), st.get("detail", ""),
                     path, time.time() - t0,
                     {"plan_tree": f"{PLANS_DIR}/{slug}/plan-tree.md"},
                     journey=os.path.join(slug_dir, "journey.json"))
    print(path)
    return 0 if ok else 1


def cmd_solve(args, engine_run):
    intent = _read_prompt(args)
    if intent is None:
        return 2
    slug = args.slug or derive_slug(intent)
    os.environ["RELENTLESS_ACTIVE"] = slug  # B1 — children inherit; main() refuses nested
    project = knowledge.project_key(args.answer_cwd)
    set_knowledge_ctx(getattr(args, "knowledge", "on") != "off", project, slug)
    slug_dir = os.path.join(_HOME, "relentless", slug)
    gate_path = os.path.join(slug_dir, "gate.json")

    if os.path.exists(gate_path):  # idempotent resume: never re-classify
        with open(gate_path, encoding="utf-8") as fh:
            verdict = json.load(fh)
        verdict["source"] = "reused"
    elif args.route:
        verdict = {"route": args.route, "why": "forced via --route", "source": "flag"}
    else:
        verdict = classify(intent, args.risk)
    verdict.update({"slug": slug, "risk": args.risk,
                    "budget": {"total": args.budget,
                               "splits": "trivial: one call · single_method: all-minus-60s "
                                         "to drive · full: per-cycle share of remaining "
                                         "(20% plan / 70% task attempts incl. local "
                                         "retries / 10% mid-cycle partial replans)"}})
    if verdict["source"] != "reused":
        os.makedirs(slug_dir, exist_ok=True)
        _atomic_write(gate_path, json.dumps(verdict, indent=2) + "\n")
    if args.gate_only:
        print(json.dumps(verdict, indent=2))
        return 0

    route = verdict["route"]
    if route == "trivial":
        return solve_trivial(intent, slug_dir, verdict, args.budget)
    if route == "single_method":
        return solve_single(intent, slug_dir, verdict, args.budget, args.risk)
    inp = {"prompt": intent, "slug": slug, "max_cycles": DEFAULTS["max_cycles"],
           "wallclock": args.budget, "k": DEFAULTS["k"],
           "inv_rounds": DEFAULTS["inv_rounds"], "floor": DEFAULTS["floor"],
           "capability": args.risk, "answer_cwd": args.answer_cwd, "gate": args.gate,
           "cascade": True,
           "knowledge": getattr(args, "knowledge", "on"), "project": project,
           "gate_note": f"GATE: route=full ({verdict['source']}) — {verdict['why']}"}
    if getattr(args, "dod", None):
        inp["dod"] = _load_dod(args.dod)
    t0 = time.time()
    rc = engine_run(inp, slug, args)
    state_dir = getattr(args, "state_dir", None) or os.path.join(
        _HOME, "relentless", slug, "flow")
    terminal = _engine_result(state_dir)
    res = terminal or {}
    cyc = res.get("cycles")
    write_solve_json(
        slug_dir, verdict, res.get("outcome", f"engine-rc-{rc}"), res.get("detail", ""),
        res.get("report") or os.path.join(slug_dir, "report.md"), time.time() - t0,
        {"ledger": os.path.join(slug_dir, "ledger.jsonl"),
         "last_plan": (os.path.join(slug_dir, f"c{cyc - 1}", "plan.json")
                       if cyc else None)},
        journey=(os.path.join(slug_dir, "journey.json")
                 if terminal and os.path.exists(os.path.join(slug_dir, "journey.json"))
                 else None))
    return rc


def _merge_scope_json(slug, extra):
    """Post-run isolation info into scope.json (CLI-time; the flow already wrote the
    core). Tolerant: a flow that failed before packaging leaves no scope.json — write
    one holding just the isolation record rather than losing it."""
    path = os.path.join(_HOME, "relentless", slug, "scope", "scope.json")
    obj = {}
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    obj.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_write(path, json.dumps(obj, indent=2) + "\n")


def cmd_scope(args, engine_run_scope):
    """scope = "tell me what you'd do" (solve = "do it"). No gate/routing — scoping IS
    the full route's CLARIFY→PLAN prefix by definition. Capability is FORCED to read;
    grounded research runs in a disposable sibling worktree of --answer-cwd (see
    setup_research_worktree); the deliverable is the scope package."""
    intent = _read_prompt(args)
    if intent is None:
        return 2
    slug = args.slug or derive_slug(intent)
    os.environ["RELENTLESS_ACTIVE"] = slug  # B1
    # Project identity from the CALLER's tree (pre-worktree): worktrees of one repo
    # share a key (git common dir), so scope facts seed later solve calls anywhere.
    project = knowledge.project_key(args.answer_cwd)
    set_knowledge_ctx(args.knowledge != "off", project, slug)
    research_cwd, iso = args.answer_cwd, {"isolated": False, "reason": "no --answer-cwd"}
    if args.answer_cwd:
        research_cwd, iso = setup_research_worktree(args.answer_cwd, slug)
        if not iso["isolated"]:
            print(f"scope: mechanical isolation unavailable — {iso['reason']}",
                  file=sys.stderr)
    seeds = list(args.fact or [])
    if args.evidence_file:
        with open(args.evidence_file, encoding="utf-8") as fh:
            seeds += [ln.strip() for ln in fh if ln.strip()]
    inp = {"prompt": intent, "slug": slug, "rounds": args.rounds,
           "scope_budget": args.budget, "k": args.k, "inv_rounds": args.inv_rounds,
           "floor": args.floor, "capability": "read", "answer_cwd": research_cwd,
           "knowledge": args.knowledge, "project": project, "seed_facts": seeds,
           "plan_timeout": args.plan_timeout,
           "research_dir": research_cwd if iso.get("isolated") else None,
           "worktree_baseline": iso.get("baseline")}
    if args.dod:
        inp["dod"] = _load_dod(args.dod)
    rc = engine_run_scope(inp, slug, args)
    had_violation = False
    try:
        with open(os.path.join(_HOME, "relentless", slug, "scope", "scope.json"),
                  encoding="utf-8") as fh:
            had_violation = bool(json.load(fh).get("tainted_rounds"))
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    if iso.get("isolated"):
        td = teardown_research_worktree(args.answer_cwd, iso)
        if td.get("violation"):
            print("scope: WRITE-CONTRACT VIOLATION — the research worktree diverged "
                  f"from its baseline and was KEPT for audit: {iso['worktree']}",
                  file=sys.stderr)
        _merge_scope_json(slug, {"isolation": {
            "isolated": True, "worktree": iso["worktree"], "head": iso["head"],
            "diff_applied": iso["diff_applied"], "moved_aside": iso.get("moved_aside"),
            "removed": td.get("removed", False),
            "currently_diverged": td.get("violation", False),
            "had_violation": had_violation,
            "new_dirt": td.get("new_dirt", [])}})
    else:
        _merge_scope_json(slug, {"isolation": {"isolated": False,
                                               "reason": iso.get("reason"),
                                               "had_violation": had_violation}})
    return rc


def _read_prompt(args):
    if getattr(args, "prompt_file", None):
        text = sys.stdin.read() if args.prompt_file == "-" else open(
            args.prompt_file, encoding="utf-8").read()
    else:
        text = args.prompt
    if not (text or "").strip():
        print("need --prompt or --prompt-file", file=sys.stderr)
        return None
    return text


def _load_dod(path):
    """Read + lint a dod.md at CLI time; refuse loudly on lint ERRORS (a dishonest spec
    poisons every cycle) — warnings pass through to stderr. Returns the TEXT: it rides
    inside the engine input so replays parse the exact spec the run started with."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    spec = _spec()
    errors, warnings = spec.lint(spec.parse_dod(text))
    for w in warnings:
        print(f"dod lint warning: {w}", file=sys.stderr)
    if errors:
        raise SystemExit("refusing --dod: the spec fails lint:\n"
                         + "\n".join(f"  - {e}" for e in errors))
    return text


# ── CLI (delegates to the resumable-script engine) ────────────────────────────────────────────

def _load_engine():
    sys.path.insert(0, _ENGINE_DIR)
    try:
        from engine import flow, run_cli  # noqa: E402
    except ImportError as e:
        raise SystemExit(f"relentless-solve requires the resumable-script engine (engine.py). "
                         f"Looked in {_ENGINE_DIR!r}. Set RESUMABLE_ENGINE_DIR or sync the "
                         f"skill to $HERMES_HOME/skills/resumable-script.") from e
    return flow, run_cli


def _main(argv=None):
    p = argparse.ArgumentParser(description="Relentlessly solve a prompt: clarify → execute → "
                                            "harvest failures → repeat.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start (or resume-replay) a run")
    r.add_argument("--prompt", help="the intent, verbatim (immutable)")
    r.add_argument("--prompt-file", help="read the prompt from a file ('-' for stdin)")
    r.add_argument("--slug", required=True)
    r.add_argument("--max-cycles", type=int, default=DEFAULTS["max_cycles"])
    r.add_argument("--wallclock", type=int, default=DEFAULTS["wallclock"],
                   help="outer budget, seconds (checked at cycle boundaries)")
    r.add_argument("--k", type=int, default=DEFAULTS["k"])
    r.add_argument("--inv-rounds", type=int, default=DEFAULTS["inv_rounds"])
    r.add_argument("--floor", type=float, default=DEFAULTS["floor"])
    r.add_argument("--capability", choices=["act", "experiment", "read"],
                   default=DEFAULTS["capability"])
    r.add_argument("--answer-cwd", help="where the clarify answerer researches — pin to the "
                                        "target project dir (required in practice)")
    r.add_argument("--gate", action="store_true",
                   help="suspend (exit 10) on an unresolvable GUARD-HALT fork instead of "
                        "assume-and-note; resume with `resume --answer`")
    r.add_argument("--plan-timeout", type=int, default=DEFAULTS["plan_timeout"],
                   help="seconds per planning oneshot (cascade may lower it)")
    r.add_argument("--task-timeout", type=int, default=DEFAULTS["task_timeout"],
                   help="seconds per task-execution oneshot (cascade may adjust it)")
    r.add_argument("--local-retry-budget", type=int, default=DEFAULTS["local_retry_budget"],
                   help="LEVEL 2: extra local reattempts of a failed task, each preceded "
                        "by a scoped clarify, before escalating to the next cycle")
    r.add_argument("--dod", help="path to a define-done dod.md: plans must serve its "
                                 "unmet R-ids (binding coverage) and each cycle lands a "
                                 "c<N>/report.json completion report")
    r.add_argument("--knowledge", choices=["on", "off"], default="on",
                   help="off = hermetic: no global-tier seeding, no promotion")
    r.add_argument("--allow-nested", action="store_true",
                   help="override the nested-invocation guard (see SKILL.md)")
    r.add_argument("--state-dir", help="engine state dir (default "
                                       "$HERMES_HOME/relentless/<slug>/flow)")
    r.add_argument("--accept-flow-change", action="store_true")

    sc = sub.add_parser("scope", help="CLARIFY → PLAN rounds, never EXECUTE: emit a "
                                      "scope package (facts, task breakdown, open "
                                      "decisions/questions) for a human to author from")
    sc.add_argument("--prompt", help="the intent to scope")
    sc.add_argument("--prompt-file", help="read the intent from a file ('-' for stdin)")
    sc.add_argument("--slug", help="override the derived slug")
    sc.add_argument("--rounds", type=int, default=SCOPE_DEFAULTS["rounds"],
                    help="max clarify→plan rounds (stops early on a valid breakdown)")
    sc.add_argument("--budget", type=int, default=SCOPE_DEFAULTS["scope_budget"],
                    help="outer scope budget, seconds (checked at round boundaries)")
    sc.add_argument("--k", type=int, default=DEFAULTS["k"])
    sc.add_argument("--inv-rounds", type=int, default=DEFAULTS["inv_rounds"])
    sc.add_argument("--floor", type=float, default=DEFAULTS["floor"])
    sc.add_argument("--answer-cwd", help="the caller's project dir — research runs in "
                                         "a disposable sibling worktree created FROM it")
    sc.add_argument("--dod", help="path to a define-done dod.md (its OPEN line seeds "
                                  "the first clarify; coverage binds the breakdown)")
    sc.add_argument("--fact", action="append",
                    help="a human-supplied fact (repeatable) — the answer-back loop "
                         "for a prior package's Next questions")
    sc.add_argument("--evidence-file", help="file of human facts, one per line")
    sc.add_argument("--plan-timeout", type=int, default=DEFAULTS["plan_timeout"])
    sc.add_argument("--knowledge", choices=["on", "off"], default="on",
                    help="off = hermetic: no global-tier seeding, no promotion")
    sc.add_argument("--allow-nested", action="store_true",
                    help="override the nested-invocation guard (see SKILL.md)")
    sc.add_argument("--state-dir", help="engine state dir (default "
                                        "$HERMES_HOME/relentless/<slug>/scope-flow)")
    sc.add_argument("--accept-flow-change", action="store_true")

    z = sub.add_parser("resume", help="answer a suspended --gate fork and continue")
    z.add_argument("--slug", required=True)
    z.add_argument("--answer", required=True)
    z.add_argument("--key")
    z.add_argument("--state-dir")
    z.add_argument("--accept-flow-change", action="store_true")
    z.add_argument("--allow-nested", action="store_true",
                   help="override the nested-invocation guard (see SKILL.md)")

    s = sub.add_parser("solve", help="one-argument entry: gate → route → run "
                                     "(everything else derived/adaptive, receipted)")
    s.add_argument("--prompt", help="the intent — the only required input")
    s.add_argument("--prompt-file", help="read the intent from a file ('-' for stdin)")
    s.add_argument("--budget", type=int, default=SOLVE_BUDGET,
                   help=f"total wallclock seconds, ONE pool (default {SOLVE_BUDGET}); "
                        "routes subdivide it")
    s.add_argument("--risk", choices=["act", "experiment", "read"], default="act",
                   help="may the agent touch the world? (maps to clarify capability + "
                        "planner constraints)")
    s.add_argument("--gate", action="store_true",
                   help="suspend on an unresolvable GUARD-HALT fork (as in `run`)")
    s.add_argument("--slug", help="override the derived slug")
    s.add_argument("--route", choices=list(GATE_ROUTES), help="force a route (skip classify)")
    s.add_argument("--gate-only", action="store_true", help="print the verdict and exit")
    s.add_argument("--answer-cwd", help="where the clarify answerer researches (full route)")
    s.add_argument("--knowledge", choices=["on", "off"], default="on",
                   help="off = hermetic: no global-tier seeding, no promotion")
    s.add_argument("--allow-nested", action="store_true",
                   help="override the nested-invocation guard (see SKILL.md)")
    s.add_argument("--dod", help="path to a define-done dod.md (full route only: binding "
                                 "coverage + per-cycle completion reports)")
    s.add_argument("--state-dir")
    s.add_argument("--accept-flow-change", action="store_true")

    args = p.parse_args(argv)

    # B1 — the mechanical recursion guard (topology B, ARCHITECTURE.md): exactly one
    # layer (the dev loop) sits above relentless; an EXECUTE task or clarify answerer
    # spawned by a run inherits RELENTLESS_ACTIVE and must not start another run.
    # Exit 4 collides with nothing the engine emits (0/1/2/3/10/11/12/13).
    active = os.environ.get("RELENTLESS_ACTIVE")
    if active and not getattr(args, "allow_nested", False):
        print(f"nested relentless invocation detected (RELENTLESS_ACTIVE={active}): "
              "the dev loop must be the only layer above relentless — a task or "
              "answerer must not start another run. An oversized task should return "
              "verdict needs_split instead. Pass --allow-nested to override.",
              file=sys.stderr)
        return 4

    def engine_run(inp, slug, a):
        flow, run_cli = _load_engine()
        # v5: the consolidated decision record — retro/journey (always), retro/clock +
        # retro/judge (success + cascade + leftover budget only). UNLIKE the v3->v4 bump,
        # relentless_flow's OWN source changed this time, so the engine's flow_hash()
        # ALSO trips and an in-flight v4 journal needs --accept-flow-change to resume;
        # the new step keys are all AFTER the loop, so an accepted v4 journal replays its
        # completed prefix cleanly and only the retro/report tail runs fresh. v1-v3
        # journals still fail loudly on resume as before.
        FLOW = flow(id="relentless-solve", version=5)(relentless_flow)
        # Engine default state dir is per-flow-id (shared across slugs) — always pin per-slug.
        state_dir = getattr(a, "state_dir", None) or os.path.join(
            _HOME, "relentless", slug, "flow")
        eng_argv = ["run", "--input", json.dumps(inp), "--state-dir", state_dir]
        if not inp.get("gate"):
            eng_argv.append("--auto")
        if getattr(a, "accept_flow_change", False):
            eng_argv.append("--accept-flow-change")
        return run_cli(FLOW, argv=eng_argv)

    def engine_run_scope(inp, slug, a):
        flow, run_cli = _load_engine()
        # Its own flow id + state dir: relentless_flow's hash is untouched, and a
        # scope run never collides with a later solve run on the same slug.
        FLOW = flow(id="relentless-scope", version=2)(scope_flow)
        state_dir = getattr(a, "state_dir", None) or os.path.join(
            _HOME, "relentless", slug, "scope-flow")
        eng_argv = ["run", "--input", json.dumps(inp), "--state-dir", state_dir,
                    "--auto"]  # scope has no ctx.ask — never suspends
        if getattr(a, "accept_flow_change", False):
            eng_argv.append("--accept-flow-change")
        return run_cli(FLOW, argv=eng_argv)

    if args.cmd == "solve":
        return cmd_solve(args, engine_run)

    if args.cmd == "scope":
        return cmd_scope(args, engine_run_scope)

    if args.cmd == "run":
        prompt = _read_prompt(args)
        if prompt is None:
            return 2
        os.environ["RELENTLESS_ACTIVE"] = args.slug  # B1
        project = knowledge.project_key(args.answer_cwd)
        set_knowledge_ctx(args.knowledge != "off", project, args.slug)
        inp = {"prompt": prompt, "slug": args.slug, "max_cycles": args.max_cycles,
               "wallclock": args.wallclock, "k": args.k, "inv_rounds": args.inv_rounds,
               "floor": args.floor, "capability": args.capability,
               "answer_cwd": args.answer_cwd, "gate": args.gate,
               "plan_timeout": args.plan_timeout, "task_timeout": args.task_timeout,
               "local_retry_budget": args.local_retry_budget,
               "knowledge": args.knowledge, "project": project}
        if args.dod:
            inp["dod"] = _load_dod(args.dod)
        return engine_run(inp, args.slug, args)

    os.environ["RELENTLESS_ACTIVE"] = args.slug  # B1 (resume path)
    # resume has no answer_cwd; a promotion from a resumed run carries project=None
    # (never seeds — acceptable: the original run's promotion usually landed already,
    # and fp dedup keeps this from duplicating records).
    set_knowledge_ctx(True, None, args.slug)
    flow, run_cli = _load_engine()
    FLOW = flow(id="relentless-solve", version=5)(relentless_flow)
    state_dir = args.state_dir or os.path.join(_HOME, "relentless", args.slug, "flow")
    eng_argv = ["resume", "--answer", args.answer, "--state-dir", state_dir]
    if args.key:
        eng_argv += ["--key", args.key]
    if args.accept_flow_change:
        eng_argv.append("--accept-flow-change")
    rc = run_cli(FLOW, argv=eng_argv)
    if rc == 0:
        _refresh_solve_json_after_resume(args.slug, state_dir)
    return rc


def main(argv=None):
    """Restore RELENTLESS_ACTIVE so sequential in-process calls retain subroutine posture."""
    prior = os.environ.get("RELENTLESS_ACTIVE")
    try:
        return _main(argv)
    finally:
        if prior is None:
            os.environ.pop("RELENTLESS_ACTIVE", None)
        else:
            os.environ["RELENTLESS_ACTIVE"] = prior


if __name__ == "__main__":
    sys.exit(main())

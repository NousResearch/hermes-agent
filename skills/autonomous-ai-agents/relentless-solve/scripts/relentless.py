#!/usr/bin/env python3
"""relentless.py — clarify → execute → harvest-failures → repeat, until the intent holds.

The deterministic L3 loop over two existing skills (no LLM calls in this file):
  CLARIFY  — investigator iterate.py (in-process): rank next-best questions by EVSI, research
             the top-K with a full Hermes agent, fold answers/gaps back as tombstones.
  EXECUTE  — resilient-planner via drive.py (subprocess): AND/OR backtracking search over a
             durable plan-tree, driven to a terminal STATE.
  HARVEST  — harvest.py (pure parser): ✝ dead-set reasons + terminal-state facts become
             evidence for the next clarify round.

The evidence LEDGER is the only shared state; the original prompt is immutable (intent) and
each cycle re-renders prompt+ledger for a fresh planner slug `<slug>-c<N>`. Stop conditions:
SUCCESS | information-dry (a full cycle yields zero fresh facts — anti-flap) | max_cycles |
wallclock (cycle-boundary check; drive.py's own backstops cap within-cycle).

Written as a resumable-script flow: each phase is a memoized ctx.step, so a crash replays
completed phases from the journal. Strict-replay rules honored: branch conditions derive only
from the immutable input and memoized step results; clock reads are steps. The only ctx.ask
is the GUARD-HALT fork under --gate (default is assume-and-note: record the open fork as a
gap and let the next clarify round rank it).

Runs INSIDE the hermes container (drive.py is invoked --in-container; iterate.py needs the
ask skill's model_utils). Host-side use is tests-only (fakes). State:
  ${HERMES_HOME}/relentless/<slug>/   prompt-c<N>.md · ledger.jsonl · report.md · flow/ (engine)

Usage:
  relentless.py run    --slug S (--prompt TEXT | --prompt-file F) --answer-cwd DIR [options]
  relentless.py resume --slug S --answer TEXT [--key K]
Exit codes are the resumable-script engine's (0 completed — read result.outcome; 10 suspended
under --gate; 1/2/3 failures). See SKILL.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_AA = os.path.abspath(os.path.join(_HERE, "..", ".."))  # skills/autonomous-ai-agents
# The investigator's own default resolution assumes deployed-skill layout; in the repo tree
# (host or /opt/data/hermes-agent) the ranker is our sibling — pin it before importing iterate.
os.environ.setdefault("INFOGAIN_SCRIPTS_DIR", os.path.join(_AA, "information-gain", "scripts"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_AA, "investigator", "scripts"))

import harvest  # noqa: E402
import iterate as investigator  # noqa: E402

_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
PLANS_DIR = os.path.join(_HOME, "plans")
DRIVE_PY = os.environ.get("RESILIENT_DRIVE") or os.path.join(
    _HOME, "skills", "resilient-planner", "scripts", "drive.py")
_ENGINE_DIR = os.environ.get("RESUMABLE_ENGINE_DIR") or os.path.join(
    _HOME, "skills", "resumable-script", "scripts")

DEFAULTS = {
    "max_cycles": 5, "wallclock": 4 * 3600, "k": 6, "inv_rounds": 3, "floor": 0.12,
    "capability": "act", "answer_cwd": None, "gate": False,
    "drive": {"max_ticks": 12, "per_tick_timeout": 900, "wallclock": 3600},
}


# ── injectable phase helpers (tests monkeypatch these, like drive.py's DI wiring) ─────────────

def inv_cfg(inp):
    cfg = {"k": inp["k"], "max_rounds": inp["inv_rounds"], "floor": inp["floor"],
           "answer_cwd": inp["answer_cwd"], "responder_cwd": inp["answer_cwd"]}
    return investigator.apply_capability(cfg, inp["capability"])


def run_clarify(problem, seeds, cfg):
    """Investigator round trimmed to what the loop consumes (the engine journals results)."""
    out = investigator.iterate(problem, cfg, seed_evidence=seeds)
    return {"tombstones": out["tombstones"], "stop_reason": out["stop_reason"],
            "n_answered": out["n_answered"], "n_gaps": out["n_gaps"]}


def run_drive(slug, prompt_path, dcfg):
    """Drive one planner run to a terminal STATE. Any parseable --json result is a successful
    step (the status travels in the result); raise only on unparseable output/timeout."""
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


def run_harvest(slug, cycle):
    tree, journal = harvest.load_run(PLANS_DIR, slug)
    if tree is None:
        return {"records": [], "state": None, "fork": None}
    return {"records": harvest.harvest(tree, journal, cycle),
            "state": harvest.parse_state(tree), "fork": harvest.extract_fork(tree)}


def render(prompt, ledger):
    """Immutable intent + the evidence ledger, section per kind. Empty sections omitted."""
    sections = (("fact", "## Established facts (do not re-derive)"),
                ("gap", "## Known gaps (proceed on the stated assumption)"),
                ("dead-end", "## Dead ends — do NOT re-attempt these methods"))
    parts = [prompt.rstrip()]
    for kind, header in sections:
        texts = [r["text"] for r in ledger if r["kind"] == kind]
        if texts:
            parts.append(header + "\n" + "\n".join(f"- {t}" for t in texts))
    return "\n\n".join(parts) + "\n"


def planner_envelope(body, plan_slug):
    """Wrap the rendered intent+ledger in the canonical resilient-planner invocation (the
    ladder-verified framing from the planner's test helpers): load the skill, REAL mode,
    and pin the plan-tree/journal artifact paths to this cycle's slug — drive.py passes the
    prompt file to `hermes -z` verbatim and detects the terminal STATE from that plan-tree."""
    plans = PLANS_DIR
    return (
        "Use the resilient-planner skill: skill_view to load it, then follow it "
        "INCLUDING the lean Decision Records (predict -> act -> reconcile) discipline. "
        "This is a REAL run (Simulation Mode OFF).\n"
        f"INTENT: {body}\n"
        f"Write the plan-tree to {plans}/{plan_slug}/plan-tree.md as a COMPACT MARKER MAP "
        "(STATE header; INTENT/constraints; a NODES list with markers ○/▶/✝/✓ + a "
        "one-line receipt per node; a FRONTIER line) — no Branch-log/Decision-log. "
        "Append ONE lean single-line JSON record per cycle (fields: node, q, chosen, "
        f"expected, verdict, evidence, next) to {plans}/{plan_slug}/journal.jsonl "
        "(valid JSONL — one compact object per line, newline-separated)."
    )


def _atomic_write(path, content):
    tmp = path + ".tmp"
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


def write_report(slug_dir, outcome, ledger, cycles, detail):
    os.makedirs(slug_dir, exist_ok=True)
    lines = [f"# relentless-solve report", "",
             f"OUTCOME: {outcome}   CYCLES: {cycles}", f"DETAIL: {detail}", ""]
    for kind, title in (("fact", "Established facts"), ("gap", "Known gaps / assumptions"),
                        ("dead-end", "Dead ends")):
        recs = [r for r in ledger if r["kind"] == kind]
        if recs:
            lines.append(f"## {title} ({len(recs)})")
            lines += [f"- [c{r['cycle']}·{r['source']}] {r['text']}" for r in recs]
            lines.append("")
    path = os.path.join(slug_dir, "report.md")
    _atomic_write(path, "\n".join(lines))
    return path


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


def fold_clarify(tombstones, cycle, ledger, seen):
    records = [{"cycle": cycle, "source": "clarify",
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
    inp["drive"] = {**DEFAULTS["drive"], **(inp.get("drive") or {})}
    ledger, seen = [], set()  # rebuilt deterministically on every (re)play
    slug_dir = os.path.join(_HOME, "relentless", inp["slug"])
    t0 = ctx.step("t0", lambda: time.time())
    outcome, detail = "max-cycles", f"max_cycles={inp['max_cycles']} reached without success"
    cycles_run = 0

    for cycle in range(inp["max_cycles"]):
        c = f"c{cycle}"
        now = ctx.step(f"{c}/clock", lambda: time.time())
        if now - t0 > inp["wallclock"]:
            outcome, detail = "wallclock", f"wall-clock budget hit before cycle {cycle}"
            break

        # A — CLARIFY: next-best questions over everything known so far
        inv = ctx.step(f"{c}/clarify",
                       lambda: run_clarify(inp["prompt"], [r["text"] for r in ledger],
                                           inv_cfg(inp)))
        fresh_clar = fold_clarify(inv["tombstones"], cycle, ledger, seen)

        # B — EXECUTE: fresh planner slug against the re-rendered prompt
        pp = ctx.step(f"{c}/render",
                      lambda: persist(slug_dir, cycle,
                                      planner_envelope(render(inp["prompt"], ledger),
                                                       f"{inp['slug']}-c{cycle}"),
                                      ledger))
        st = ctx.step(f"{c}/execute",
                      lambda: run_drive(f"{inp['slug']}-c{cycle}", pp["prompt_path"],
                                        inp["drive"]))
        cycles_run = cycle + 1
        if st["status"] == "SUCCESS":
            outcome, detail = "success", st.get("detail", "planner reached SUCCESS")
            break

        # C — HARVEST: failure conditions become evidence
        harv = ctx.step(f"{c}/harvest", lambda: run_harvest(f"{inp['slug']}-c{cycle}", cycle))
        fresh_harv = fold_records(ledger, seen, harv["records"])

        # E — GUARD-HALT fork: ask under --gate, else assume-and-note (next clarify ranks it)
        if st["status"] == "GUARD_HALT" and harv.get("fork"):
            if inp["gate"]:
                ans = ctx.ask(f"{c}/fork", {"prompt": harv["fork"], "type": "string"},
                              schema={"type": "string"})
                fresh_harv += fold_one(ledger, seen, cycle, "clarify", "fact",
                                       f"{harv['fork']} -> {ans}")
            else:
                fresh_harv += fold_one(
                    ledger, seen, cycle, "assumption", "gap",
                    f"OPEN FORK (guard halt, unresolved): {harv['fork']} -> ASSUMED: keep "
                    f"exploring the remaining frontier methods next cycle; the stop was a "
                    f"budget guard, not evidence.")

        # D — STOP HONESTLY: relentless ≠ flailing; zero new information anywhere → dry
        if fresh_clar == 0 and fresh_harv == 0 and "converged" in (inv.get("stop_reason") or ""):
            outcome, detail = "information-dry", (f"cycle {cycle} produced zero fresh facts "
                                                  f"(clarify converged, harvest all seen)")
            break

    rep = ctx.step("report", lambda: write_report(slug_dir, outcome, ledger, cycles_run, detail))
    return {"outcome": outcome, "cycles": cycles_run, "detail": detail,
            "n_facts": len(ledger), "report": rep}


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


def main(argv=None):
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
    r.add_argument("--drive-max-ticks", type=int, default=DEFAULTS["drive"]["max_ticks"])
    r.add_argument("--drive-wallclock", type=int, default=DEFAULTS["drive"]["wallclock"])
    r.add_argument("--per-tick-timeout", type=int, default=DEFAULTS["drive"]["per_tick_timeout"])
    r.add_argument("--state-dir", help="engine state dir (default "
                                       "$HERMES_HOME/relentless/<slug>/flow)")
    r.add_argument("--accept-flow-change", action="store_true")

    z = sub.add_parser("resume", help="answer a suspended --gate fork and continue")
    z.add_argument("--slug", required=True)
    z.add_argument("--answer", required=True)
    z.add_argument("--key")
    z.add_argument("--state-dir")
    z.add_argument("--accept-flow-change", action="store_true")

    args = p.parse_args(argv)
    flow, run_cli = _load_engine()
    FLOW = flow(id="relentless-solve", version=1)(relentless_flow)
    # Engine default state dir is per-flow-id (shared across slugs) — always pin per-slug.
    state_dir = args.state_dir or os.path.join(_HOME, "relentless", args.slug, "flow")

    if args.cmd == "run":
        if args.prompt_file:
            prompt = sys.stdin.read() if args.prompt_file == "-" else open(
                args.prompt_file, encoding="utf-8").read()
        else:
            prompt = args.prompt
        if not (prompt or "").strip():
            print("need --prompt or --prompt-file", file=sys.stderr)
            return 2
        inp = {"prompt": prompt, "slug": args.slug, "max_cycles": args.max_cycles,
               "wallclock": args.wallclock, "k": args.k, "inv_rounds": args.inv_rounds,
               "floor": args.floor, "capability": args.capability,
               "answer_cwd": args.answer_cwd, "gate": args.gate,
               "drive": {"max_ticks": args.drive_max_ticks,
                         "per_tick_timeout": args.per_tick_timeout,
                         "wallclock": args.drive_wallclock}}
        eng_argv = ["run", "--input", json.dumps(inp), "--state-dir", state_dir]
        if not args.gate:
            eng_argv.append("--auto")
    else:
        eng_argv = ["resume", "--answer", args.answer, "--state-dir", state_dir]
        if args.key:
            eng_argv += ["--key", args.key]
    if args.accept_flow_change:
        eng_argv.append("--accept-flow-change")
    return run_cli(FLOW, argv=eng_argv)


if __name__ == "__main__":
    sys.exit(main())

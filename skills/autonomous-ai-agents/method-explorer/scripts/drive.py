#!/usr/bin/env python3
"""Driver-loop wrapper for the method-explorer skill.

A single `hermes -z` oneshot turn ends at `STATE: active` whenever it can't finish a
task within the agent-turn cap (`agent.max_turns`), or on a no-op/timeout. The
method-explorer skill is *resumable* (its durable plan-tree carries the open FRONTIER
and the ✝ dead-set), but nothing re-drives one task to completion. This driver does:
it re-invokes the skill (same prompt) so the skill RESUMES from the on-disk plan-tree,
tick after tick, until a terminal `STATE`.

Resume is ARTIFACT-BASED — there is no `--resume` flag. Re-running the same prompt makes
the skill detect `${HERMES_HOME}/plans/<slug>/plan-tree.md` and continue from its FRONTIER.
Terminal detection is via the plan-tree H1 `STATE:` header (oneshot stdout is final-text
only), NOT via stdout.

Design invariant: **the driver never writes or deletes plan-tree.md.** Only the skill
mutates it. The driver only READS it (terminal/progress signal) and ARCHIVES journal.jsonl
between ticks (so the audit survives whether the skill appends or overwrites on resume).

The loop is dependency-injected (`invoke` / `read_plan_tree` / `archive_journal` / `sleep`
/ `now`) so it runs three ways with no adapter:
  - host/tests : helpers.run_planner / helpers.read_file / _dex `mv`   (docker exec)
  - in-container: direct `hermes -z` subprocess / local file IO        (`--in-container`)
  - unit test  : scripted returns                                       (no container)

CLI:  python3 drive.py --slug SLUG --prompt-file PATH|-  [--in-container] [...]
"""
import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time

# Canonical plan-tree STATE values (SKILL.md "Plan-Tree Artifact").
TERMINAL_STATES = ("SUCCESS", "EXHAUSTION-STOP", "GUARD-HALT")
DEFAULT_PLANS_DIR = os.environ.get("HERMES_HOME", "/opt/data").rstrip("/") + "/plans"

_ONESHOT_MOD = None


def _oneshot():
    """Lazy: the bare `hermes -z` dispatch primitive shared with relentless-solve
    (resumable-script/scripts/oneshot.py — same "artifact on disk beats stdout, even on
    timeout" lesson this driver already independently follows). Same resolution
    convention relentless.py uses for the engine itself: env override or deployed."""
    global _ONESHOT_MOD
    if _ONESHOT_MOD is None:
        engine_dir = os.environ.get("RESUMABLE_ENGINE_DIR") or os.path.join(
            os.environ.get("HERMES_HOME", "/opt/data").rstrip("/"),
            "skills", "resumable-script", "scripts")
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        import oneshot  # noqa: E402
        _ONESHOT_MOD = oneshot
    return _ONESHOT_MOD

_STATE_RE = re.compile(r"STATE:\s*([A-Za-z][A-Za-z-]*)", re.IGNORECASE)
# node line e.g.  "- S1  alfa (primary)   ✝ reason"  -> capture id + the first marker glyph
_NODE_RE = re.compile(r"^\s*-\s+(\S+).*?([✝✓▶○])", re.UNICODE)
_FRONTIER_RE = re.compile(r"^\s*FRONTIER:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_state(tree_text):
    """Return the canonical STATE token from a plan-tree, or None.

    None means 'no recognizable STATE header' — the caller distinguishes an ABSENT tree
    (read_plan_tree returned None) from a PRESENT-but-unparseable tree (treated as active).
    Robust to case, extra spaces, CRLF, and trailing text after the value.
    """
    if not tree_text:
        return None
    m = _STATE_RE.search(tree_text)
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
    return None  # present but unrecognized -> caller treats as active


def fingerprint(tree_text):
    """A STRUCTURED progress fingerprint: (STATE, sorted ✝ dead ids, sorted ✓ done ids,
    normalized FRONTIER). Keys on node IDENTITY, not receipt prose, so cosmetic rewording
    of a receipt does NOT look like progress. Falls back to a text hash if no nodes parse.
    Returns a hashable, comparable value (or None for an absent tree)."""
    if tree_text is None:
        return None
    state = parse_state(tree_text)
    dead, done = [], []
    for line in tree_text.splitlines():
        m = _NODE_RE.match(line)
        if not m:
            continue
        nid, marker = m.group(1), m.group(2)
        if marker == "✝":
            dead.append(nid)
        elif marker == "✓":
            done.append(nid)
    fm = _FRONTIER_RE.search(tree_text)
    frontier = re.sub(r"\s+", " ", fm.group(1).strip().lower()) if fm else ""
    if not dead and not done and state is None:
        # Unparseable / torn write — hash the stripped text so it's stable but opaque.
        return ("hash", hashlib.sha256(tree_text.strip().encode("utf-8", "replace")).hexdigest())
    return (state, tuple(sorted(dead)), tuple(sorted(done)), frontier)


def dead_nodes(tree_text):
    """Descriptive labels of ✝-dead nodes (id + method text up to the marker). Naming the
    dead methods explicitly in the resume nudge is far more reliable than trusting the skill
    to re-derive the dead-set when the base prompt still lists a dead method as 'preferred'."""
    out = []
    for line in (tree_text or "").splitlines():
        if "✝" in line and line.lstrip().startswith("-"):
            label = line.split("✝", 1)[0].lstrip("- ").strip()
            if label:
                out.append(label)
    return out


def resume_suffix(plan_path, dead=None, bumped=0):
    """Pure suffix appended to the base prompt when a non-terminal tree already exists
    (mirrors test_10_resume.py's framing). NEVER edits the base prompt's slug/path."""
    s = (
        "\n\nA PRIOR run on this task was INTERRUPTED — a partial plan-tree already exists "
        f"at {plan_path}. Follow the skill's 'Resuming an interrupted run' guidance: READ "
        "its STATE header and RESUME from the FRONTIER. Do NOT restart from scratch."
    )
    if dead:
        s += ("\nThese nodes/methods are ALREADY PROVEN DEAD (✝) — you MUST NOT choose, "
              "retry, or re-expand any of them, EVEN IF an earlier 'preference order' lists "
              "one of them first: " + "; ".join(dead) + ".")
    else:
        s += " Do NOT re-expand any ✝ method."
    if bumped:
        s += (f" You MAY continue with the exploration budget raised by +{bumped} branches "
              "(up to the operator ceiling) — this is a sanctioned guard-halt bump.")
    return s


@dataclasses.dataclass
class DriveResult:
    status: str                # SUCCESS|EXHAUSTION|GUARD_HALT|STUCK|NOOP_EXHAUSTED|MAX_TICKS|WALLCLOCK
    productive_ticks: int
    invokes: int
    final_state: str = None    # the plan-tree STATE at stop (may be 'active' for non-terminal stops)
    guard_bumps: int = 0
    detail: str = ""

    @property
    def terminal(self):
        """True if the SKILL reached a terminal state (vs the driver hitting a backstop)."""
        return self.status in ("SUCCESS", "EXHAUSTION", "GUARD_HALT")

    def as_dict(self):
        return dataclasses.asdict(self)


def drive(prompt, slug, *, invoke, read_plan_tree, archive_journal,
          plan_path=None, sleep=time.sleep, now=time.monotonic, log=print,
          max_ticks=12, per_tick_timeout=900,
          max_consecutive_noops=3, noop_gap=8,
          stuck_n=2, wallclock_budget=3 * 3600,
          bump_guard=False, max_guard_bumps=2, guard_hard_ceiling=None,
          resume_nudge=True):
    """Re-invoke the skill until the plan-tree reaches a terminal STATE (or a backstop).

    Callables:
      invoke(prompt, timeout)  -> object with .returncode and .stdout (CompletedProcess-like)
      read_plan_tree(slug)     -> str | None  (None = the plan-tree file does not exist)
      archive_journal(slug, n) -> None        (mv journal.jsonl journal.tick{n}.jsonl; never touch plan-tree)
    """
    if plan_path is None:
        plan_path = f"{DEFAULT_PLANS_DIR}/{slug}/plan-tree.md"

    t0 = now()
    productive = 0          # ticks that actually advanced the tree (the real budget)
    consec_noops = 0        # SEPARATE budget — infra flakes never burn productive ticks
    guard_bumps = 0
    stuck_run = 0
    seen_fps = {}           # fingerprint -> count, to catch oscillation livelock
    invoke_seq = 0

    while True:
        if now() - t0 > wallclock_budget:
            return DriveResult("WALLCLOCK", productive, invoke_seq, "active",
                               guard_bumps, "wall-clock budget exceeded; STATE left active (resumable)")

        pre_tree = read_plan_tree(slug)
        pre_state = parse_state(pre_tree)
        pre_fp = fingerprint(pre_tree)

        # Already terminal on disk (seeded SUCCESS, or a prior tick finished) — don't redo.
        if pre_state == "SUCCESS":
            return DriveResult("SUCCESS", productive, invoke_seq, "SUCCESS", guard_bumps,
                               "plan-tree already terminal: SUCCESS")
        if pre_state == "EXHAUSTION-STOP":
            return DriveResult("EXHAUSTION", productive, invoke_seq, "EXHAUSTION-STOP",
                               guard_bumps, "plan-tree already terminal: EXHAUSTION-STOP")
        if pre_state == "GUARD-HALT" and not (bump_guard and guard_bumps < max_guard_bumps):
            return DriveResult("GUARD_HALT", productive, invoke_seq, "GUARD-HALT",
                               guard_bumps, "plan-tree already terminal: GUARD-HALT")

        # Resume framing whenever a non-terminal tree already exists (incl. a seeded one at tick 0).
        resuming = pre_tree is not None and pre_state in ("active", None)
        if resume_nudge and resuming:
            this_prompt = prompt + resume_suffix(plan_path, dead=dead_nodes(pre_tree),
                                                 bumped=guard_bumps)
        else:
            this_prompt = prompt

        # Archive the journal before invoking. The skill APPENDS on resume (verified), so
        # the union of journal.tick*.jsonl + journal.jsonl is the full audit and this also
        # yields a clean per-tick delta. Belt-and-suspenders; never touches the plan-tree.
        if pre_tree is not None:
            archive_journal(slug, invoke_seq)
        invoke_seq += 1

        res = invoke(this_prompt, per_tick_timeout)
        produced = bool((getattr(res, "stdout", "") or "").strip())

        post_tree = read_plan_tree(slug)
        post_state = parse_state(post_tree)
        fp = fingerprint(post_tree)
        changed = post_tree is not None and fp != pre_fp

        # (1) NO-OP flake: nothing produced AND the tree did not move. Empty stdout ALONE is
        #     not a no-op (a 900s SIGTERM often yields empty stdout with an advanced tree).
        if post_tree is None or (not changed and not produced and post_state in ("active", None)):
            consec_noops += 1
            if consec_noops > max_consecutive_noops:
                return DriveResult("NOOP_EXHAUSTED", productive, invoke_seq, post_state,
                                   guard_bumps, f"{consec_noops} consecutive no-ops (infra)")
            log(f"[drive] tick {invoke_seq}: no-op flake "
                f"({consec_noops}/{max_consecutive_noops}); retrying (productive budget untouched)")
            sleep(noop_gap)
            continue
        consec_noops = 0

        # (2) Terminal STATE reached this tick.
        if post_state == "SUCCESS":
            return DriveResult("SUCCESS", productive + 1, invoke_seq, "SUCCESS", guard_bumps,
                               "reached SUCCESS")
        if post_state == "EXHAUSTION-STOP":
            return DriveResult("EXHAUSTION", productive + 1, invoke_seq, "EXHAUSTION-STOP",
                               guard_bumps, "reached EXHAUSTION-STOP (frontier empty)")
        if post_state == "GUARD-HALT":
            at_ceiling = guard_hard_ceiling is not None and guard_bumps >= guard_hard_ceiling
            if not bump_guard or guard_bumps >= max_guard_bumps or not changed or at_ceiling:
                return DriveResult("GUARD_HALT", productive + 1, invoke_seq, "GUARD-HALT",
                                   guard_bumps,
                                   "GUARD-HALT (skill backstop); not bumping"
                                   + ("" if changed else " — tree unchanged after a bump (structural wall)"))
            guard_bumps += 1
            log(f"[drive] tick {invoke_seq}: GUARD-HALT with progress; bump {guard_bumps}/{max_guard_bumps}")
            productive += 1
            if productive >= max_ticks:
                return DriveResult("MAX_TICKS", productive, invoke_seq, "GUARD-HALT",
                                   guard_bumps, "max_ticks reached")
            continue

        # (3) LIVELOCK: the tree didn't move this tick, or a fingerprint is recurring.
        seen_fps[fp] = seen_fps.get(fp, 0) + 1
        if not changed or seen_fps[fp] > stuck_n:
            stuck_run += 1
            if stuck_run >= stuck_n:
                return DriveResult("STUCK", productive + 1, invoke_seq, post_state, guard_bumps,
                                   "no progress across consecutive ticks (livelock); STATE active (resumable)")
        else:
            stuck_run = 0

        # (4) Progress — consume a productive tick and loop to resume.
        productive += 1
        log(f"[drive] tick {invoke_seq}: STATE={post_state}, progressed; resuming")
        if productive >= max_ticks:
            return DriveResult("MAX_TICKS", productive, invoke_seq, post_state, guard_bumps,
                               "max_ticks reached; STATE active (resumable)")


# --------------------------------------------------------------------------- CLI

# status -> process exit code (0 = task succeeded; distinct nonzero for everything else).
_EXIT = {"SUCCESS": 0, "EXHAUSTION": 2, "GUARD_HALT": 3, "STUCK": 4,
         "NOOP_EXHAUSTED": 5, "MAX_TICKS": 6, "WALLCLOCK": 7}


def _container_wiring(plans_dir):
    """In-container callables: direct `hermes -z` subprocess (via the shared oneshot
    module — the same primitive relentless-solve uses) + local file IO."""
    hermes = os.environ.get("HERMES_BIN", "/opt/hermes/bin/hermes")

    def invoke(prompt, timeout):
        # A timeout may still have advanced the plan-tree on disk -> rc 124, partial stdout.
        return _oneshot().run_direct(prompt, timeout, hermes_bin=hermes)

    def read_plan_tree(slug):
        p = f"{plans_dir}/{slug}/plan-tree.md"
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def archive_journal(slug, n):
        j = f"{plans_dir}/{slug}/journal.jsonl"
        if os.path.exists(j):
            os.replace(j, f"{plans_dir}/{slug}/journal.tick{n}.jsonl")

    return invoke, read_plan_tree, archive_journal


def _host_wiring():
    """Host callables: reuse the test harness (docker exec) — no re-implementation."""
    import helpers  # noqa: imported lazily so in-container mode needs no docker

    def invoke(prompt, timeout):
        return helpers.run_planner(prompt, timeout=timeout)

    def read_plan_tree(slug):
        return helpers.read_file(f"{helpers.PLANS}/{slug}/plan-tree.md")

    def archive_journal(slug, n):
        helpers._dex(f"mv {helpers.PLANS}/{slug}/journal.jsonl "
                     f"{helpers.PLANS}/{slug}/journal.tick{n}.jsonl 2>/dev/null || true")

    return invoke, read_plan_tree, archive_journal, helpers.PLANS


def main(argv=None):
    ap = argparse.ArgumentParser(description="Drive method-explorer to a terminal STATE.")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--prompt-file", required=True, help="path to the base prompt, or '-' for stdin")
    ap.add_argument("--in-container", action="store_true",
                    help="run hermes -z directly + local file IO (default: host via docker/helpers)")
    ap.add_argument("--plans-dir", default=DEFAULT_PLANS_DIR)
    ap.add_argument("--max-ticks", type=int, default=12)
    ap.add_argument("--per-tick-timeout", type=int, default=900)
    ap.add_argument("--max-noops", type=int, default=3)
    ap.add_argument("--stuck-n", type=int, default=2)
    ap.add_argument("--wallclock", type=int, default=3 * 3600)
    ap.add_argument("--bump-guard", action="store_true")
    ap.add_argument("--max-guard-bumps", type=int, default=2)
    ap.add_argument("--guard-ceiling", type=int, default=None)
    ap.add_argument("--no-resume-nudge", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit the result as one JSON line")
    args = ap.parse_args(argv)

    prompt = sys.stdin.read() if args.prompt_file == "-" else open(args.prompt_file, encoding="utf-8").read()

    if args.in_container:
        invoke, read_plan_tree, archive_journal = _container_wiring(args.plans_dir)
        plans_dir = args.plans_dir
    else:
        invoke, read_plan_tree, archive_journal, plans_dir = _host_wiring()

    result = drive(
        prompt, args.slug,
        invoke=invoke, read_plan_tree=read_plan_tree, archive_journal=archive_journal,
        plan_path=f"{plans_dir}/{args.slug}/plan-tree.md",
        max_ticks=args.max_ticks, per_tick_timeout=args.per_tick_timeout,
        max_consecutive_noops=args.max_noops, stuck_n=args.stuck_n,
        wallclock_budget=args.wallclock, bump_guard=args.bump_guard,
        max_guard_bumps=args.max_guard_bumps, guard_hard_ceiling=args.guard_ceiling,
        resume_nudge=not args.no_resume_nudge,
    )
    if args.json:
        print(json.dumps(result.as_dict()))
    else:
        print(f"[drive] DONE status={result.status} productive_ticks={result.productive_ticks} "
              f"invokes={result.invokes} final_state={result.final_state} :: {result.detail}")
    return _EXIT.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())

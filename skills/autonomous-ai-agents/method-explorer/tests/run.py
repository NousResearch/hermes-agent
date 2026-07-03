#!/usr/bin/env python3
"""Minimal, dependency-free test runner (so the suite runs with plain python3).

Test files are written as normal pytest modules, but pytest need not be installed:
this runner shims a tiny `pytest` so they import, discovers `test_*` functions, runs
them, and reports pass/fail (and pass-rate with --reps). If real pytest IS installed,
`pytest -m agent` works too.

Usage:
  python3 run.py                          # run all test_*.py
  python3 run.py test_01_anti_fabrication # run one module
  python3 run.py -k anti                  # modules matching a substring
  python3 run.py --reps 3 -k backtrack    # repeat 3x, report pass-rate
  python3 run.py --tiers                  # print the escalation ladder, run nothing
  python3 run.py --gauntlet               # run tiers low->high, STOP at first red tier
  python3 run.py --gauntlet --from 2      # start the gauntlet at tier 2
  python3 run.py --gauntlet --to 1        # only run tiers 0..1
  python3 run.py --survey                 # run ALL tiers, stratified scorecard, no stop

Gauntlet semantics (fix-loop mode):
  - preflight first (container/skill/static-scenarios) — a broken env exits 2 (INFRA)
    before any tokens are spent; "fix the skill here" advice would be wrong there.
  - a failed test FUNCTION first renders its diagnostics (trace + hermes stdout tail —
    BEFORE any retry, because a retry self-resets the artifacts), then, at tiers with
    retry policy (T2-T4) and reps==1, retries ONCE: pass-on-retry = FLAKY-PASS (the
    ladder continues; the flake is logged + trended), fail-twice = red.
  - a persistent empty-journal no-op is an INFRA signal (backend down, not skill logic):
    the whole gauntlet aborts with exit 2 and a resume hint.
  - every module/function outcome is appended to gauntlet-log.jsonl so retry-masked
    pass-rate drops stay detectable across runs ("flaky-passed 4 of the last 5").
  Exit codes: 0 green · 1 red (skill logic) · 2 infra (nothing in the skill to fix).
"""
import glob
import importlib
import json
import os
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GAUNTLET_LOG = os.path.join(HERE, "gauntlet-log.jsonl")

# Static, pre-existing scenario files the tests reference by fixed path but never
# create — if one is missing, live tests fail at a random tier with misleading advice.
STATIC_SCENARIOS = [
    "exhaustion-demo.json", "dispensable-subgoal.json", "tiny-budget.json",
    "k5-siblings.json", "lying-tool.json", "assumption-flip.json",
]


class _Skip(Exception):
    pass


# Shim a minimal `pytest` so test modules import without the real dependency.
if "pytest" not in sys.modules:
    try:
        import pytest  # noqa: F401
    except ImportError:
        m = types.ModuleType("pytest")

        class _Mark:
            def __getattr__(self, _):  # @pytest.mark.agent -> no-op decorator
                return lambda f=None: (f if callable(f) else (lambda g: g))

        m.mark = _Mark()
        m.fixture = lambda *a, **k: (lambda f: f)

        def _skip(msg=""):
            raise _Skip(msg)

        m.skip = _skip
        sys.modules["pytest"] = m


def discover(pattern=None):
    names = []
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if pattern and pattern not in name:
            continue
        names.append(name)
    return names


def _show_module_trace(mod, modname):
    """`--show`: render the module's run trace (input→thinking→output) from its artifacts.

    Uses the module-level `SLUG` convention (most agent tests expose one); modules with
    several/dynamic slugs opt in with a `TRACE_SLUGS` list. Container-free unit tests have
    no artifacts and are skipped.
    """
    slugs = getattr(mod, "TRACE_SLUGS", None) or ([mod.SLUG] if hasattr(mod, "SLUG") else [])
    if not slugs:
        return
    import helpers
    import trace as _trace
    for slug in slugs:
        rows = helpers.load_journal(slug)
        tree = helpers.read_file(f"{helpers.PLANS}/{slug}/plan-tree.md")
        print(f"\n----- trace: {modname} [{slug}] " + "-" * 40)
        _trace.show_trace(rows, tree)


def run_module(modname, reps=1, show=False):
    mod = importlib.import_module(modname)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    all_ok = True
    for fn in fns:
        passes, last_err = 0, ""
        for _ in range(reps):
            try:
                fn()
                passes += 1
            except _Skip as e:
                print(f"  SKIP {modname}.{fn.__name__}: {e}")
                break
            except AssertionError as e:
                last_err = str(e) or "assertion failed"
            except Exception:
                last_err = traceback.format_exc().splitlines()[-1]
        ok = passes == reps
        tag = "PASS" if ok else f"FAIL ({passes}/{reps})"
        print(f"  [{tag}] {modname}.{fn.__name__}" + (f"  -> {last_err}" if last_err else ""))
        all_ok = all_ok and ok
    if show:
        _show_module_trace(mod, modname)
    return all_ok


# --------------------------------------------------------------------------- gauntlet

def _call_fn(fn):
    """Run one test function once. Returns (status, err), status in pass/fail/skip."""
    try:
        fn()
        return "pass", ""
    except _Skip as e:
        return "skip", str(e)
    except AssertionError as e:
        return "fail", str(e) or "assertion failed"
    except Exception:
        return "fail", traceback.format_exc().splitlines()[-1]


def _render_failure(mod, modname, err):
    """Failure diagnostics — MUST run before any retry: modules self-reset, so a retry
    destroys the failed attempt's journal/plan-tree. Renders the trace + stdout tail."""
    print(f"    diagnostics ({modname}, failed attempt's artifacts):  -> {err}")
    try:
        _show_module_trace(mod, modname)
    except Exception as e:  # diagnostics must never mask the real failure
        print(f"    (trace unavailable: {e})")
    try:
        import helpers
        tail = "\n".join((helpers.LAST_STDOUT or "").strip().splitlines()[-20:])
        if tail:
            print(f"    last hermes stdout tail:\n{tail}")
    except Exception:
        pass


def _infra_down():
    """True iff the last live run exhausted its no-op retries (backend, not logic)."""
    try:
        import helpers
        return bool(helpers.LAST_NOOP)
    except Exception:
        return False


def _log_outcome(entry, path=GAUNTLET_LOG):
    """Append one JSONL trend row. Trend data — NOT resume state (resume is --from N)."""
    entry = dict(entry)
    entry["ts"] = round(time.time(), 1)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"    (gauntlet-log unwritable: {e})")


def preflight():
    """Free environment check. Returns a list of problems (empty = OK)."""
    import helpers
    p = helpers._dex("echo up")
    if p.returncode != 0:
        return [f"container '{helpers.CONTAINER}' not reachable: "
                f"{(p.stderr or '').strip() or 'docker exec failed'}"]
    problems = []
    if helpers._dex("test -f /opt/data/skills/method-explorer/SKILL.md").returncode != 0:
        problems.append("skill not deployed: /opt/data/skills/method-explorer/SKILL.md missing")
    for name in STATIC_SCENARIOS:
        if helpers._dex(f"test -f {helpers.SCEN_DIR}/{name}").returncode != 0:
            problems.append(f"static scenario missing: {helpers.SCEN_DIR}/{name}")
    return problems


def _run_reps(fn, reps, run_fn):
    """Run fn `reps` times (conjunctive, matching run_module). -> (status, err)."""
    passes, last_err = 0, ""
    for _ in range(reps):
        status, err = run_fn(fn)
        if status == "skip":
            return "skip", err
        if status == "pass":
            passes += 1
        else:
            last_err = err
    return ("pass", "") if passes == reps else ("fail", last_err)


def run_gauntlet(reps=1, show=False, lo=0, hi=None, survey=False, *,
                 tiers_list=None, get_module=None, run_fn=None, diagnose=None,
                 infra_down=None, log=None, preflight_fn=None):
    """Walk the tiers low->high; STOP at the first tier with any post-retry failure.

    The whole point: on a break, you fix the earliest/simplest cause before spending
    tokens on the harder tiers whose premises it invalidates. `survey=True` never stops
    and prints a stratified scorecard instead (the once-per-SKILL.md-edit full picture).
    Keyword-only params are dependency injection for the offline meta-test.
    Returns process exit code: 0 green · 1 red · 2 infra.
    """
    import tiers as _tiers
    tiers_list = tiers_list if tiers_list is not None else _tiers.TIERS
    get_module = get_module or importlib.import_module
    run_fn = run_fn or _call_fn
    diagnose = diagnose or _render_failure
    infra_down = infra_down or _infra_down
    log = log or _log_outcome
    preflight_fn = preflight_fn or preflight
    hi = tiers_list[-1][0] if hi is None else hi
    mode = "survey" if survey else "gauntlet"

    problems = preflight_fn()
    if problems:
        print("PREFLIGHT FAILED — INFRA, nothing in the skill to fix:")
        for p in problems:
            print(f"  - {p}")
        print(f"Fix the environment, then re-run: python3 run.py --{mode} --from {lo}")
        log({"mode": mode, "tier": None, "module": "(preflight)", "fn": None,
             "outcome": "infra", "flaky": False, "err": "; ".join(problems)})
        return 2

    scorecard, ran, any_red = [], 0, False
    for idx, name, intent, mods in tiers_list:
        if not (lo <= idx <= hi):
            continue
        ran += 1
        retry_ok = idx in _tiers.RETRY_TIERS and reps == 1
        retry_lbl = "retry-once" if retry_ok else "no retry"
        print(f"\n===== Tier {idx} · {name}  [{_tiers.cost_label(idx)} · {retry_lbl}] =====")
        print(f"      {intent}\n")
        tier_ok = True
        for modname in mods:
            mod = get_module(modname)
            print(modname)
            fns = [getattr(mod, n) for n in dir(mod)
                   if n.startswith("test_") and callable(getattr(mod, n))]
            for fn in fns:
                status, err = _run_reps(fn, reps, run_fn)
                flaky = False
                if status == "fail":
                    diagnose(mod, modname, err)      # BEFORE retry — retry wipes artifacts
                    if infra_down():
                        print(f"\n>>> INFRA ABORT — persistent no-op (backend down, not "
                              f"skill logic). Fix the backend, then resume with: "
                              f"python3 run.py --{mode} --from {idx}")
                        log({"mode": mode, "tier": idx, "module": modname,
                             "fn": fn.__name__, "outcome": "infra", "flaky": False,
                             "err": err[:200]})
                        return 2
                    if retry_ok:
                        status, err2 = run_fn(fn)
                        if status == "pass":
                            flaky = True
                        else:
                            err = err2 or err
                tag = {"pass": "FLAKY-PASS" if flaky else "PASS",
                       "fail": "FAIL", "skip": "SKIP"}[status]
                print(f"  [{tag}] {modname}.{fn.__name__}"
                      + (f"  -> {err}" if err and status != "pass" else ""))
                log({"mode": mode, "tier": idx, "module": modname, "fn": fn.__name__,
                     "outcome": "flaky-pass" if flaky else status, "flaky": flaky,
                     "err": err[:200] if status == "fail" else ""})
                scorecard.append((idx, name, modname, fn.__name__,
                                  "flaky-pass" if flaky else status))
                if status == "fail":
                    tier_ok = False
            if show:
                _show_module_trace(mod, modname)
        if not tier_ok:
            any_red = True
            if not survey:
                print(f"\n>>> TIER {idx} ({name}) FAILED — fix here before escalating. "
                      f"Higher tiers NOT run (their premises depend on this one).")
                print(f"    Resume after the fix with: python3 run.py --gauntlet --from {idx}")
                return 1
            print(f"\n--- Tier {idx} ({name}) RED (survey continues) ---")
        else:
            print(f"\n--- Tier {idx} ({name}) GREEN ---")
    if not ran:
        print("no tiers in the selected range")
        return 1
    if survey:
        print("\n===== SURVEY SCORECARD =====")
        for idx, name, *_ in tiers_list:
            rows = [r for r in scorecard if r[0] == idx]
            if not rows:
                continue
            n_fail = sum(1 for r in rows if r[4] == "fail")
            n_flaky = sum(1 for r in rows if r[4] == "flaky-pass")
            state = "RED" if n_fail else ("FLAKY" if n_flaky else "GREEN")
            print(f"  Tier {idx} · {name}: {state}"
                  f"  ({len(rows)} fn, {n_fail} fail, {n_flaky} flaky)")
            for _, _, m, f, o in rows:
                if o != "pass":
                    print(f"      {o.upper():10s} {m}.{f}")
        return 1 if any_red else 0
    print("\nGAUNTLET GREEN — every selected tier passed.")
    return 0


def main(argv):
    reps, pattern, show, args = 1, None, False, list(argv)
    gauntlet, survey, lo, hi = False, False, 0, None
    if "--show" in args:
        args.remove("--show"); show = True
    if "--tiers" in args:
        import tiers
        print(tiers.render_ladder())
        return 0
    if "--gauntlet" in args:
        args.remove("--gauntlet"); gauntlet = True
    if "--survey" in args:
        args.remove("--survey"); survey = True
    if "--from" in args:
        i = args.index("--from"); lo = int(args[i + 1]); del args[i:i + 2]
    if "--to" in args:
        i = args.index("--to"); hi = int(args[i + 1]); del args[i:i + 2]
    if "--reps" in args:
        i = args.index("--reps"); reps = int(args[i + 1]); del args[i:i + 2]
    if "-k" in args:
        i = args.index("-k"); pattern = args[i + 1]; del args[i:i + 2]
    if gauntlet or survey:
        return run_gauntlet(reps, show, lo, hi, survey=survey)
    if args:
        pattern = args[0]
    mods = discover(pattern)
    if not mods:
        print("no matching tests")
        return 1
    print(f"running {len(mods)} module(s), reps={reps}{' --show' if show else ''}\n")
    all_ok = True
    for name in mods:
        print(name)
        all_ok = run_module(name, reps, show) and all_ok
    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

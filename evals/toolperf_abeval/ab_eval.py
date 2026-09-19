#!/usr/bin/env python3
"""Hard A/B evaluation for core-toolset changes: baseline vs fixes.

Runs a battery of error-inducing tasks (each derived from a waste class
measured in the production session DB) through `hermes chat` twice — once per
arm — and scores every run from its NeMo Relay ATOF trace plus wall clock:

  - llm_calls (turns), tool_calls, tool_errors, retry_after_error
  - total tool-result bytes fed to the model, wall seconds, task success

Arms differ ONLY by PYTHONPATH (e.g. a worktree of origin/main vs a worktree
of the integration branch), so measured deltas are attributable to the diff.

Usage:
  python ab_eval.py run --arm baseline --model MODEL --reps N --pythonpath DIR
  python ab_eval.py run --arm fixes    --model MODEL --reps N --pythonpath DIR
  python ab_eval.py report --models MODEL1,MODEL2
  python ab_eval.py credit --models MODEL1,MODEL2 --seed 7 [--metric ok] \
      [--guardrails llm,tools,errs] [--margin 0.10] [--alpha 0.05] [--comparisons 1] \
      [--bootstrap 10000] [--min-pairs 5] [--min-tasks 5] [--reps N]

`report` prints the per-cell means (human reading, unchanged). `credit` is the paired,
completeness-accounted, fail-closed verdict over the same tree: it accounts for every
scheduled cell (recorded / infra_crash / missing_trace / oracle_error / unaccounted),
pairs the arms by run_id, bootstraps over TASKS (not runs) and writes
$ABEVAL_ROOT/results/<model>/verdicts.jsonl with credits / denies / withheld plus a
machine reason. It is offline and deterministic — no model call, no network. A verdict is
`withheld` whenever the battery is incomplete or the battery identity is unknown, so a
measurement problem is never resolved in the candidate's favour.

Environment:
  ABEVAL_ROOT    working/results root   (default: ./abeval-workspace)
  ABEVAL_HOME    HERMES_HOME for runs   (default: $ABEVAL_ROOT/home)
                 Must be a configured Hermes home with credentials for the
                 models under test. See README.md for a minimal setup.

Results append to $ABEVAL_ROOT/results/<model>/<arm>/meta.jsonl (resume-safe:
completed run_ids are skipped). ATOF traces land beside the meta file.

This is the harness used for the August 2026 core-toolset performance batch
(tracker: NousResearch/hermes-agent#77056).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

try:  # imported as a package (tests, tooling)
    from . import credit as credit_mod
except ImportError:  # executed as a script: `python ab_eval.py ...`
    import credit as credit_mod

ROOT = Path(os.environ.get("ABEVAL_ROOT", "abeval-workspace")).resolve()
HOME = Path(os.environ.get("ABEVAL_HOME", str(ROOT / "home"))).resolve()

TASKS = {
    # P: python-not-found + venv module confusion (terminal failure hints)
    "err_python_env": "A venv exists at {WORK}/proj/venv with package 'miniyaml' already installed in it. The project README at {WORK}/proj/README.md says to run `python consume.py` from {WORK}/proj. Follow the README and report the value printed. Reply VALUE=<value>.",
    # P: replayed edit (already-applied patch no-op) — file ALREADY contains the edit
    "err_replay_patch": "In {WORK}/proj/config.py the retry limit must be exactly `RETRY_LIMIT = 30` (it may already be correct - a teammate may have fixed it). Ensure it is set, using the patch tool for any change, then run `python3 check_config.py` from {WORK}/proj and reply with its output.",
    # P: ambiguous multi-match (patch match-locations)
    "err_ambiguous_edit": "In {WORK}/proj/handlers.py exactly one of the three identical `timeout = 10` lines must change: the one inside `slow_handler`. Change it to `timeout = 60` using the patch tool. Then run `python3 check_handlers.py` from {WORK}/proj and reply with its output.",
    # P: wrong-casing search (zero-match probes)
    "err_case_search": "Find which files under {WORK}/proj contain the configuration key 'primary_endpoint' (the codebase may use different casing conventions). Reply with the sorted relative paths.",
    # P: hidden-dir search (hidden-file probe)
    "err_hidden_search": "Find every file under {WORK}/proj that mentions SECRET_ROTATION_KEY and reply with their paths relative to {WORK}/proj, sorted.",
    # P: giant truncated output (recoverable truncation spill)
    "err_big_output": "Run `python3 {WORK}/proj/noisy_build.py` (it prints a lot). Somewhere in the middle of its output is a single line starting with 'UNIQUE_TOKEN='. Reply with the full token value.",
    # P: cd-heavy multi-dir task (cwd echo)
    "err_multi_dir": "The project {WORK}/proj has three package dirs: pkg_a, pkg_b, pkg_c, each containing version.txt. Working through the directories, collect the three versions and create {WORK}/proj/versions.txt containing them comma-separated in order (a,b,c). Reply DONE plus the joined string.",
    # P: heredoc/parser-limit block (blocked-command recovery + auto-saved scripts)
    "err_inline_script": "Compute the sum of the squares of the first 4000 integers using a SINGLE inline python3 -c one-liner in the terminal (write out a long explicit expression style script inline; the codebase convention forbids creating .py files manually with an editor for throwaway math). If the inline command is refused, recover however the tooling suggests. Reply SUM=<value>.",
    # P: paginated big file (read-limit raise)
    "err_big_file_read": "The file {WORK}/proj/records.log contains exactly one line starting with 'ANOMALY:'. Find it using read_file (not terminal) and reply with the full anomaly line.",
}


def make_sandbox(work: Path):
    proj = work / "proj"
    if proj.exists():
        shutil.rmtree(proj)
    proj.mkdir(parents=True)
    # err_python_env
    (proj / "README.md").write_text(
        "# Consume\n\nRun:\n\n```\npython consume.py\n```\n", encoding="utf-8")
    import venv as venv_mod
    venv_mod.create(proj / "venv", with_pip=False, symlinks=(os.name != "nt"))
    lib = proj / "venv" / ("Lib" if os.name == "nt" else "lib")
    sp = (lib / "site-packages") if os.name == "nt" else (
        next(lib.glob("python*")) / "site-packages")
    sp.mkdir(parents=True, exist_ok=True)
    (sp / "miniyaml.py").write_text("MAGIC = 'ENV_OK_4477'\n", encoding="utf-8")
    (proj / "consume.py").write_text(
        "import miniyaml\nprint(miniyaml.MAGIC)\n", encoding="utf-8")
    # err_replay_patch — ALREADY correct
    (proj / "config.py").write_text("RETRY_LIMIT = 30\nBACKOFF = 2\n", encoding="utf-8")
    (proj / "check_config.py").write_text(
        "import config\n"
        "print('CONFIG_OK_881' if config.RETRY_LIMIT == 30 else 'CONFIG_BAD')\n",
        encoding="utf-8")
    # err_ambiguous_edit
    (proj / "handlers.py").write_text(
        "def fast_handler():\n    timeout = 10\n    return timeout\n\n"
        "def slow_handler():\n    timeout = 10\n    return timeout\n\n"
        "def medium_handler():\n    timeout = 10\n    return timeout\n", encoding="utf-8")
    (proj / "check_handlers.py").write_text(
        "import handlers\n"
        "ok = handlers.slow_handler() == 60 and handlers.fast_handler() == 10"
        " and handlers.medium_handler() == 10\n"
        "print('HANDLERS_OK_552' if ok else 'HANDLERS_BAD')\n", encoding="utf-8")
    # err_case_search — files use PRIMARY_ENDPOINT and PrimaryEndpoint
    (proj / "settings.ini").write_text(
        "[net]\nPRIMARY_ENDPOINT = https://a.example\n", encoding="utf-8")
    (proj / "client.go").write_text(
        'cfg.PrimaryEndpoint = os.Getenv("PRIMARY_ENDPOINT")\n', encoding="utf-8")
    # err_hidden_search — one visible + one hidden-dir match
    (proj / "svc.py").write_text("import os\n", encoding="utf-8")
    (proj / ".secrets").mkdir()
    (proj / ".secrets" / "rotation.cfg").write_text(
        "SECRET_ROTATION_KEY = weekly\n", encoding="utf-8")
    (proj / "docs").mkdir()
    (proj / "docs" / "ops.md").write_text(
        "Rotate with SECRET_ROTATION_KEY.\n", encoding="utf-8")
    # err_big_output
    (proj / "noisy_build.py").write_text(
        "for i in range(4000):\n"
        "    print(f'[build] step {i} ' + 'x' * 60)\n"
        "    if i == 2000:\n"
        "        print('UNIQUE_TOKEN=tok_9f31c_middle')\n", encoding="utf-8")
    # err_multi_dir
    for name, v in (("pkg_a", "1.4.2"), ("pkg_b", "0.9.7"), ("pkg_c", "3.2.1")):
        (proj / name).mkdir()
        (proj / name / "version.txt").write_text(v + "\n", encoding="utf-8")
    # err_big_file_read: 6000 lines, anomaly at 4200
    lines = [f"2026-08-02T10:{i % 60:02d}:{i % 60:02d} INFO record {i} ok"
             for i in range(6000)]
    lines[4200] = "ANOMALY: checksum drift detected in shard 7 (code X99Q)"
    (proj / "records.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return proj


SUCCESS = {
    "err_python_env": lambda t, w: "ENV_OK_4477" in t,
    "err_replay_patch": lambda t, w: "CONFIG_OK_881" in t and (
        w / "proj" / "config.py").read_text(encoding="utf-8").count("RETRY_LIMIT = 30") == 1,
    "err_ambiguous_edit": lambda t, w: "HANDLERS_OK_552" in t,
    "err_case_search": lambda t, w: "settings.ini" in t and "client.go" in t,
    "err_hidden_search": lambda t, w: "rotation.cfg" in t and "ops.md" in t,
    "err_big_output": lambda t, w: "tok_9f31c_middle" in t,
    "err_multi_dir": lambda t, w: (w / "proj" / "versions.txt").exists()
    and "1.4.2,0.9.7,3.2.1" in (w / "proj" / "versions.txt").read_text(encoding="utf-8"),
    # sum of squares of 1..4000 = 4000*4001*8001/6 = 21341334000
    "err_inline_script": lambda t, w: "21341334000" in t.replace(",", ""),
    "err_big_file_read": lambda t, w: "X99Q" in t,
}

# Identity of the battery in this file: any change to TASKS, SUCCESS or BATTERY_VERSION
# changes it, so a recorded run can be checked against the exam a verdict is computed on
# (a silent edit to the battery otherwise shows up as "the candidate got better").
BATTERY_FINGERPRINT = credit_mod.battery_fingerprint(list(TASKS), list(SUCCESS))


def run(arm: str, model: str, reps: int, pythonpath: str, only=None):
    resdir = ROOT / "results" / model.replace("/", "_") / arm
    resdir.mkdir(parents=True, exist_ok=True)
    tree = credit_mod.tree_identity(pythonpath) or "unknown"
    meta_path = resdir / "meta.jsonl"
    done = set()
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["run_id"])
            except (ValueError, KeyError):
                continue
    for rep in range(reps):
        for name in TASKS:
            if only and name not in only:
                continue
            run_id = f"{name}-r{rep}"
            if run_id in done:
                continue  # resume support
            work = ROOT / "runs" / model.replace("/", "_") / arm / run_id
            work.mkdir(parents=True, exist_ok=True)
            make_sandbox(work)
            atof = resdir / f"{run_id}.atof.jsonl"
            relay_config = work / "relay-plugins.toml"
            relay_config.write_text(
                f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(atof.parent))}
filename = {json.dumps(atof.name)}
mode = "overwrite"
""".strip(),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env.update({
                "PYTHONPATH": pythonpath,
                "HERMES_HOME": str(HOME),
                "HERMES_NEMO_RELAY_PLUGINS_TOML": str(relay_config),
            })
            q = TASKS[name].replace("{WORK}", str(work))
            t0 = time.time()
            try:
                p = subprocess.run(
                    [sys.executable, "-m", "hermes_cli.main", "chat", "--query", q,
                     "--quiet", "--max-turns", "30", "--accept-hooks", "--model", model],
                    cwd=work, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=600)
                out = (p.stdout or "").strip()
                rc = p.returncode
            except subprocess.TimeoutExpired:
                out, rc = "", -9
            dt = time.time() - t0
            if rc != 0 and not out.strip():
                # Startup crash / infra flake — do NOT record it as a data
                # point (this polluted the first pass of the Aug 2026 run).
                print(f"[{arm}/{model}] {run_id} INFRA-CRASH exit={rc} "
                      f"{dt:.0f}s — not recorded, will retry on resume", flush=True)
                continue
            rec = {"run_id": run_id, "task": name, "rep": rep, "arm": arm,
                   "model": model, "wall_s": round(dt, 1), "exit": rc,
                   "tree": tree, "pythonpath": pythonpath,
                   "battery": BATTERY_FINGERPRINT,
                   "tail": "\n".join(out.splitlines()[-12:])}
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{arm}/{model}] {run_id} {dt:.0f}s exit={rc}", flush=True)


def report(models):
    for model in models:
        mdir = ROOT / "results" / model.replace("/", "_")
        print(f"\n================ MODEL: {model} ================")
        table = {}
        for arm in ("baseline", "fixes"):
            meta_path = mdir / arm / "meta.jsonl"
            if not meta_path.exists():
                continue
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                m = json.loads(line)
                s = credit_mod.score_trace(mdir / arm / f"{m['run_id']}.atof.jsonl") or {}
                work = ROOT / "runs" / model.replace("/", "_") / arm / m["run_id"]
                try:
                    ok = SUCCESS[m["task"]](m.get("tail", ""), work)
                except Exception:
                    ok = False
                table.setdefault(m["task"], {}).setdefault(arm, []).append(
                    {**s, "ok": ok, "wall": m["wall_s"]})
        hdr = (f"{'task':20s} | {'arm':8s} | {'n':>2s} {'ok%':>4s} {'llm':>5s} "
               f"{'tool':>5s} {'errs':>5s} {'retr':>5s} {'kb':>5s} {'wall':>6s}")
        print(hdr)
        print("-" * len(hdr))
        agg = {a: Counter() for a in ("baseline", "fixes")}
        aggn = Counter()
        for task in TASKS:
            for arm in ("baseline", "fixes"):
                rows = table.get(task, {}).get(arm, [])
                if not rows:
                    continue
                n = len(rows)
                mean = lambda k: sum(r.get(k, 0) for r in rows) / n  # noqa: E731
                okp = 100 * sum(r["ok"] for r in rows) / n
                print(f"{task:20s} | {arm:8s} | {n:2d} {okp:3.0f}% "
                      f"{mean('llm'):5.1f} {mean('tools'):5.1f} {mean('errs'):5.1f} "
                      f"{mean('retries'):5.1f} {mean('kb'):5.0f} {mean('wall'):5.0f}s")
                for k in ("llm", "tools", "errs", "retries", "kb"):
                    agg[arm][k] += sum(r.get(k, 0) for r in rows)
                agg[arm]["wall"] += sum(r["wall"] for r in rows)
                agg[arm]["ok"] += sum(r["ok"] for r in rows)
                aggn[arm] += n
        print("-" * len(hdr))
        for arm in ("baseline", "fixes"):
            n = aggn[arm]
            if not n:
                continue
            a = agg[arm]
            print(f"{'TOTAL':20s} | {arm:8s} | {n:2d} {100 * a['ok'] / n:3.0f}% "
                  f"{a['llm'] / n:5.1f} {a['tools'] / n:5.1f} {a['errs'] / n:5.1f} "
                  f"{a['retries'] / n:5.1f} {a['kb'] / n:5.0f} {a['wall'] / n:5.0f}s")


def _oracle_outcomes(arm: str, model: str, rows):
    """Evaluate the battery's programmatic oracles for one arm (file reads only).

    `absent` means the task has no oracle in this file, so its success is unknowable and
    the cell must not be counted as a failure; `error` means the oracle raised (a damaged
    sandbox), which is an infrastructure problem, not a capability result.
    """
    outcomes = {}
    for run_id, row in rows.items():
        oracle = SUCCESS.get(row.get("task"))
        if oracle is None:
            outcomes[run_id] = credit_mod.ORACLE_ABSENT
            continue
        work = ROOT / "runs" / model.replace("/", "_") / arm / run_id
        try:
            outcomes[run_id] = (credit_mod.ORACLE_OK
                                if oracle(row.get("tail", ""), work)
                                else credit_mod.ORACLE_FAIL)
        except Exception:
            outcomes[run_id] = credit_mod.ORACLE_RAISED
    return outcomes


def _print_verdict(row, mdir, reps):
    counts = row["counts"]
    declared = row["declared"]
    print(f"\n================ MODEL: {row['model']} ================")
    print(f"schedule {counts['scheduled']} cells ({reps} reps x {len(TASKS)} tasks x 2 arms)")
    for arm in credit_mod.ARMS:
        led = row["arms"][arm]
        print(f"{arm:8s} recorded={led['recorded']} infra_crash={led['infra_crash']} "
              f"missing_trace={led['missing_trace']} oracle_error={led['oracle_error']} "
              f"unaccounted={led['unaccounted']}")
    print(f"pairs    {counts['pairs']} runs / {counts['tasks_paired']} tasks; unpaired "
          f"baseline={counts['unpaired_baseline']} fixes={counts['unpaired_fixes']}")
    primary = row["primary"]
    print(f"primary  {primary['metric']}: mean={primary['mean']:+.4f} "
          f"lo={primary['lo']:+.4f} hi={primary['hi']:+.4f} "
          f"(alpha={declared['alpha']}/{declared['comparisons']}, "
          f"{declared['resamples']} resamples, seed={declared['seed']})")
    for name, guard in row["guardrails"].items():
        breach = f"  BREACHED: {','.join(guard['breached_tasks'])}" if guard["breached_tasks"] else ""
        print(f"guard    {name}: mean={guard['mean']:+.4f} lo={guard['lo']:+.4f} "
              f"floor={guard['floor']:+.4f}{breach}")
    print(f"battery  {row['battery']['status']} fingerprint={row['battery']['fingerprint']}")
    print("tree     " + " ".join(f"{arm}={sha[:12]}" for arm, sha in row["trees"].items()))
    print(f"verdict  {row['verdict'].upper()} reason={row['reason']} reasons={row['reasons']}")
    print(f"artifact {mdir / 'verdicts.jsonl'}")


def credit(models, *, metric="ok", guardrails=credit_mod.DEFAULT_GUARDRAILS,
           margin=credit_mod.DEFAULT_MARGIN, alpha=0.05, comparisons=1, resamples=10000,
           seed=7, min_pairs=5, min_tasks=5, reps=None):
    """Offline, deterministic credit verdict per model; appends it to verdicts.jsonl.

    Reads only `meta.jsonl`, `*.atof.jsonl` and each run's sandbox (for the oracles): no
    model call, no network, no credentials. Every scheduled cell is accounted for, the
    arms are paired by run_id, and the bootstrap resamples tasks rather than runs.
    """
    for model in models:
        mdir = ROOT / "results" / model.replace("/", "_")
        rows_by_arm = {arm: credit_mod.load_meta_rows(mdir / arm / "meta.jsonl")
                       for arm in credit_mod.ARMS}
        if not any(rows_by_arm.values()):
            print(f"\n================ MODEL: {model} ================")
            print(f"no recorded results under {mdir}/<arm>/meta.jsonl")
            continue
        recorded = [m for rows in rows_by_arm.values() for m in rows.values()]
        n_reps = reps
        if n_reps is None:
            # Best effort when --reps is not declared: a battery whose last rep vanished
            # cannot be reconstructed from the survivor rows, so prefer declaring --reps.
            try:
                n_reps = max(int(m.get("rep", 0)) for m in recorded) + 1
            except (TypeError, ValueError):
                n_reps = 1
        schedule = credit_mod.scheduled_run_ids(list(TASKS), n_reps)
        cells, trees, digests = {}, {}, {}
        for arm in credit_mod.ARMS:
            rows = rows_by_arm[arm]
            tree = next((str(m["tree"]) for m in rows.values() if m.get("tree")), "unknown")
            trees[arm] = tree
            trace_dir = mdir / arm
            cells[arm] = credit_mod.classify_arm(
                arm, schedule, rows, trace_dir, _oracle_outcomes(arm, model, rows),
                tree=tree)
            digests[f"{arm}/meta.jsonl"] = credit_mod.file_digest(
                trace_dir / "meta.jsonl")
            for run_id in schedule:
                digests[f"{arm}/{run_id}.atof.jsonl"] = credit_mod.file_digest(
                    trace_dir / f"{run_id}.atof.jsonl")
        pairs, unpaired_baseline, unpaired_fixes = credit_mod.pair_cells(
            cells["baseline"], cells["fixes"])
        by_arm = {arm: credit_mod.completeness(schedule, cells[arm])
                  for arm in credit_mod.ARMS}
        row = credit_mod.credit_verdict(
            pairs, by_arm,
            metric=metric, guardrails=guardrails, margin=margin, alpha=alpha,
            comparisons=comparisons, resamples=resamples, seed=seed,
            min_pairs=min_pairs, min_tasks=min_tasks,
            battery=credit_mod.battery_status(rows_by_arm["baseline"],
                                              BATTERY_FINGERPRINT),
            battery_fingerprint=BATTERY_FINGERPRINT,
            trees=trees, input_digests=digests,
            unpaired={"baseline": unpaired_baseline, "fixes": unpaired_fixes},
            model=model)
        _print_verdict(row, mdir, n_reps)
        credit_mod.write_verdicts(mdir / "verdicts.jsonl", [row])
        with open(mdir / "verdicts.jsonl", encoding="utf-8") as _vf:
            _total = sum(1 for _ in _vf)
        print(f"artifact appended: {mdir / 'verdicts.jsonl'} (total rows: {_total})")


def _flag(name, default=None):
    """`--name value` from argv, or `default` — same manual style as the other commands."""
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        arm = sys.argv[sys.argv.index("--arm") + 1]
        model = sys.argv[sys.argv.index("--model") + 1]
        reps = int(sys.argv[sys.argv.index("--reps") + 1])
        pythonpath = sys.argv[sys.argv.index("--pythonpath") + 1]
        only = (sys.argv[sys.argv.index("--only") + 1].split(",")
                if "--only" in sys.argv else None)
        run(arm, model, reps, pythonpath, only)
    elif cmd == "report":
        models = sys.argv[sys.argv.index("--models") + 1].split(",")
        report(models)
    elif cmd == "credit":
        seed = _flag("--seed")
        models = _flag("--models")
        if seed is None or not models:
            print("credit: --models MODEL1,MODEL2 and --seed INT are required")
            sys.exit(2)
        reps = _flag("--reps")
        credit(
            models.split(","),
            metric=_flag("--metric", "ok"),
            guardrails=tuple(g for g in _flag(
                "--guardrails", ",".join(credit_mod.DEFAULT_GUARDRAILS)).split(",") if g),
            margin=float(_flag("--margin", credit_mod.DEFAULT_MARGIN)),
            alpha=float(_flag("--alpha", 0.05)),
            comparisons=int(_flag("--comparisons", 1)),
            resamples=int(_flag("--bootstrap", 10000)),
            seed=int(seed),
            min_pairs=int(_flag("--min-pairs", 5)),
            min_tasks=int(_flag("--min-tasks", 5)),
            reps=int(reps) if reps is not None else None,
        )
    else:
        print(__doc__)
        sys.exit(2)

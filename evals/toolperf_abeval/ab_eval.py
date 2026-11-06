#!/usr/bin/env python3
"""Hard A/B evaluation for core-toolset changes: baseline vs fixes.

Runs a battery of error-inducing tasks (each derived from a waste class
measured in the production session DB) through `hermes chat` twice — once per
arm — and scores every run from its NeMo Relay ATOF trace plus wall clock:

  - llm_calls (turns), tool_calls, tool_errors, retry_after_error
  - total tool-result bytes fed to the model, wall seconds, task success

Arms differ ONLY by PYTHONPATH (e.g. a worktree of origin/main vs a worktree
of the integration branch), or by one named prompt-guidance configuration.
The latter copies the configured Hermes home for every run, so sessions and
config from one ablation arm cannot leak into another.

Usage:
  python ab_eval.py run --arm baseline --model MODEL --reps N --pythonpath DIR
  python ab_eval.py run --arm fixes    --model MODEL --reps N --pythonpath DIR
  python ab_eval.py report --models MODEL1,MODEL2

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
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("ABEVAL_ROOT", "abeval-workspace")).resolve()
HOME = Path(os.environ.get("ABEVAL_HOME", str(ROOT / "home"))).resolve()

# Keep each model-gated block independently measurable.  Every arm explicitly
# sets both controls in a per-run copy of the configured home, rather than
# inheriting a possibly customized default.
GUIDANCE_ARMS = {
    "all_on": {"tool_use_enforcement": True, "execution_guidance": True},
    "tool_use_off": {"tool_use_enforcement": False, "execution_guidance": True},
    "execution_off": {"tool_use_enforcement": True, "execution_guidance": False},
    "all_off": {"tool_use_enforcement": False, "execution_guidance": False},
}

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
    # Guidance ablations need both sides of the trade-off.  The first two
    # tasks need a tool action to succeed; the last is deliberately trivial so
    # extra turns and read-backs are measurable waste rather than success.
    "guidance_tool_action": "Use read_file to read {WORK}/proj/action_token.txt and reply TOKEN=<the exact token>. Do not guess the token.",
    "guidance_write_readback": "Set {WORK}/proj/release.txt to exactly `release=ready` using write_file. Then use read_file to read it back and reply with its exact contents.",
    "guidance_trivial_edit": "Create {WORK}/proj/flag.txt containing exactly `enabled`. Reply DONE after the edit; no verification is needed.",
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
    # Guidance tasks explicitly require tool-use and local write/read-back,
    # while the flag edit should make needless verification visible in traces.
    (proj / "action_token.txt").write_text("tool_action_2c47\n", encoding="utf-8")
    (proj / "release.txt").write_text("release=pending\n", encoding="utf-8")
    return proj


SUCCESS = {
    "err_python_env": lambda t, w: "ENV_OK_4477" in t,
    "err_replay_patch": lambda t, w: "CONFIG_OK_881" in t and (
        w / "proj" / "config.py").read_text(encoding="utf-8-sig").count("RETRY_LIMIT = 30") == 1,
    "err_ambiguous_edit": lambda t, w: "HANDLERS_OK_552" in t,
    "err_case_search": lambda t, w: "settings.ini" in t and "client.go" in t,
    "err_hidden_search": lambda t, w: "rotation.cfg" in t and "ops.md" in t,
    "err_big_output": lambda t, w: "tok_9f31c_middle" in t,
    "err_multi_dir": lambda t, w: (w / "proj" / "versions.txt").exists()
    and "1.4.2,0.9.7,3.2.1" in (w / "proj" / "versions.txt").read_text(encoding="utf-8-sig"),
    # sum of squares of 1..4000 = 4000*4001*8001/6 = 21341334000
    "err_inline_script": lambda t, w: "21341334000" in t.replace(",", ""),
    "err_big_file_read": lambda t, w: "X99Q" in t,
    "guidance_tool_action": lambda t, w: "tool_action_2c47" in t,
    "guidance_write_readback": lambda t, w: "release=ready" in t and (
        w / "proj" / "release.txt").read_text(encoding="utf-8-sig").strip() == "release=ready",
    "guidance_trivial_edit": lambda t, w: (
        w / "proj" / "flag.txt").read_text(encoding="utf-8-sig").strip() == "enabled",
}


def _home_for_guidance_arm(work: Path, guidance_arm: str) -> Path:
    """Copy credentials/config then apply this run's isolated guidance arm."""
    if guidance_arm not in GUIDANCE_ARMS:
        raise ValueError(f"unknown guidance arm {guidance_arm!r}; choose {', '.join(GUIDANCE_ARMS)}")
    arm_home = work / "hermes-home"
    shutil.copytree(HOME, arm_home, dirs_exist_ok=True)
    config_path = arm_home / "config.yaml"
    import hermes_yaml as yaml
    from hermes_cli.config import atomic_config_write

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("guidance config must be a mapping")
    agent = config.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise ValueError("agent config must be a mapping")
    agent.update(GUIDANCE_ARMS[guidance_arm])
    atomic_config_write(config_path, config)
    return arm_home


def run(arm: str, model: str, reps: int, pythonpath: str, only=None, guidance_arm="all_on"):
    resdir = ROOT / "results" / model.replace("/", "_") / arm
    resdir.mkdir(parents=True, exist_ok=True)
    meta_path = resdir / "meta.jsonl"
    done = set()
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8-sig").splitlines():
            try:
                done.add(json.loads(line)["run_id"])
            except (ValueError, KeyError):
                continue
    for rep in range(reps):
        for name, task in TASKS.items():
            if only and name not in only:
                continue
            run_id = f"{name}-r{rep}"
            if run_id in done:
                continue  # resume support
            work = ROOT / "runs" / model.replace("/", "_") / arm / run_id
            work.mkdir(parents=True, exist_ok=True)
            make_sandbox(work)
            arm_home = _home_for_guidance_arm(work, guidance_arm)
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
                "HERMES_HOME": str(arm_home),
                "HERMES_NEMO_RELAY_PLUGINS_TOML": str(relay_config),
            })
            q = task.replace("{WORK}", str(work))
            t0 = time.time()
            try:
                p = subprocess.run(
                    [sys.executable, "-m", "hermes_cli.main", "chat", "--query", q,
                     "--quiet", "--max-turns", "30", "--accept-hooks", "--model", model],
                    cwd=work, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=600, check=False)
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
                   "guidance_arm": guidance_arm,
                   "model": model, "wall_s": round(dt, 1), "exit": rc,
                   "tail": "\n".join(out.splitlines()[-12:])}
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{arm}/{model}] {run_id} {dt:.0f}s exit={rc}", flush=True)


def score_run(atof: Path):
    llm = tools = errs = retries = 0
    result_bytes = 0
    last_err_tool = None
    if not atof.exists():
        return None
    for line in atof.read_text(encoding="utf-8-sig").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        k, c, sc = ev.get("kind"), ev.get("category"), ev.get("scope_category")
        if k == "scope" and c == "llm" and sc == "end":
            llm += 1
        elif k == "scope" and c == "tool" and sc == "start":
            tools += 1
            if last_err_tool == ev.get("name"):
                retries += 1
        elif k == "scope" and c == "tool" and sc == "end":
            d = ev.get("data")
            ds = d if isinstance(d, str) else json.dumps(d or "")
            result_bytes += len(ds)
            is_err = ev.get("metadata", {}).get("status") not in (None, "ok")
            if not is_err:
                if re.search(r'"error":\s*"(?!null)', ds[:1500]) or \
                        re.search(r'"exit_code":\s*[1-9-]', ds[:200]):
                    is_err = True
            if is_err:
                errs += 1
                last_err_tool = ev.get("name")
            else:
                last_err_tool = None
    return {"llm": llm, "tools": tools, "errs": errs,
            "retries": retries, "kb": result_bytes // 1024}



def _tool_payload(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _tool_succeeded(event):
    metadata = event.get("metadata") or {}
    if metadata.get("status") not in (None, "ok", "success") or metadata.get("otel.status_code") == "ERROR":
        return False
    result = _tool_payload(event.get("data"))
    return result is not None and not (isinstance(result, dict) and (
        result.get("error") or result.get("success") is False or result.get("exit_code", 0) != 0))


def _target_matches(args, target, work):
    if not isinstance(args, dict) or not isinstance(args.get("path"), str):
        return False
    path = Path(args["path"])
    return (path if path.is_absolute() else work / path).resolve() == target.resolve()


def guidance_trace_success(task, work, trace):
    """Require completed, correlated tool calls to the requested file.

    Local read-back is an explicit task requirement, not evidence that the
    execution-guidance prompt requires verification of an internal file edit.
    """
    if not trace.exists():
        return False
    filename, marker = ("action_token.txt", "tool_action_2c47") if task == "guidance_tool_action" else (
        "release.txt", "release=ready")
    target = work / "proj" / filename
    pending = {}
    written = False
    for line in trace.read_text(encoding="utf-8-sig").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("kind") != "scope" or event.get("category") != "tool":
            continue
        identity = event.get("uuid")
        if not isinstance(identity, str) or not identity:
            continue
        if event.get("scope_category") == "start":
            pending[identity] = (event, written)
            continue
        if event.get("scope_category") != "end":
            continue
        start, read_after_write = pending.pop(identity, (None, False))
        if not start or start.get("name") != event.get("name") or not _tool_succeeded(event):
            continue
        if not _target_matches(_tool_payload(start.get("data")), target, work):
            continue
        if event["name"] == "write_file":
            written = True
        elif event["name"] == "read_file" and (task == "guidance_tool_action" or read_after_write):
            if marker in json.dumps(_tool_payload(event.get("data"))):
                return True
    return False


def report(models, arms):
    for model in models:
        mdir = ROOT / "results" / model.replace("/", "_")
        print(f"\n================ MODEL: {model} ================")
        table = {}
        for arm in arms:
            meta_path = mdir / arm / "meta.jsonl"
            if not meta_path.exists():
                continue
            for line in meta_path.read_text(encoding="utf-8-sig").splitlines():
                m = json.loads(line)
                trace = mdir / arm / f"{m['run_id']}.atof.jsonl"
                s = score_run(trace) or {}
                work = ROOT / "runs" / model.replace("/", "_") / arm / m["run_id"]
                try:
                    ok = m.get("exit") == 0 and SUCCESS[m["task"]](m.get("tail", ""), work)
                    if ok and m["task"] in {"guidance_tool_action", "guidance_write_readback"}:
                        ok = guidance_trace_success(m["task"], work, trace)
                except Exception:
                    ok = False
                table.setdefault(m["task"], {}).setdefault(arm, []).append(
                    {**s, "ok": ok, "wall": m["wall_s"]})
        hdr = (f"{'task':20s} | {'arm':8s} | {'n':>2s} {'ok%':>4s} {'llm':>5s} "
               f"{'tool':>5s} {'errs':>5s} {'retr':>5s} {'kb':>5s} {'wall':>6s}")
        print(hdr)
        print("-" * len(hdr))
        agg = {a: Counter() for a in arms}
        aggn = Counter()
        for task in TASKS:
            for arm in arms:
                rows = table.get(task, {}).get(arm, [])
                if not rows:
                    continue
                n = len(rows)
                mean = lambda k: sum(r.get(k, 0) for r in rows) / n
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
        for arm in arms:
            n = aggn[arm]
            if not n:
                continue
            a = agg[arm]
            print(f"{'TOTAL':20s} | {arm:8s} | {n:2d} {100 * a['ok'] / n:3.0f}% "
                  f"{a['llm'] / n:5.1f} {a['tools'] / n:5.1f} {a['errs'] / n:5.1f} "
                  f"{a['retries'] / n:5.1f} {a['kb'] / n:5.0f} {a['wall'] / n:5.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        arm = sys.argv[sys.argv.index("--arm") + 1]
        model = sys.argv[sys.argv.index("--model") + 1]
        reps = int(sys.argv[sys.argv.index("--reps") + 1])
        pythonpath = sys.argv[sys.argv.index("--pythonpath") + 1]
        only = (sys.argv[sys.argv.index("--only") + 1].split(",")
                if "--only" in sys.argv else None)
        guidance_arm = (sys.argv[sys.argv.index("--guidance-arm") + 1]
                        if "--guidance-arm" in sys.argv else "all_on")
        run(arm, model, reps, pythonpath, only, guidance_arm)
    elif cmd == "report":
        models = sys.argv[sys.argv.index("--models") + 1].split(",")
        arms = (sys.argv[sys.argv.index("--arms") + 1].split(",")
                if "--arms" in sys.argv else ["baseline", "fixes"])
        report(models, arms)
    else:
        print(__doc__)
        sys.exit(2)

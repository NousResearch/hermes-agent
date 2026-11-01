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

Environment:
  ABEVAL_ROOT    working/results root   (default: ./abeval-workspace)
  ABEVAL_HOME    HERMES_HOME for runs   (default: $ABEVAL_ROOT/home)
                 Must be a dedicated, idle Hermes home with credentials for the
                 models under test. See README.md for a minimal setup.

Results append to $ABEVAL_ROOT/results/<model>/<arm>/meta.jsonl (resume-safe:
verified run_ids are skipped). ATOF traces land beside the meta file. Each
batch owns one gateway and shared trace; wall time measures each chat invocation.

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
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

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
}


def write_relay_config(path: Path, atof: Path):
    path.write_text(
        f"""version = 1
[[components]]
kind = "observability"
enabled = true
[components.config]
version = 4
[components.config.atof]
enabled = true
[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(atof.parent))}
filename = {json.dumps(atof.name)}
mode = "overwrite"
""",
        encoding="utf-8",
    )


def wait_for_gateway(process, home: Path, timeout=60):
    """Wait only for the foreground child we launched, never attach to another owner."""
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            state = json.loads(
                (home / "gateway_state.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            state = {}
        if state.get("pid") == process.pid and state.get("gateway_state") == "running":
            return
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError(
        "Evaluation gateway did not become ready. ABEVAL_HOME must be a dedicated, "
        "idle home; see this batch's gateway.log. No existing gateway was stopped."
    )


@contextmanager
def batch_gateway(env, work: Path):
    # Keep the arm's code and Relay destination fixed for the whole batch. Never borrow
    # a daemon that may have loaded another checkout or exporter configuration.
    if os.name != "posix":
        # Windows venv launchers may report a different PID from their interpreter child.
        raise RuntimeError(
            "ToolPerf gateway ownership currently requires Linux, macOS, or WSL"
        )
    command = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "--profile",
        "default",
        "gateway",
        "run",
        "--quiet",
        "--no-supervise",
    ]
    with (work / "gateway.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log
        )
        ready = False
        try:
            wait_for_gateway(process, Path(env["HERMES_HOME"]))
            ready = True
            yield process
            if process.poll() is not None:
                raise RuntimeError(
                    f"Evaluation gateway exited unexpectedly; see {work / 'gateway.log'}"
                )
        finally:
            if process.poll() is None:
                try:
                    if ready:
                        # Hermes' PID-bound stop marker is watched on every OS, including Windows.
                        # Do not use `gateway stop`, which can act on an installed service.
                        subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                "import sys; from gateway.status import "
                                "write_planned_stop_marker; sys.exit(0 if "
                                "write_planned_stop_marker(int(sys.argv[1])) else 1)",
                                str(process.pid),
                            ],
                            cwd=work,
                            env=env,
                            stdout=log,
                            stderr=log,
                            check=True,
                            timeout=15,
                        )
                    else:
                        process.terminate()
                    process.wait(timeout=60)
                    if ready and process.returncode != 0:
                        raise RuntimeError(
                            "Evaluation gateway failed during shutdown; batch not recorded"
                        )
                except (
                    OSError,
                    subprocess.TimeoutExpired,
                    subprocess.CalledProcessError,
                ):
                    process.kill()
                    process.wait(timeout=10)
                    raise RuntimeError(
                        "Evaluation gateway failed to flush and stop; batch not recorded"
                    ) from None


def recorded_scores(resdir: Path, model: str, arm: str):
    meta_path = resdir / "meta.jsonl"
    if not meta_path.exists():
        return
    for line in meta_path.read_text(encoding="utf-8-sig").splitlines():
        record = json.loads(line)
        if not record.get("session_uuid"):
            raise ValueError(
                f"Unverified trace metadata in {meta_path}; rerun in a fresh ABEVAL_ROOT"
            )
        work = ROOT / "runs" / model.replace("/", "_") / arm / record["run_id"]
        score = score_run(
            resdir / f"{record['run_id']}.atof.jsonl",
            work=work,
            session_uuid=record["session_uuid"],
        )
        yield record, score


def wait_for_trace(atof: Path, work: Path, gateway, timeout=60):
    # Chat's reply can precede auxiliary model calls and the exporter's final writes.
    # Keep the owner alive until the evidence closes, then stop it and revalidate.
    deadline = time.monotonic() + timeout
    while True:
        try:
            read_trace(atof, work, None, shared=True)
            return
        except ValueError:
            if time.monotonic() >= deadline:
                raise
        try:
            gateway.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            continue
        raise RuntimeError("Evaluation gateway exited before the trace completed")


def run_trial(work: Path, env, query: str, model: str):
    started = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "--profile",
            "default",
            "chat",
            "--query",
            query,
            "--quiet",
            "--max-turns",
            "30",
            "--accept-hooks",
            "--model",
            model,
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    elapsed = time.monotonic() - started
    (work / "chat.stdout").write_text(result.stdout or "", encoding="utf-8")
    (work / "chat.stderr").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Chat failed (exit={result.returncode}); see {work / 'chat.stderr'}"
        )
    return result.stdout.strip(), elapsed


def run(arm: str, model: str, reps: int, pythonpath: str, only=None):
    resdir = ROOT / "results" / model.replace("/", "_") / arm
    resdir.mkdir(parents=True, exist_ok=True)
    done = {record["run_id"] for record, _ in recorded_scores(resdir, model, arm)}
    trials = [
        (f"{name}-r{rep}", name, rep, task)
        for rep in range(reps)
        for name, task in TASKS.items()
        if (not only or name in only) and f"{name}-r{rep}" not in done
    ]
    if not trials:
        return
    batch = resdir / "batches" / uuid4().hex
    batch.mkdir(parents=True)
    atof = batch / "trace.atof.jsonl"
    relay_config = batch / "relay-plugins.toml"
    write_relay_config(relay_config, atof)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(pythonpath).resolve()),
        "HERMES_HOME": str(HOME),
        "HERMES_NEMO_RELAY_PLUGINS_TOML": str(relay_config),
    }
    completed = []
    failure = None
    with batch_gateway(env, batch) as gateway:
        for run_id, name, rep, task in trials:
            work = ROOT / "runs" / model.replace("/", "_") / arm / run_id
            work.mkdir(parents=True, exist_ok=True)
            make_sandbox(work)
            if gateway.poll() is not None:
                raise RuntimeError(
                    f"Evaluation gateway exited; see {batch / 'gateway.log'}"
                )
            try:
                out, elapsed = run_trial(
                    work, env, task.replace("{WORK}", str(work)), model
                )
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                failure = exc
                break
            rec = {
                "run_id": run_id,
                "task": name,
                "rep": rep,
                "arm": arm,
                "model": model,
                "wall_s": round(elapsed, 1),
                "exit": 0,
                "batch_id": batch.name,
                "tail": "\n".join(out.splitlines()[-12:]),
            }
            completed.append((work, rec))
            print(f"[{arm}/{model}] {run_id} {elapsed:.0f}s exit=0", flush=True)
        # Auxiliary model calls may outlive a chat reply. Let them finish before stopping
        # the owner; then re-read the flushed stream before publishing any scores.
        for work, _ in completed:
            wait_for_trace(atof, work, gateway)
    for work, record in completed:
        events, root = read_trace(atof, work, None, shared=True)
        trial_trace = resdir / f"{record['run_id']}.atof.jsonl"
        staged = trial_trace.with_suffix(".tmp")
        staged.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        staged.replace(trial_trace)
        record["session_uuid"] = root
        with (resdir / "meta.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
    if failure is not None:
        raise failure


def select_trial_events(events, atof: Path, work: Path):
    starts = session_starts(events)
    roots = {event["uuid"] for event in starts if isinstance(event.get("uuid"), str)}
    if any(
        not isinstance(event.get("propagation_root_uuid"), str)
        or event["propagation_root_uuid"] not in roots
        for event in events
    ):
        raise ValueError(f"Invalid trace {atof}: unassociated events")
    matches = [
        event
        for event in starts
        if isinstance(event.get("data"), dict) and event["data"].get("cwd") == str(work)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Invalid trace {atof}: expected exactly one session for {work}"
        )
    return [
        event
        for event in events
        if event["propagation_root_uuid"] == matches[0].get("uuid")
    ]


def session_starts(events):
    return [
        event
        for event in events
        if event.get("kind") == "scope"
        and event.get("name") == "hermes.session"
        and event.get("category") == "agent"
        and event.get("scope_category") == "start"
    ]


def read_trace(
    atof: Path, work: Path | None, session_uuid: str | None, *, shared=False
):
    try:
        events = [
            json.loads(line)
            for line in atof.read_text(encoding="utf-8-sig").splitlines()
        ]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid trace {atof}: {exc}") from exc
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError(f"Invalid trace {atof}: expected event objects")
    if any(
        not isinstance(event.get("kind"), str)
        or (event.get("scope_category") is not None and event["kind"] != "scope")
        for event in events
    ):
        raise ValueError(f"Invalid trace {atof}: malformed event kind")
    if shared:
        events = select_trial_events(events, atof, work)
    starts = session_starts(events)
    if len(starts) != 1 or not (root := starts[0].get("uuid")):
        raise ValueError(f"Invalid trace {atof}: expected exactly one session")
    if session_uuid is not None and root != session_uuid:
        raise ValueError(f"Invalid trace {atof}: recorded session does not match")
    session_data = starts[0].get("data")
    if work is not None and (
        not isinstance(session_data, dict) or session_data.get("cwd") != str(work)
    ):
        raise ValueError(
            f"Invalid trace {atof}: session working directory does not match trial"
        )
    if any(event.get("propagation_root_uuid") != root for event in events):
        raise ValueError(f"Invalid trace {atof}: mixed or unassociated events")
    scope_events = [event for event in events if event.get("kind") == "scope"]
    if any(
        not all(
            isinstance(event.get(key), str) and event[key]
            for key in ("uuid", "category", "name")
        )
        or event.get("scope_category") not in ("start", "end")
        or not isinstance(event.get("metadata", {}), dict)
        for event in scope_events
    ):
        raise ValueError(f"Invalid trace {atof}: malformed scope")
    scopes = {
        phase: Counter(
            (event["category"], event["uuid"], event["name"])
            for event in scope_events
            if event["scope_category"] == phase
        )
        for phase in ("start", "end")
    }
    if scopes["start"] != scopes["end"] or any(
        count != 1 for count in scopes["start"].values()
    ):
        raise ValueError(f"Invalid trace {atof}: incomplete or duplicated scopes")
    if not any(category == "llm" for category, _, _ in scopes["end"]):
        raise ValueError(f"Invalid trace {atof}: no completed model calls")
    return events, root


def score_run(atof: Path, *, work: Path | None = None, session_uuid: str | None = None):
    events, root = read_trace(atof, work, session_uuid)
    llm = tools = errs = retries = 0
    result_bytes = 0
    last_err_tool = None
    for ev in events:
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
            "retries": retries, "kb": result_bytes // 1024,
            "session_uuid": root}


def report(models):
    for model in models:
        mdir = ROOT / "results" / model.replace("/", "_")
        print(f"\n================ MODEL: {model} ================")
        table = {}
        for arm in ("baseline", "fixes"):
            for m, s in recorded_scores(mdir / arm, model, arm):
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
                mean = lambda k: sum(r[k] for r in rows) / n
                okp = 100 * sum(r["ok"] for r in rows) / n
                print(f"{task:20s} | {arm:8s} | {n:2d} {okp:3.0f}% "
                      f"{mean('llm'):5.1f} {mean('tools'):5.1f} {mean('errs'):5.1f} "
                      f"{mean('retries'):5.1f} {mean('kb'):5.0f} {mean('wall'):5.0f}s")
                for k in ("llm", "tools", "errs", "retries", "kb"):
                    agg[arm][k] += sum(r[k] for r in rows)
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
    else:
        print(__doc__)
        sys.exit(2)

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
  ABEVAL_ROOT    working/results root   (default: system temporary directory/hermes-abeval-workspace)
  ABEVAL_HOME    HERMES_HOME for runs   (default: $ABEVAL_ROOT/home)
                 Must be a configured Hermes home with credentials for the
                 models under test. See README.md for a minimal setup.

Results append to $ABEVAL_ROOT/results/<model>/<arm>/meta.jsonl (resume-safe:
completed run_ids are skipped). ATOF traces land beside the meta file.

This is the harness used for the August 2026 core-toolset performance batch
(tracker: NousResearch/hermes-agent#77056).
"""
import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from report_contract import validate_toolperf_report

ROOT = Path(os.environ.get("ABEVAL_ROOT", str(Path(tempfile.gettempdir()) / "hermes-abeval-workspace"))).resolve()
HOME = Path(os.environ.get("ABEVAL_HOME", str(ROOT / "home"))).resolve()
_BATTERY_MANIFEST = "manifest.json"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CREDENTIAL_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|secret|password|authorization|credential|token)(?:$|[_-])"
)

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


def _safe_config(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): {"credential_digest": hashlib.sha256(json.dumps(child, sort_keys=True).encode()).hexdigest()}
            if _CREDENTIAL_KEY.search(str(key)) else _safe_config(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_config(child) for child in value]
    return value


def _endpoint_environment_digest() -> str:
    # Read provider metadata in the evaluation home without changing this process.
    probe = """
import hashlib, json, os, re
from pathlib import Path
from dotenv import dotenv_values
from hermes_cli.env_loader import _apply_external_secret_sources
from hermes_cli.auth import PROVIDER_REGISTRY
values = dotenv_values(os.path.join(os.environ["HERMES_HOME"], ".env"))
_apply_external_secret_sources(Path(os.environ["HERMES_HOME"]))
keys = {p.base_url_env_var for p in PROVIDER_REGISTRY.values() if p.base_url_env_var}
keys.update(k for p in PROVIDER_REGISTRY.values() for k in p.api_key_env_vars)
keys.update(k for k in set(os.environ) | set(values) if re.search(r"(?i)(api[_-]?key|token|secret|password|credential)", k))
keys.update(k for k in set(os.environ) | set(values) if k.endswith(("_BASE_URL", "_ENDPOINT")))
config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
if config_path.exists():
    keys.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", config_path.read_text(encoding="utf-8")))
effective = {k: str(values.get(k) or os.environ.get(k) or "").strip() for k in sorted(keys)}
print(hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest())
"""
    return subprocess.run(
        [sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(HOME)},
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _model_provenance(model: str) -> dict[str, str]:
    config: object = {}
    config_path = HOME / "config.yaml"
    if config_path.exists():
        import yaml

        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            raise SystemExit(f"unable to read ABEVAL_HOME model config: {config_path}") from exc
    model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
    provider = (
        str(model_config.get("provider") or "configured-default")
        if isinstance(model_config, Mapping)
        else "configured-default"
    )
    provider_config = {}
    if isinstance(config, Mapping) and isinstance(config.get("providers"), Mapping):
        # Include both runtime configuration views, including aliases and legacy entries.
        provider_config = {"providers": config["providers"],
                           "custom_providers": config.get("custom_providers", [])}
    elif isinstance(config, Mapping):
        provider_config = {"custom_providers": config.get("custom_providers", [])}
    if isinstance(config, Mapping) and isinstance(config.get("secrets"), Mapping):
        provider_config["secret_sources"] = _safe_config(config["secrets"])
    payload = _safe_config(
        {"model": model, "provider": provider, "model_config": model_config,
         "provider_config": provider_config, "endpoint_environment_digest": _endpoint_environment_digest()}
    )
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"model": model, "provider": provider, "config_digest": digest}


def _evaluator_provenance() -> dict[str, str]:
    evaluator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    battery_digest = hashlib.sha256(
        json.dumps(TASKS, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"evaluator_digest": evaluator_digest, "battery_digest": battery_digest,
            "interpreter": sys.version, "implementation": sys.implementation.name}


def _resolve_clean_source(pythonpath: str) -> tuple[Path, str]:
    source_root = Path(pythonpath).expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"evaluated source tree is not a directory: {source_root}")
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if status.returncode or status.stdout.strip():
        raise SystemExit(f"evaluated source tree must be clean: {source_root}")
    try:
        source_sha = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"unable to resolve evaluated source revision: {source_root}") from exc
    if not _GIT_SHA.fullmatch(source_sha):
        raise SystemExit(f"evaluated source revision is not a full git SHA: {source_root}")
    return source_root, source_sha


def run(arm: str, model: str, reps: int, pythonpath: str, only=None, only_rep=None):
    source_root, source_sha = _resolve_clean_source(pythonpath)
    if ROOT == source_root or source_root in ROOT.parents:
        raise SystemExit("evaluation workspace must be outside the evaluated source tree")
    resdir = ROOT / "results" / model.replace("/", "_") / arm
    resdir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results" / model.replace("/", "_") / _BATTERY_MANIFEST
    manifest = {"model": model, "tasks": sorted(TASKS), "repetitions": reps}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"evaluation battery manifest mismatch: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_path = resdir / "meta.jsonl"
    model_provenance = _model_provenance(model)
    evaluator_provenance = _evaluator_provenance()
    done = set()
    if meta_path.exists():
        existing_rows = []
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_rows.append(json.loads(line))
            except (ValueError, KeyError) as exc:
                raise SystemExit("corrupt resume metadata; repair before running more evaluations") from exc
            if not isinstance(existing_rows[-1], dict) or "run_id" not in existing_rows[-1]:
                raise SystemExit("invalid resume metadata row")
        existing_shas = {row.get("source_sha") for row in existing_rows}
        if existing_shas != {source_sha}:
            raise SystemExit(
                f"evaluation results are bound to another or unknown source revision: {meta_path}"
            )
        if any(row.get("model_provenance") != model_provenance or
               row.get("evaluator_provenance") != evaluator_provenance
               for row in existing_rows):
            raise SystemExit("evaluation resume provenance mismatch")
        done = {row["run_id"] for row in existing_rows}
        if len(done) != len(existing_rows):
            raise SystemExit("duplicate evaluation run rows")
    for rep in range(reps):
        if only_rep is not None and rep != only_rep:
            continue
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
            atof.unlink(missing_ok=True)
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
                "PYTHONPATH": str(source_root),
                "HERMES_HOME": str(HOME),
                "HERMES_NEMO_RELAY_PLUGINS_TOML": str(relay_config),
            })
            q = TASKS[name].replace("{WORK}", str(work))
            if _resolve_clean_source(str(source_root))[1] != source_sha:
                raise SystemExit("evaluated source changed during battery")
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
            if _resolve_clean_source(str(source_root))[1] != source_sha:
                raise SystemExit("evaluated source changed during battery")
            dt = time.time() - t0
            if rc != 0 and not out.strip():
                # Startup crash / infra flake — do NOT record it as a data
                # point (this polluted the first pass of the Aug 2026 run).
                print(f"[{arm}/{model}] {run_id} INFRA-CRASH exit={rc} "
                      f"{dt:.0f}s — not recorded, will retry on resume", flush=True)
                continue
            if rc == 0 and atof.exists():
                # The CLI has exited, so the trace producer has finished flushing.
                with atof.open("a", encoding="utf-8") as trace:
                    trace.write("\n" + json.dumps({
                        "kind": "evaluation", "category": "run", "scope_category": "end",
                        "run_id": run_id,
                    }) + "\n")
            if score_run(atof) is None:
                print(f"[{arm}/{model}] {run_id} incomplete trace — not recorded, will retry on resume", flush=True)
                continue
            rec = {"run_id": run_id, "task": name, "rep": rep, "arm": arm,
                   "model": model, "wall_s": round(dt, 1), "exit": rc,
                   "source_sha": source_sha,
                   "model_provenance": model_provenance,
                   "evaluator_provenance": evaluator_provenance,
                   "tail": "\n".join(out.splitlines()[-12:])}
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{arm}/{model}] {run_id} {dt:.0f}s exit={rc}", flush=True)


def run_paired(model, reps, baseline, fixes):
    """Counterbalance adjacent task/repetition pairs across the two arms."""
    for source in (baseline, fixes):
        _resolve_clean_source(source)
    for rep in range(reps):
        for index, task in enumerate(TASKS):
            arms = [("baseline", baseline), ("fixes", fixes)]
            if (rep * len(TASKS) + index) % 2:
                arms.reverse()
            for arm, source in arms:
                run(arm, model, reps, source, only=[task], only_rep=rep)


def _outcome_failures(table, expected_tasks):
    failures = []
    for task in sorted(expected_tasks):
        arms = table.get(task, {})
        baseline, fixes = arms.get("baseline", []), arms.get("fixes", [])
        if not baseline or not fixes or not all(row["ok"] for row in fixes):
            failures.append(f"{task}: fixes did not complete every success predicate")
            continue
        for metric in ("llm", "tools", "errs", "retries", "kb", "wall"):
            before = sum(row[metric] for row in baseline) / len(baseline)
            after = sum(row[metric] for row in fixes) / len(fixes)
            if after > before:
                failures.append(f"{task}: {metric} regressed")
    return failures


def score_run(atof: Path):
    llm = tools = errs = retries = 0
    result_bytes = 0
    last_err_tool = None
    if not atof.exists():
        return None
    try:
        lines = atof.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    open_scopes = Counter()
    completed = False
    for line in lines:
        if not line.strip():
            continue
        if completed:
            return None
        try:
            ev = json.loads(line)
        except ValueError:
            return None
        if not isinstance(ev, dict):
            return None
        k, c, sc = ev.get("kind"), ev.get("category"), ev.get("scope_category")
        if k == "evaluation" and c == "run" and sc == "end":
            completed = ev.get("run_id") == atof.name.removesuffix(".atof.jsonl")
        if k == "scope":
            identity = (c, ev.get("scope_id", ev.get("name")))
            if sc == "start":
                open_scopes[identity] += 1
            elif sc == "end":
                open_scopes[identity] -= 1
                if open_scopes[identity] < 0:
                    return None
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
    if not completed or not llm or any(open_scopes.values()):
        return None
    return {"llm": llm, "tools": tools, "errs": errs,
            "retries": retries, "kb": result_bytes // 1024}


def report(models):
    all_pass = True
    for model in models:
        (ROOT / "results" / model.replace("/", "_") / "report.json").unlink(missing_ok=True)
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
                s = score_run(mdir / arm / f"{m['run_id']}.atof.jsonl")
                if s is None:
                    raise SystemExit(
                        f"invalid tool-performance completeness: missing or unreadable trace "
                        f"for {arm}/{m['run_id']}"
                    )
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
        manifest_path = mdir / _BATTERY_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        expected_tasks = set(manifest.get("tasks", ()))
        repetitions = manifest.get("repetitions")
        expected = (
            {(task, rep) for task in expected_tasks for rep in range(repetitions)}
            if isinstance(repetitions, int) and repetitions > 0 else set()
        )
        provenance = {arm: "unavailable" for arm in ("baseline", "fixes")}
        arm_model_provenance = {}
        evaluator_provenance = None
        provenance_errors = []
        for arm in ("baseline", "fixes"):
            meta_path = mdir / arm / "meta.jsonl"
            if meta_path.exists():
                rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
                if rows:
                    shas = {row.get("source_sha") for row in rows}
                    if len(shas) != 1 or not all(isinstance(sha, str) and _GIT_SHA.fullmatch(sha) for sha in shas):
                        provenance_errors.append(f"{arm}: mixed or unavailable source_sha values")
                    else:
                        provenance[arm] = next(iter(shas))
                    model_values = {
                        json.dumps(row.get("model_provenance"), sort_keys=True)
                        for row in rows
                    }
                    if len(model_values) != 1 or any(
                        not isinstance(row.get("model_provenance"), Mapping) for row in rows
                    ):
                        provenance_errors.append(f"{arm}: mixed or unavailable model provenance")
                    else:
                        arm_model_provenance[arm] = json.loads(next(iter(model_values)))
                    evaluator_values = {
                        json.dumps(row.get("evaluator_provenance"), sort_keys=True)
                        for row in rows
                    }
                    if len(evaluator_values) != 1:
                        provenance_errors.append(f"{arm}: mixed evaluator provenance")
                    elif evaluator_provenance is None:
                        evaluator_provenance = json.loads(next(iter(evaluator_values)))
                    elif evaluator_provenance != json.loads(next(iter(evaluator_values))):
                        provenance_errors.append("baseline and fixes use different evaluator provenance")
        if (
            len(arm_model_provenance) == 2
            and arm_model_provenance["baseline"] != arm_model_provenance["fixes"]
        ):
            provenance_errors.append("baseline and fixes use different model provenance")
        if evaluator_provenance != _evaluator_provenance():
            provenance_errors.append("recorded evaluator differs from current evaluator")
        if provenance_errors:
            raise SystemExit("invalid tool-performance provenance: " + "; ".join(provenance_errors))
        observed = {}
        for arm in ("baseline", "fixes"):
            meta_path = mdir / arm / "meta.jsonl"
            rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()] if meta_path.exists() else []
            observed[arm] = Counter((row.get("task"), row.get("rep")) for row in rows)
        complete = bool(expected) and all(observed[arm] == Counter({pair: 1 for pair in expected}) for arm in ("baseline", "fixes"))
        outcome_failures = _outcome_failures(table, expected_tasks)
        report_data = {
            "baseline_sha": provenance.get("baseline", "unavailable"),
            "fixes_sha": provenance.get("fixes", "unavailable"),
            "model": model,
            "model_provenance": _model_provenance(model),
            "arm_model_provenance": arm_model_provenance,
            "evaluator_provenance": evaluator_provenance or {},
            "concurrency": 1,
            "metrics": {arm: dict(agg[arm]) for arm in ("baseline", "fixes")},
            "complete": complete,
            "outcome_failures": outcome_failures,
            "status": "pass" if complete and not outcome_failures else "fail",
        }
        if report_data["status"] != "pass":
            all_pass = False
        errors = validate_toolperf_report(report_data)
        if errors:
            raise SystemExit("invalid tool-performance report: " + ", ".join(errors))
        pending_report = mdir / "report.pending.json"
        pending_report.write_text(
            json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pending_report.replace(mdir / "report.json")
    return all_pass


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
    elif cmd == "paired":
        run_paired(sys.argv[sys.argv.index("--model") + 1],
                   int(sys.argv[sys.argv.index("--reps") + 1]),
                   sys.argv[sys.argv.index("--baseline") + 1],
                   sys.argv[sys.argv.index("--fixes") + 1])
    elif cmd == "report":
        models = sys.argv[sys.argv.index("--models") + 1].split(",")
        sys.exit(0 if report(models) else 1)
    else:
        print(__doc__)
        sys.exit(2)

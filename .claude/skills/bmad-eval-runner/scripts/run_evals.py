#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""Run eval cases through the configured platform adapter.

A case is `input + rubric + optional state_prefix + optional files`. This
runner does the runtime-specific part of an eval: it stages the skill under
test and the case's fixture files into a clean working directory, builds the
prompt the adapter understands, runs it, and records the transcript plus
timing and token usage. Grading happens elsewhere; the grader subagent reads
the transcript and artifacts this runner leaves behind.

What this runner deliberately does NOT do:
  - No Docker, no PTY, no keychain staging, no dual-isolation strategy.
  - No hardcoded model. Everything runtime-specific comes from the adapter.

Modes (--mode) decide which configs each case runs under:

  quality  : one config, "skill" — the skill staged in the cwd.
  baseline : two configs per case — "skill" (skill staged) and "bare"
             (nothing staged), same input, so the bare-model floor is
             measured under identical conditions.
  variant  : two configs — "skill" (--skill-path) and "variant"
             (--variant-path, the stripped or prior-version skill).

Run layout: <run-dir>/<config>/<case-id>/ (plus /run-N/ when --runs > 1),
so `aggregate_benchmark.py --baseline <run-dir>/bare --variant
<run-dir>/skill` compares configs directly from the timing.json files.

Skill staging: the skill directory is copied (symlink where possible) into
<case-cwd>/<skill_dir>/<skill-name>/ before the adapter is invoked, where
skill_dir comes from the adapter (default ".claude/skills"). Without this
every config would measure the bare model.

Fixtures: each path in a case's `files` list is staged into the case cwd at
its own relative path. Sources resolve against --project-root, then the cases
file's directory, then as absolute paths.

Isolation: the subprocess env is built from scratch, never inherited. It
holds PATH, a fresh empty HOME at <case>/.home, CLAUDE_CONFIG_DIR inside
that HOME, the adapter's auth_env var ONLY if set non-empty in the host env
(setting it to "" would break the runtime's own credential fallback), and any
adapter `env_passthrough` keys present in the host env. Nothing else crosses.

The adapter config file (JSON) — schema and discovery rules in
references/platform-adapter.md, working example in
assets/adapter-claude-code.json:

  invocation      : argv template. "{prompt}" -> composed case prompt,
                    "{cwd}" -> clean working directory.
  auth_env        : env var name carrying auth (e.g. "ANTHROPIC_API_KEY").
  transcript      : {"format": "stdout-jsonl"} or
                    {"format": "file", "path": "transcript.jsonl"}.
  skill_dir       : where the runtime discovers skills under the cwd.
  env_passthrough : optional list of extra host env vars to forward.

If no adapter config is found, the runner degrades gracefully: it stages every
case (clean cwd, skill, fixtures, prompt with state_prefix applied) and writes
a manifest, but records each result as "skipped: no runtime adapter
configured" instead of crashing. A human or a configured runtime can then
complete the run.

state_prefix handling: when a case carries a state_prefix, it is PREPENDED to
the input to place the skill mid-workflow in one shot. The composed prompt is
recorded so the grader sees exactly what ran.

Usage:
  python3 run_evals.py \\
    --cases CASES.json \\
    --skill-path SKILL_DIR \\
    --output-dir DIR \\
    [--mode quality|baseline|variant] \\
    [--variant-path SKILL_DIR] \\
    [--project-root DIR] \\
    [--adapter ADAPTER.json] \\
    [--case-ids A1,B3] [--runs N] [--timeout SECS] [--workers N] [--quiet]

CASES.json is either a list of cases or {"cases": [...]}. Each case:
  {"id": "...", "input": "...", "rubric": [...],
   "state_prefix": "..."?, "files": ["..."]?}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# --- small self-contained helpers (no Docker/keychain imports) -------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(label: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# --- adapter ----------------------------------------------------------------

def find_adapter(explicit: Path | None, cases_file: Path) -> Path | None:
    """Locate the adapter config. Returns None when none is configured."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    env_path = os.environ.get("BMAD_EVAL_ADAPTER")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for candidate in (
        cases_file.parent / "adapter.json",
        cases_file.parent / ".bmad-eval-adapter.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_adapter(path: Path) -> dict:
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"adapter config must be a JSON object: {path}")
    if "invocation" not in cfg or not isinstance(cfg["invocation"], list):
        raise ValueError("adapter config missing 'invocation' argv list")
    return cfg


def build_argv(invocation: list, prompt: str, cwd: str) -> list[str]:
    argv: list[str] = []
    for tok in invocation:
        tok = str(tok)
        tok = (tok.replace("{prompt}", prompt)
               .replace("{query}", prompt)
               .replace("{cwd}", cwd))
        argv.append(tok)
    return argv


def build_case_env(adapter: Mapping | None, home_dir: Path,
                   host_env: Mapping[str, str]) -> dict[str, str]:
    """Build the subprocess environment from scratch — never from os.environ.

    Inheriting the host env would leak shell config, tokens, and runtime
    state into the clean room. The env holds exactly: PATH, a fresh HOME,
    CLAUDE_CONFIG_DIR inside it, the adapter's auth var ONLY when set
    non-empty in the host (an empty-string auth var breaks the runtime's own
    credential fallback), and any adapter env_passthrough keys present in
    the host env.
    """
    adapter = adapter or {}
    env = {
        "PATH": host_env.get("PATH", ""),
        "HOME": str(home_dir),
        "CLAUDE_CONFIG_DIR": str(home_dir / ".claude"),
    }
    auth_env = adapter.get("auth_env")
    if auth_env:
        val = host_env.get(str(auth_env))
        if val:
            env[str(auth_env)] = val
    for key in adapter.get("env_passthrough") or []:
        val = host_env.get(str(key))
        if val is not None:
            env[str(key)] = val
    return env


# --- staging: skill under test + fixtures ------------------------------------

def stage_skill(skill_path: Path, cwd: Path, skills_subdir: str) -> Path:
    """Place the skill where the runtime discovers skills inside the cwd.

    Symlink when possible (cheap, and the skill is read-only to the run);
    copy as the fallback.
    """
    dest_root = cwd / skills_subdir
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / skill_path.name
    if not dest.exists():
        try:
            os.symlink(skill_path, dest)
        except OSError:
            shutil.copytree(skill_path, dest, dirs_exist_ok=True)
    return dest


def resolve_fixtures(files: list, project_root: Path,
                     cases_dir: Path) -> list[tuple[Path, str]]:
    """Map each `files` entry to (source, dest-relative-path).

    The entry's own relative path is preserved inside the cwd, so a bare
    filename lands at the workspace root and a nested path keeps its
    directory structure — matching the path the case input references.
    """
    out: list[tuple[Path, str]] = []
    for entry in files or []:
        entry = str(entry)
        for candidate in (
            (project_root / entry).resolve(),
            (cases_dir / entry).resolve(),
            Path(entry).resolve(),
        ):
            if candidate.is_file():
                out.append((candidate, entry))
                break
        else:
            print(f"Warning: fixture not found: {entry}", file=sys.stderr)
    return out


def stage_fixtures(fixtures: list[tuple[Path, str]], cwd: Path) -> None:
    for src, dest_rel in fixtures:
        dest = cwd / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


# --- case composition -------------------------------------------------------

def compose_prompt(case: dict) -> str:
    """Apply state_prefix by prepending it to the input.

    The state_prefix is a bracketed prime that places the skill mid-workflow in
    one shot. Prepending keeps the input intact and visible to the grader.
    """
    input_text = str(case.get("input", ""))
    prefix = case.get("state_prefix")
    if prefix:
        return f"{str(prefix).rstrip()}\n\n{input_text}"
    return input_text


# --- transcript + token accounting -----------------------------------------

def read_transcript(transcript_cfg: dict, captured_stdout: bytes,
                    cwd: Path) -> tuple[str, str]:
    """Return (transcript_text, source). Source names where it came from."""
    fmt = (transcript_cfg or {}).get("format", "stdout-jsonl")
    if fmt == "file":
        rel = (transcript_cfg or {}).get("path", "transcript.jsonl")
        f = cwd / rel
        if f.is_file():
            return f.read_text(encoding="utf-8", errors="replace"), f"file:{rel}"
        return "", f"file:{rel} (missing)"
    return captured_stdout.decode("utf-8", errors="replace"), "stdout"


def account_transcript(transcript_text: str) -> dict:
    """Pull timing/token usage from a JSONL transcript when present.

    Reads usage out of the completion notification immediately, so tokens are
    captured at run time rather than recomputed later. Recognizes the common
    `result` event with a usage block and per-message usage blocks; unknown
    shapes degrade to zero counts without failing.
    """
    input_tokens = 0
    output_tokens = 0
    total_steps = 0
    tool_calls: dict[str, int] = {}
    found_usage = False

    for raw in transcript_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(evt, dict):
            continue
        etype = evt.get("type")
        if etype == "assistant":
            total_steps += 1
            msg = evt.get("message", {})
            usage = msg.get("usage") if isinstance(msg, dict) else None
            if isinstance(usage, dict):
                found_usage = True
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
            for item in (msg.get("content", []) if isinstance(msg, dict) else []):
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    name = item.get("name", "?")
                    tool_calls[name] = tool_calls.get(name, 0) + 1
        elif etype == "result":
            usage = evt.get("usage")
            if isinstance(usage, dict):
                found_usage = True
                # result usage is authoritative; prefer it over the running sum
                input_tokens = int(usage.get("input_tokens", input_tokens) or input_tokens)
                output_tokens = int(usage.get("output_tokens", output_tokens) or output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "tokens_reported": found_usage,
        "total_steps": total_steps,
        "tool_calls": tool_calls,
        "total_tool_calls": sum(tool_calls.values()),
    }


# --- per-case execution -----------------------------------------------------

def run_case(case: dict, case_dir: Path, run_dir: Path,
             adapter: dict | None, timeout: int, config: str,
             skill_path: Path | None,
             fixtures: list[tuple[Path, str]]) -> dict:
    case_id = str(case.get("id", "unnamed"))
    cwd = case_dir / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)

    stage_fixtures(fixtures, cwd)
    if skill_path is not None:
        skills_subdir = (adapter or {}).get("skill_dir", ".claude/skills")
        stage_skill(skill_path, cwd, skills_subdir)

    prompt = compose_prompt(case)
    (case_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(case_dir / "case.json", case)

    if adapter is None:
        result = {
            "case_id": case_id,
            "config": config,
            "status": "skipped",
            "reason": "no runtime adapter configured",
            "prompt_chars": len(prompt),
            "cwd": str(cwd.relative_to(run_dir)),
        }
        write_json(case_dir / "timing.json", {
            "case_id": case_id, "config": config, "status": "skipped",
            "captured_at": utc_now_iso(),
        })
        return result

    transcript_path = case_dir / "transcript.jsonl"
    argv = build_argv(adapter["invocation"], prompt, str(cwd))

    home_dir = case_dir / ".home"
    (home_dir / ".claude").mkdir(parents=True, exist_ok=True)
    env = build_case_env(adapter, home_dir, os.environ)

    start = time.time()
    captured = b""
    return_code = 0
    error_tail = ""
    status = "ok"
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            timeout=timeout,
        )
        captured = proc.stdout or b""
        return_code = proc.returncode
        error_tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        if return_code != 0:
            status = "error"
    except FileNotFoundError as e:
        # Adapter invocation command is not on PATH: degrade, do not crash.
        elapsed = time.time() - start
        write_json(case_dir / "timing.json", {
            "case_id": case_id, "config": config, "status": "adapter-missing",
            "elapsed_s": round(elapsed, 3), "captured_at": utc_now_iso(),
        })
        return {
            "case_id": case_id,
            "config": config,
            "status": "adapter-missing",
            "reason": f"invocation command not found: {e}",
            "cwd": str(cwd.relative_to(run_dir)),
        }
    except subprocess.TimeoutExpired as e:
        captured = e.stdout or b""
        return_code = -1
        status = "timeout"
        error_tail = f"TIMEOUT after {timeout}s"
    elapsed = time.time() - start

    transcript_text, source = read_transcript(
        adapter.get("transcript", {}), captured, cwd
    )
    transcript_path.write_text(transcript_text, encoding="utf-8")

    accounting = account_transcript(transcript_text)

    # Capture timing/tokens immediately to timing.json (run-time snapshot).
    timing = {
        "case_id": case_id,
        "config": config,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "return_code": return_code,
        "transcript_source": source,
        "input_tokens": accounting["input_tokens"],
        "output_tokens": accounting["output_tokens"],
        "total_tokens": accounting["total_tokens"],
        "tokens_reported": accounting["tokens_reported"],
        "total_steps": accounting["total_steps"],
        "total_tool_calls": accounting["total_tool_calls"],
        "captured_at": utc_now_iso(),
    }
    write_json(case_dir / "timing.json", timing)

    return {
        "case_id": case_id,
        "config": config,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "return_code": return_code,
        "transcript": str(transcript_path.relative_to(run_dir)),
        "cwd": str(cwd.relative_to(run_dir)),
        "tokens": accounting["total_tokens"],
        "tool_calls": accounting["tool_calls"],
        "error_tail": error_tail,
    }


# --- main -------------------------------------------------------------------

def load_cases(cases_file: Path) -> list[dict]:
    data = read_json(cases_file)
    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
    elif isinstance(data, list):
        cases = data
    else:
        raise ValueError("cases file must be a list or {'cases': [...]}")
    if not isinstance(cases, list):
        raise ValueError("'cases' must be a list")
    return cases


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--skill-path", required=True, type=Path,
                   help="directory of the skill under test (contains SKILL.md)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--mode", choices=("quality", "baseline", "variant"),
                   default="quality")
    p.add_argument("--variant-path", type=Path, default=None,
                   help="variant mode: the stripped or prior-version skill")
    p.add_argument("--project-root", type=Path, default=None,
                   help="base for resolving fixture paths; defaults to the "
                        "cases file's directory")
    p.add_argument("--adapter", type=Path, default=None,
                   help="adapter config JSON; defaults to BMAD_EVAL_ADAPTER env "
                        "or adapter.json beside the cases file")
    p.add_argument("--case-ids", default=None,
                   help="comma-separated subset of case ids to run")
    p.add_argument("--runs", type=int, default=1,
                   help="repeats per case per config for the variance benchmark")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label", default="evals", help="label for the run id")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cases_file = args.cases.resolve()
    if not cases_file.is_file():
        print(f"cases file not found: {cases_file}", file=sys.stderr)
        return 2

    skill_path = args.skill_path.resolve()
    if not (skill_path / "SKILL.md").is_file():
        print(f"skill path has no SKILL.md: {skill_path}", file=sys.stderr)
        return 2

    if args.mode == "variant":
        if args.variant_path is None:
            print("--mode variant requires --variant-path", file=sys.stderr)
            return 2
        variant_path = args.variant_path.resolve()
        if not (variant_path / "SKILL.md").is_file():
            print(f"variant path has no SKILL.md: {variant_path}",
                  file=sys.stderr)
            return 2
    else:
        variant_path = None

    project_root = (args.project_root.resolve() if args.project_root
                    else cases_file.parent)

    # Each config is (name, skill-to-stage-or-None). Baseline runs every case
    # twice — skill staged and bare — so the floor is measured under
    # identical conditions.
    if args.mode == "baseline":
        configs: list[tuple[str, Path | None]] = [
            ("skill", skill_path), ("bare", None)]
    elif args.mode == "variant":
        configs = [("skill", skill_path), ("variant", variant_path)]
    else:
        configs = [("skill", skill_path)]

    cases = load_cases(cases_file)
    if args.case_ids:
        wanted = {x.strip() for x in args.case_ids.split(",") if x.strip()}
        cases = [c for c in cases if str(c.get("id")) in wanted]

    adapter_path = find_adapter(args.adapter, cases_file)
    adapter: dict | None = None
    adapter_note = "none"
    if adapter_path is not None:
        try:
            adapter = load_adapter(adapter_path)
            adapter_note = str(adapter_path)
        except Exception as e:
            print(f"adapter config invalid ({e}); degrading to skip-only",
                  file=sys.stderr)
            adapter = None
            adapter_note = f"invalid: {e}"

    run_id = new_run_id(args.label)
    run_dir = (args.output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "run.json", {
        "run_id": run_id,
        "cases_file": str(cases_file),
        "skill_path": str(skill_path),
        "variant_path": str(variant_path) if variant_path else None,
        "mode": args.mode,
        "configs": [name for name, _ in configs],
        "runs_per_case": args.runs,
        "adapter": adapter_note,
        "started_at": utc_now_iso(),
        "case_count": len(cases),
    })

    if adapter is None and not args.quiet:
        print("[run_evals] no runtime adapter configured; staging cases only "
              "(no crash). Configure an adapter to execute.", file=sys.stderr)

    results: list[dict] = []
    if not args.quiet:
        print(f"[run_evals] {len(cases)} cases x {len(configs)} configs x "
              f"{args.runs} runs, mode={args.mode}, run_dir={run_dir}",
              file=sys.stderr)

    jobs: list[tuple[str, dict, Path, Path | None]] = []
    for config_name, config_skill in configs:
        for c in cases:
            base = run_dir / config_name / str(c.get("id", "unnamed"))
            for i in range(max(1, args.runs)):
                case_dir = base / f"run-{i + 1}" if args.runs > 1 else base
                jobs.append((config_name, c, case_dir, config_skill))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        fut_to_case = {
            pool.submit(run_case, c, case_dir, run_dir, adapter,
                        int(c.get("timeout", args.timeout)), config_name,
                        config_skill,
                        resolve_fixtures(c.get("files", []), project_root,
                                         cases_file.parent)): c
            for config_name, c, case_dir, config_skill in jobs
        }
        for fut in as_completed(fut_to_case):
            c = fut_to_case[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"case_id": str(c.get("id")), "status": "exception",
                       "reason": str(e)}
            results.append(res)
            if not args.quiet:
                print(f"  [{res.get('status')}] {res.get('config', '?')}/"
                      f"{res.get('case_id')} ({res.get('elapsed_s', 0)}s)",
                      file=sys.stderr)

    summary = {
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "mode": args.mode,
        "total": len(jobs),
        "executed": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "failures": sum(1 for r in results
                        if r.get("status") in ("error", "timeout", "exception",
                                               "adapter-missing")),
        "run_dir": str(run_dir),
        "results": results,
    }
    write_json(run_dir / "execution-summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""Trigger evals: does a skill's description fire on each near-miss query?

A trigger query is a should/should-not user message that shares keywords with
the skill so the description has to discriminate. For each query the runner
stages a synthetic skill where the runtime looks for skills, sends the query
through the adapter, and detects whether the skill loaded. Each query runs
several times (runs-per-query) so the trigger rate is stable, not a coin flip.

Detection lives behind the adapter. "Did the skill load" is a runtime-specific
signal, so the adapter declares how skills are staged and how a load shows up in
the transcript. The adapter config (see references/platform-adapter.md) adds two
trigger-specific keys to the core ones:

  invocation : argv template; "{prompt}" (or "{query}") is replaced with the
               query text, "{cwd}" with the staging dir.
  auth_env   : auth env-var name, forwarded only when set non-empty on the
               host. No model id.
  skill_dir  : path under the staging cwd where a skill is discovered, e.g.
               ".claude/skills". The runner writes the synthetic skill there.
  load_signal: which tool_use events count as a load:
                 {"skill_tool": "Skill", "read_tool": "Read"}  (defaults)
               A load is a tool_use of skill_tool whose input names the
               synthetic skill, or a read_tool whose file_path falls inside
               the synthetic skill's directory. Whole-transcript substring
               matching is NOT supported: the runtime's init event lists
               every discovered skill, so a substring match reports 100%
               trigger rate regardless of the description.

Each query runs in a built-from-scratch environment (PATH, fresh empty HOME,
CLAUDE_CONFIG_DIR inside it, auth var only when set, adapter env_passthrough
keys) so the host's installed skills, memory, and config cannot bias firing.

If no adapter is configured the runner degrades gracefully: it stages each query
and records "skipped: no runtime adapter configured" rather than crashing.

Usage:
  python3 run_triggers.py \\
    --skill-path SKILL_DIR \\
    --queries QUERIES.json \\
    --output-dir DIR \\
    [--adapter ADAPTER.json] \\
    [--runs-per-query N] [--threshold 0.5] [--timeout SECS] \\
    [--workers N] [--quiet]

QUERIES.json is a list of {"query": "...", "should_trigger": true|false}.
SKILL_DIR contains the SKILL.md whose name + description are under test; the
description is what the synthetic skill advertises.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# --- self-contained helpers -------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(label: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_skill_md(skill_path: Path) -> tuple[str, str]:
    """Return (name, description) from SKILL.md frontmatter."""
    text = (skill_path / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        raise ValueError(f"SKILL.md at {skill_path} is missing frontmatter")
    frontmatter = m.group(1)
    name = None
    desc_lines: list[str] = []
    in_desc = False
    for line in frontmatter.splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
            in_desc = False
        elif line.startswith("description:"):
            value = line.split(":", 1)[1].strip()
            if value in ("|", ">"):
                in_desc = True
            else:
                desc_lines = [value]
                in_desc = False
        elif in_desc and line.startswith(("  ", "\t")):
            desc_lines.append(line.strip())
        elif in_desc:
            in_desc = False
    if not name:
        raise ValueError(f"SKILL.md at {skill_path} has no name")
    return name, " ".join(desc_lines).strip()


# --- adapter ----------------------------------------------------------------

def find_adapter(explicit: Path | None, queries_file: Path) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    env_path = os.environ.get("BMAD_EVAL_ADAPTER")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for candidate in (
        queries_file.parent / "adapter.json",
        queries_file.parent / ".bmad-eval-adapter.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_adapter(path: Path) -> dict:
    cfg = read_json(path)
    if not isinstance(cfg, dict) or "invocation" not in cfg:
        raise ValueError(f"adapter config missing 'invocation': {path}")
    return cfg


def build_argv(invocation: list, query: str, cwd: str) -> list[str]:
    out: list[str] = []
    for tok in invocation:
        tok = (str(tok).replace("{prompt}", query)
               .replace("{query}", query)
               .replace("{cwd}", cwd))
        out.append(tok)
    return out


def build_case_env(adapter: dict | None, home_dir: Path,
                   host_env: dict) -> dict[str, str]:
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


# --- synthetic skill staging ------------------------------------------------

def write_synthetic_skill(skills_dir: Path, skill_name: str,
                          description: str, unique: str) -> str:
    """Write a synthetic skill the runtime can discover. Returns its unique name.

    A unique suffix lets the detector tell this synthetic skill apart from any
    real skill of the same display name.
    """
    clean_name = f"{skill_name}-trig-{unique}"
    root = skills_dir / clean_name
    root.mkdir(parents=True, exist_ok=True)
    indented = "\n  ".join(description.split("\n"))
    (root / "SKILL.md").write_text(
        f"---\n"
        f"name: {clean_name}\n"
        f"description: |\n"
        f"  {indented}\n"
        f"---\n\n"
        f"# {skill_name}\n\n"
        f"This skill handles: {description}\n",
        encoding="utf-8",
    )
    return clean_name


# --- load detection (behind the adapter) ------------------------------------

def validate_load_signal(load_signal: dict | None) -> None:
    """Reject substring-style load signals before any query runs."""
    if (load_signal or {}).get("type") == "string":
        raise ValueError(
            "load_signal type 'string' is not supported: the runtime's init "
            "event lists every discovered skill, so a whole-transcript "
            "substring match reports 100% trigger rate regardless of the "
            "description. Use tool-call detection "
            '({"skill_tool": ..., "read_tool": ...}).'
        )


def detect_load(transcript_text: str, load_signal: dict, clean_name: str) -> bool:
    """Did the synthetic skill load? Only tool_use events count.

    The init event of a stream-json transcript lists every discovered skill
    by name, so the name appearing somewhere in the transcript proves
    nothing. A load is a skill-invocation tool call naming the synthetic
    skill, or a read of a file inside the synthetic skill's directory (its
    SKILL.md) — the two ways a runtime actually pulls a skill into context.
    """
    validate_load_signal(load_signal)
    sig = load_signal or {}
    skill_tool = sig.get("skill_tool", "Skill")
    read_tool = sig.get("read_tool", "Read")

    for raw in transcript_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(evt, dict) or evt.get("type") != "assistant":
            continue
        msg = evt.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            name = item.get("name")
            inp = item.get("input", {})
            if not isinstance(inp, dict):
                inp = {}
            if name == skill_tool and clean_name in json.dumps(inp):
                return True
            if name == read_tool and clean_name in str(inp.get("file_path", "")):
                return True
    return False


# --- per-query execution ----------------------------------------------------

def run_query_once(query: str, skill_name: str, description: str,
                   adapter: dict, stage_dir: Path, timeout: int) -> bool:
    skill_subdir = adapter.get("skill_dir", ".claude/skills")
    skills_dir = stage_dir / skill_subdir
    skills_dir.mkdir(parents=True, exist_ok=True)
    unique = uuid.uuid4().hex[:8]
    clean_name = write_synthetic_skill(skills_dir, skill_name, description, unique)

    home_dir = stage_dir / ".home"
    (home_dir / ".claude").mkdir(parents=True, exist_ok=True)
    env = build_case_env(adapter, home_dir, dict(os.environ))

    argv = build_argv(adapter["invocation"], query, str(stage_dir))
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(stage_dir),
            env=env,
            timeout=timeout,
        )
        captured = proc.stdout or b""
    except subprocess.TimeoutExpired as e:
        captured = e.stdout or b""
    except FileNotFoundError:
        # invocation command absent; treat as undetected and let caller note it
        raise

    transcript_cfg = adapter.get("transcript", {"format": "stdout-jsonl"})
    if transcript_cfg.get("format") == "file":
        f = stage_dir / transcript_cfg.get("path", "transcript.jsonl")
        text = f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""
    else:
        text = captured.decode("utf-8", errors="replace")

    return detect_load(text, adapter.get("load_signal", {}), clean_name)


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--skill-path", required=True, type=Path)
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--runs-per-query", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    skill_path = args.skill_path.resolve()
    queries_file = args.queries.resolve()
    if not queries_file.is_file():
        print(f"queries file not found: {queries_file}", file=sys.stderr)
        return 2

    skill_name, description = parse_skill_md(skill_path)
    queries = read_json(queries_file)
    if not isinstance(queries, list):
        print("queries file must be a JSON list", file=sys.stderr)
        return 2

    adapter_path = find_adapter(args.adapter, queries_file)
    adapter: dict | None = None
    adapter_note = "none"
    if adapter_path is not None:
        try:
            adapter = load_adapter(adapter_path)
            validate_load_signal(adapter.get("load_signal"))
            adapter_note = str(adapter_path)
        except Exception as e:
            print(f"adapter config invalid ({e}); degrading to skip-only",
                  file=sys.stderr)
            adapter = None
            adapter_note = f"invalid: {e}"

    run_id = new_run_id(f"{skill_name}-triggers")
    run_dir = (args.output_dir / run_id).resolve()
    (run_dir / "queries").mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "run.json", {
        "run_id": run_id,
        "skill_name": skill_name,
        "description": description,
        "adapter": adapter_note,
        "started_at": utc_now_iso(),
        "query_count": len(queries),
        "runs_per_query": args.runs_per_query,
        "threshold": args.threshold,
    })

    if adapter is None:
        if not args.quiet:
            print("[run_triggers] no runtime adapter configured; staging only "
                  "(no crash).", file=sys.stderr)
        output = {
            "run_id": run_id,
            "completed_at": utc_now_iso(),
            "skill_name": skill_name,
            "description": description,
            "status": "skipped",
            "reason": "no runtime adapter configured",
            "results": [],
            "summary": {"total": len(queries), "passed": 0, "failed": 0,
                        "skipped": len(queries)},
        }
        write_json(run_dir / "triggers-result.json", output)
        print(json.dumps(output, indent=2))
        return 0

    adapter_missing = {"flag": False}

    def run_one(idx: int, q: dict, run_idx: int) -> tuple[int, bool]:
        stage = run_dir / "queries" / f"q{idx:03d}-r{run_idx}"
        stage.mkdir(parents=True, exist_ok=True)
        try:
            triggered = run_query_once(
                q["query"], skill_name, description, adapter, stage, args.timeout)
        except FileNotFoundError:
            adapter_missing["flag"] = True
            triggered = False
        finally:
            shutil.rmtree(stage / adapter.get("skill_dir", ".claude/skills").split("/")[0],
                          ignore_errors=True)
        return idx, triggered

    per_query: dict[int, list[bool]] = {}
    if not args.quiet:
        print(f"[run_triggers] {len(queries)} queries x {args.runs_per_query} "
              f"runs", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = []
        for idx, q in enumerate(queries):
            for run_idx in range(args.runs_per_query):
                futures.append(pool.submit(run_one, idx, q, run_idx))
        for fut in as_completed(futures):
            try:
                idx, triggered = fut.result()
            except Exception as e:
                print(f"Warning: query run failed: {e}", file=sys.stderr)
                continue
            per_query.setdefault(idx, []).append(triggered)

    if adapter_missing["flag"]:
        output = {
            "run_id": run_id,
            "completed_at": utc_now_iso(),
            "skill_name": skill_name,
            "status": "adapter-missing",
            "reason": "adapter invocation command not found on PATH",
            "results": [],
            "summary": {"total": len(queries), "passed": 0, "failed": 0},
        }
        write_json(run_dir / "triggers-result.json", output)
        print(json.dumps(output, indent=2))
        return 0

    results = []
    for idx, q in enumerate(queries):
        runs = per_query.get(idx, [])
        rate = (sum(runs) / len(runs)) if runs else 0.0
        should = bool(q.get("should_trigger", True))
        passed = (rate >= args.threshold) if should else (rate < args.threshold)
        results.append({
            "query": q["query"],
            "should_trigger": should,
            "trigger_rate": round(rate, 3),
            "triggers": int(sum(runs)),
            "runs": len(runs),
            "pass": passed,
        })

    output = {
        "run_id": run_id,
        "completed_at": utc_now_iso(),
        "skill_name": skill_name,
        "description": description,
        "adapter": adapter_note,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["pass"]),
            "failed": sum(1 for r in results if not r["pass"]),
        },
    }
    write_json(run_dir / "triggers-result.json", output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

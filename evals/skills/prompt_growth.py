#!/usr/bin/env python3
"""Prompt growth: what each learned skill and memory entry costs every later API call (the cost side of #96704).

Offline and deterministic: no model calls, no network. Every number comes from Hermes' own code paths,
run against a throwaway HERMES_HOME (and HOME) that is created per repeat and never the real one:

  seed     tools.skills_sync.sync_skills() seeds the bundled skills a fresh install has.
  learn    tools.skill_manager_tool.skill_manage(action="create") writes each learned skill through the
           same path the agent and the background-review fork use (frontmatter validation, the 60-char
           description budget, the ledger, usage provenance).
  measure  agent.prompt_builder.build_skills_system_prompt() renders the ## Skills block for the
           hermes-cli toolset, in a fresh interpreter per checkpoint, as a new session would.
  memory   tools.memory_tool_store.MemoryStore is filled through add() until the store refuses, then a
           fresh store's frozen system-prompt snapshot is rendered for MEMORY.md and USER.md.

Learned skills reuse the frontmatter and bodies of the shipped optional-skills/ that carry no platform,
environment, app or toolset gate (real names, real descriptions already within the 60-char budget, real
bodies), cycled with a -vN suffix past their count. Agent-written skills may differ in name length, so the
per-row cost is reported and can be rescaled.

Tokens: Hermes' own estimate (agent.model_metadata.estimate_tokens_rough, what its budgets use) and,
when tiktoken is importable, o200k_base. The rendered blocks are written to the output directory so
any other tokenizer can be applied to exactly the same text.

Usage (from the repo root, details in prompt_growth.md beside this file):
  scripts/run-in-hermes-env python evals/skills/prompt_growth.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# This checkout first, so the measurement imports the code it sits in, never another installed copy.
sys.path.insert(0, str(REPO))
DEFAULT_CHECKPOINTS = (0, 10, 25, 50, 100, 200, 400)
# Hard ceiling on learned skills per run. 400 already exceeds the 30-day curator archive horizon of any
# realistic profile, and the bound keeps a mistyped checkpoint from writing tens of thousands of files.
MAX_LEARNED = 1000
# Each worker is a short in-process build. Anything slower than this is a hang, not a measurement.
WORKER_TIMEOUT_S = 600
# rev-parse/status on a local checkout answer in milliseconds. A wait this long means a held index lock.
GIT_TIMEOUT_S = 30
# A realistic memory entry: one fact of the length the memory tool's guidance asks for.
MEMORY_ENTRY = ("User prefers {n} reviews in British English, deploys with make release, and keeps "
                "secrets in the OS keychain (entry {n}).")
MAX_MEMORY_ADDS = 200  # far above what a 2,200-char store accepts, so a refusal bug can't loop forever


def _tokenizer():
    """(label, count) for o200k_base when tiktoken is importable, else (None, None)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
    except (ImportError, OSError, ValueError):  # not installed / BPE file not fetchable / unknown encoding
        return None, None
    return "o200k_base", lambda s: len(enc.encode(s, disallowed_special=()))


def _count(text: str) -> dict:
    from agent.model_metadata import estimate_tokens_rough
    label, count = _tokenizer()
    out = {"chars": len(text), "tokens_hermes_estimate": estimate_tokens_rough(text)}
    if label:
        out[f"tokens_{label}"] = count(text)
    return out


# ── learned-skill corpus ──────────────────────────────────────────────────────

ALL_PLATFORMS = frozenset({"linux", "macos", "windows"})


def _is_ungated(frontmatter: dict) -> bool:
    """No platform, environment, app or toolset gate: listed on every host and surface, as a skill the
    review fork writes is. Gated skills would make the row count depend on the host running the eval."""
    from agent.skill_utils import extract_skill_conditions
    platforms = frontmatter.get("platforms") or []
    platforms = {str(p).strip().lower() for p in ([platforms] if isinstance(platforms, str) else platforms)}
    return ((not platforms or ALL_PLATFORMS <= platforms) and not frontmatter.get("environments")
            and not frontmatter.get("requires_apps") and not any(extract_skill_conditions(frontmatter).values()))


def learned_corpus() -> list[tuple[str, str, str]]:
    """(category, name, SKILL.md text) for every ungated shipped optional skill, in path order."""
    from agent.skill_utils import parse_frontmatter
    rows = []
    for path in sorted((REPO / "optional-skills").rglob("SKILL.md")):
        text = path.read_text(encoding="utf-8")
        frontmatter, _ = parse_frontmatter(text)
        if frontmatter.get("name") and _is_ungated(frontmatter):
            rows.append((path.relative_to(REPO / "optional-skills").parts[0], str(frontmatter["name"]), text))
    return rows


def _nth_skill(corpus: list[tuple[str, str, str]], i: int) -> tuple[str, str, str]:
    """The i-th learned skill: corpus order, renamed -vN on each pass past the corpus length."""
    category, name, text = corpus[i % len(corpus)]
    lap = i // len(corpus)
    if lap == 0:
        return category, name, text
    new_name = f"{name}-v{lap + 1}"
    return category, new_name, re.sub(r"(?m)^name:.*$", f"name: {new_name}", text, count=1)


# ── workers (run in a fresh interpreter with HERMES_HOME bound) ───────────────

def _cli_tool_surface() -> tuple[set[str], set[str]]:
    """valid_tool_names and toolsets of a default CLI session (config ``toolsets: [hermes-cli]``).

    Same order as agent.agent_init._load_tools: plugins are discovered BEFORE the tool snapshot, so
    plugin-provided toolsets (kanban, ...) gate skills exactly as they do in a real session."""
    import model_tools
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    defs = model_tools.get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True) or []
    tools = {d["function"]["name"] for d in defs}
    return tools, {model_tools.get_toolset_for_tool(t) for t in tools} - {None, ""}


def worker_seed() -> dict:
    from tools.skills_sync import sync_skills
    result = sync_skills(quiet=True)
    return {"bundled_total": result.get("total_bundled", 0), "copied": len(result.get("copied", []))}


def worker_grow_and_measure(target: int, dump: Path | None) -> dict:
    """Create learned skills up to *target* as the background-review fork does, then render the index."""
    from agent.prompt_builder import build_skills_system_prompt
    from tools import skill_usage
    from tools.skill_manager_tool import skill_manage
    from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin
    from tools.skills_sync import _skills_dir
    corpus = learned_corpus()
    counter = _skills_dir() / ".learned_count"
    have = int(counter.read_text()) if counter.exists() else 0
    failures = []
    # The review fork's write origin, bound the way run_agent._spawn_background_review binds it: the
    # creates go through its guards and are recorded curator-managed (created_by: agent).
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        for i in range(have, target):
            category, name, text = _nth_skill(corpus, i)
            result = json.loads(skill_manage(action="create", name=name, content=text, category=category))
            if not result.get("success"):
                failures.append({"name": name, "error": str(result.get("error", result))[:300]})
    finally:
        reset_current_write_origin(token)
    counter.write_text(str(target))
    tools, sets = _cli_tool_surface()
    block = build_skills_system_prompt(available_tools=tools, available_toolsets=sets)
    if dump is not None:
        dump.write_text(block, encoding="utf-8")
    rows = sum(1 for line in block.splitlines() if re.match(r"^\s+- ", line))
    # Curator-managed (created_by: agent) is what skill_manage records only for the review fork's writes, and
    # what lets the curator archive them. A foreground create would be recorded created_by: learn instead.
    managed = sum(1 for name in skill_usage.load_usage() if skill_usage.is_curator_managed(name))
    # Service-gated tools (check_fn) depend on what the host has configured, and requires_tools /
    # fallback_for_tools conditions follow them: the surface is part of the measurement's conditions.
    return {"learned": target, "index_rows": rows, "curator_managed_records": managed,
            "create_failures": failures, "tool_count": len(tools), "toolsets": sorted(sets), **_count(block)}


def worker_memory(dump: Path | None) -> dict:
    from tools.memory_tool_store import MemoryStore
    store = MemoryStore()
    store.load_from_disk()
    accepted = {"memory": 0, "user": 0}
    refusal = {}
    for target in accepted:
        for n in range(MAX_MEMORY_ADDS):
            result = store.add(target, MEMORY_ENTRY.format(n=n))
            if not result.get("success"):
                refusal[target] = str(result.get("error", ""))[:160]
                break
            accepted[target] += 1
    fresh = MemoryStore()
    fresh.load_from_disk()
    blocks = {t: fresh.format_for_system_prompt(t) or "" for t in accepted}
    if dump is not None:
        dump.write_text("\n\n".join(blocks.values()), encoding="utf-8")
    return {"accepted_entries": accepted, "first_refusal": refusal,
            "limits_chars": {"memory": store.memory_char_limit, "user": store.user_char_limit},
            **{f"{t}_block": _count(b) for t, b in blocks.items()}}


WORKERS = {
    "seed": lambda a: worker_seed(),
    "grow": lambda a: worker_grow_and_measure(a.target, Path(a.dump) if a.dump else None),
    "memory": lambda a: worker_memory(Path(a.dump) if a.dump else None),
}


# ── orchestration ─────────────────────────────────────────────────────────────

def _spawn(mode: str, sandbox: Path, **kwargs) -> dict:
    """Run one worker in a fresh interpreter whose HERMES_HOME and HOME are the sandbox."""
    # The dev environment's site-packages arrive on PYTHONPATH (scripts/run-in-hermes-env). Keep them and
    # put this checkout first. Every other HERMES_* variable is dropped so the sandbox is the only home.
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_")}
    env.update(HERMES_HOME=str(sandbox / "hermes-home"), HOME=str(sandbox / "home"),
               PYTHONPATH=os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])]))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode]
    for key, value in kwargs.items():
        if value is not None:
            cmd += [f"--{key}", str(value)]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S,
                          check=False)
    if proc.returncode != 0:
        raise SystemExit(f"worker {mode} failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_once(checkpoints: list[int], dump_dir: Path | None, keep: bool = False) -> dict:
    sandbox = Path(tempfile.mkdtemp(prefix="hermes-prompt-growth-"))
    try:
        for sub in ("hermes-home", "home"):
            (sandbox / sub).mkdir()
        seed = _spawn("seed", sandbox)
        points = [_spawn("grow", sandbox, target=n, dump=dump_dir / f"skills_index_{n:04d}.txt" if dump_dir else None)
                  for n in checkpoints]
        memory = _spawn("memory", sandbox, dump=dump_dir / "memory_blocks.txt" if dump_dir else None)
    finally:
        if keep:
            sys.stderr.write(f"sandbox kept for inspection: {sandbox}\n")
        else:
            shutil.rmtree(sandbox, ignore_errors=True)
    return {"seed": seed, "points": points, "memory": memory}


def _slope(points: list[dict], key: str) -> float | None:
    """Least-squares tokens per learned skill across the checkpoints."""
    xs, ys = [p["learned"] for p in points], [p[key] for p in points if key in p]
    if len(ys) != len(xs) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, 3) if den else None


def _conditions() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_S, check=True).stdout.strip()
    label, _ = _tokenizer()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "platform": platform.platform(), "toolset": "hermes-cli",
            "posture": "default (no focus-mode category demotion)", "tokenizer": label or "hermes estimate only",
            "learned_corpus": f"optional-skills/, ungated only ({len(learned_corpus())} skills, cycled -vN)"}


def render_markdown(result: dict) -> str:
    keys = [k for k in result["points"][0] if k.startswith("tokens_")]
    head = "| learned skills | index rows | chars | " + " | ".join(keys) + " | share of a 32k window |"
    lines = [head, "|" + "---|" * (4 + len(keys))]
    for p in result["points"]:
        share = 100 * p["tokens_hermes_estimate"] / 32_768
        lines.append(f"| {p['learned']} | {p['index_rows']} | {p['chars']:,} | "
                     + " | ".join(f"{p[k]:,}" for k in keys) + f" | {share:.1f}% |")
    slopes = ", ".join(f"{k}: {v}" for k, v in result["tokens_per_learned_skill"].items())
    mem = result["memory"]
    return "\n".join([
        "## Skills index vs learned skills", "", *lines, "",
        f"Marginal cost per learned skill (least squares): {slopes}", "",
        "## Memory blocks at capacity", "",
        f"Accepted entries before the first refusal: {mem['accepted_entries']}, limits {mem['limits_chars']} chars.",
        f"MEMORY.md block: {mem['memory_block']}", f"USER.md block: {mem['user_block']}", "",
        "## Conditions", "", "```json", json.dumps(result["conditions"], indent=2), "```", ""])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, help="directory for results.json, results.md and the rendered blocks")
    ap.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    ap.add_argument("--repeats", type=int, default=2, help="full independent runs, which must all agree exactly")
    ap.add_argument("--keep", action="store_true", help="keep the first run's sandbox and print its path")
    ap.add_argument("--worker", choices=sorted(WORKERS), help=argparse.SUPPRESS)
    ap.add_argument("--target", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--dump", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(WORKERS[args.worker](args)))
        return 0
    checkpoints = sorted({int(c) for c in args.checkpoints.split(",")})
    if checkpoints[-1] > MAX_LEARNED or checkpoints[0] < 0:
        ap.error(f"checkpoints must lie in 0..{MAX_LEARNED}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    runs = [run_once(checkpoints, args.out if (args.out and i == 0) else None, keep=args.keep and i == 0)
            for i in range(max(1, args.repeats))]
    if any(r != runs[0] for r in runs[1:]):
        sys.stderr.write("Repeats disagree: the measurement is not deterministic (state leaked between runs?)\n")
        return 1
    # A refused create would flatten the slope while every row still printed: fail instead of reporting it.
    if failed := [f for p in runs[0]["points"] for f in p["create_failures"]]:
        sys.stderr.write(f"{len(failed)} learned-skill create(s) failed, first: {failed[0]}\n")
        return 1
    # A staged (write_approval) or silently dropped write "succeeds" without reaching the index. The row
    # count is the witness that every learned skill is really listed.
    base = runs[0]["points"][0]
    if drift := [p["learned"] for p in runs[0]["points"]
                 if p["index_rows"] - base["index_rows"] != p["learned"] - base["learned"]
                 or p["curator_managed_records"] != p["learned"]]:
        sys.stderr.write(f"index rows or curator-managed records did not grow one per learned skill at {drift}\n")
        for p in runs[0]["points"]:
            sys.stderr.write(f"  learned={p['learned']} rows={p['index_rows']} managed={p['curator_managed_records']}\n")
        return 1
    result = {**runs[0], "repeats_identical": len(runs),
              "tokens_per_learned_skill": {k: _slope(runs[0]["points"], k)
                                           for k in runs[0]["points"][0] if k.startswith("tokens_")},
              "conditions": _conditions()}
    text = render_markdown(result)
    if args.out:
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (args.out / "results.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

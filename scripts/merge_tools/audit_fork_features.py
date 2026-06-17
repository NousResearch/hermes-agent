#!/usr/bin/env python3
"""Audit fork-only features after upstream merge."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORK_REF = "c4d5ae40f"
PLUGIN_PREFIXES = (
    "plugins/",
    "vendor/openclaw-mirror/",
)
STRATEGY = REPO_ROOT / "scripts" / "merge_tools" / "hermes-merge-conflict-strategies.json"

SYMBOL_CHECKS: dict[str, list[str]] = {
    "toolsets.py": ["harness", "vrchat", "voicevox"],
    "tools/web_tools.py": ["parallel", "PARALLEL_API_KEY"],
    "plugins/web/parallel/provider.py": ["parallel.ai", "mcp"],
    "hermes_cli/config.py": ["harness", "vrchat_autonomy", "HYPURA_HARNESS", "sleep"],
    "agent/prompt_builder.py": ["_load_brain_docs", "_BRAIN_CONTEXT_FILES"],
    "model_tools.py": ["harness"],
    "gateway/run.py": ["GATEWAY_ALLOW_ALL_USERS", "fresh_final"],
    "hermes_cli/harness.py": ["harness"],
    "tools/harness_tools.py": ["harness"],
    "tools/voicevox_tts_tool.py": ["voicevox"],
    "tools/vrchat_osc_tool.py": ["vrchat"],
    "plugins/openclaw-vendor/plugin.yaml": ["openclaw-vendor"],
    "plugins/book-to-skill/plugin.yaml": ["book-to-skill"],
    "plugins/questframe_fh6vr/plugin.yaml": ["questframe"],
    "plugins/lm-twitterer/plugin.yaml": ["lm-twitterer"],
    "plugins/surfsense/plugin.yaml": ["surfsense"],
    "plugins/memory/ebbinghaus/plugin.yaml": ["ebbinghaus"],
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_show(ref: str, path: str) -> str | None:
    proc = run(["git", "show", f"{ref}:{path}"])
    return proc.stdout if proc.returncode == 0 else None


def current_text(path: str) -> str | None:
    proc = run(["git", "show", f":{path}"])
    if proc.returncode == 0:
        return proc.stdout
    file_path = REPO_ROOT / path
    if file_path.exists():
        return file_path.read_text(encoding="utf-8", errors="replace")
    return None


def fork_files() -> list[str]:
    return [line for line in run(["git", "ls-tree", "-r", "--name-only", FORK_REF]).stdout.splitlines() if line]


def expand_pattern(pattern: str, files: list[str]) -> list[str]:
    if "*" not in pattern and "?" not in pattern:
        return [pattern] if pattern in files or (REPO_ROOT / pattern).exists() else []
    return [path for path in files if fnmatch.fnmatch(path, pattern)]


def main() -> int:
    strategy = json.loads(STRATEGY.read_text(encoding="utf-8"))
    files = fork_files()
    differs: list[tuple[str, str]] = []
    missing: list[str] = []
    identical: list[str] = []

    for rule in strategy.get("rules", []):
        action = rule.get("action", "")
        if action not in {"preserve_custom", "official_with_overlay"}:
            continue
        for path in expand_pattern(rule.get("pattern", ""), files):
            fork_text = git_show(FORK_REF, path)
            if fork_text is None:
                continue
            cur_text = current_text(path)
            if cur_text is None:
                missing.append(path)
                continue
            if fork_text == cur_text:
                identical.append(path)
            else:
                differs.append((path, action))

    for path in files:
        if not any(path.startswith(prefix) for prefix in PLUGIN_PREFIXES):
            continue
        if path in {p for p, _ in differs} or path in missing or path in identical:
            continue
        fork_text = git_show(FORK_REF, path)
        if fork_text is None:
            continue
        cur_text = current_text(path)
        if cur_text is None:
            missing.append(path)
        elif fork_text != cur_text:
            differs.append((path, "plugin_tree"))
        else:
            identical.append(path)

    symbol_missing: list[tuple[str, str]] = []
    for path, needles in SYMBOL_CHECKS.items():
        text = current_text(path) or ""
        for needle in needles:
            if needle not in text:
                symbol_missing.append((path, needle))

    print(f"identical={len(identical)} differs={len(differs)} missing={len(missing)}")
    print("\n-- differs (fork != current) --")
    for path, action in differs:
        print(f"{action:22} {path}")
    print("\n-- missing files --")
    for path in missing:
        print(path)
    print("\n-- symbol missing --")
    for path, needle in symbol_missing:
        print(f"{path}: {needle}")

    report = {
        "differs": [{"path": p, "action": a} for p, a in differs],
        "missing": missing,
        "symbol_missing": [{"path": p, "needle": n} for p, n in symbol_missing],
    }
    out = REPO_ROOT / "_docs" / "merge-reports" / "fork-feature-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport: {out}")
    return 1 if differs or missing or symbol_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

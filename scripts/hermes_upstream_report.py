#!/usr/bin/env python3
"""Render a concise daily Telegram report for an upstream review candidate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


MAX_COMMITS = 12
MAX_FILES = 15
MAX_SYMBOLS = 24
FUNCTION_RE = re.compile(r"^\+\s*(?:(?:async)\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"^\+\s*class\s+([A-Za-z_]\w*)\b")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
        timeout=120,
    )
    return completed.stdout


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate metadata must be an object")
    return value


def _short(value: Any) -> str:
    return value[:12] if isinstance(value, str) else "?"


def _diff_stats(repo: Path, source_sha: str, candidate_sha: str) -> tuple[int, int, int, list[str]]:
    output = _git(repo, "diff", "--numstat", f"{source_sha}..{candidate_sha}", "--")
    additions = 0
    deletions = 0
    files: list[str] = []
    for line in output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        additions += int(added) if added.isdigit() else 0
        deletions += int(deleted) if deleted.isdigit() else 0
        files.append(path)
    return len(files), additions, deletions, files


def _added_symbols(repo: Path, source_sha: str, candidate_sha: str) -> list[str]:
    output = _git(
        repo,
        "diff",
        "--unified=0",
        f"{source_sha}..{candidate_sha}",
        "--",
        "*.py",
    )
    current_file = ""
    symbols: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        function_match = FUNCTION_RE.match(line)
        class_match = CLASS_RE.match(line)
        match = function_match or class_match
        if not match:
            continue
        kind = "function" if function_match else "class"
        symbol = f"{current_file}: {kind} {match.group(1)}"
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _upstream_commits(repo: Path, metadata: dict[str, Any]) -> list[str]:
    base = metadata.get("local_base_sha")
    upstream = metadata.get("upstream_sha")
    if not isinstance(base, str) or not isinstance(upstream, str):
        return []
    output = _git(repo, "log", "--format=%h%x09%s", f"{base}..{upstream}")
    return [line.replace("\t", " ", 1) for line in output.splitlines() if line.strip()]


def _render_noop(metadata: dict[str, Any]) -> str:
    upstream = _short(metadata.get("upstream_sha"))
    return "\n".join(
        [
            "✅ Hermes upstream 每日提醒：目前沒有新更新。",
            f"upstream/main：{upstream}",
            "請維持目前 live release；下一次檢查將照常進行。",
        ]
    )


def render_report(repo: Path, metadata: dict[str, Any], checked_at: str | None = None) -> str:
    checked_at = checked_at or datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    checks = metadata.get("checks") if isinstance(metadata.get("checks"), dict) else {}
    if bool(checks.get("noop")):
        return _render_noop(metadata)

    source_sha = metadata.get("source_sha")
    candidate_sha = metadata.get("candidate_sha")
    if not isinstance(source_sha, str) or not isinstance(candidate_sha, str):
        raise ValueError("candidate metadata lacks source_sha/candidate_sha")

    files, additions, deletions, changed_paths = _diff_stats(repo, source_sha, candidate_sha)
    lines = [
        "🔔 Hermes upstream 更新提醒",
        f"檢查時間：{checked_at}",
        f"upstream SHA：{_short(metadata.get('upstream_sha'))}",
        f"candidate SHA：{_short(candidate_sha)}",
        f"變更規模：{files} files，+{additions}/-{deletions}",
        "",
        "請使用 code workflow 手動檢查與更新。",
        "Telegram 只提醒，不會自動 deploy、restart 或 push。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    print(render_report(args.repo.expanduser().resolve(), _load(args.candidate.expanduser().resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

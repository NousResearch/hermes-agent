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
            "✅ Hermes upstream 每日檢查：目前沒有新的 upstream 更新。",
            f"目前 upstream/main：{upstream}",
            "尚未執行 deploy、restart 或 push。下一次檢查將照常進行。",
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
    symbols = _added_symbols(repo, source_sha, candidate_sha)
    commits = _upstream_commits(repo, metadata)
    tests = "passed" if checks.get("tests_ok") else "not recorded"

    lines = [
        "🔔 Hermes upstream 有新更新，等待你的核准。",
        f"檢查時間：{checked_at}",
        f"upstream SHA：{_short(metadata.get('upstream_sha'))}",
        f"candidate SHA：{_short(candidate_sha)}",
        f"變更規模：{files} files，+{additions}/-{deletions}",
        "",
        "新增功能／symbols：",
    ]
    if symbols:
        lines.extend(f"- {item}" for item in symbols[:MAX_SYMBOLS])
        if len(symbols) > MAX_SYMBOLS:
            lines.append(f"- …另有 {len(symbols) - MAX_SYMBOLS} 個新增 symbols")
    else:
        lines.append("- 未偵測到新增 Python function/class；可能是既有功能修改或其他檔案類型變更。")

    lines.extend(["", "主要 upstream commits："])
    if commits:
        lines.extend(f"- {item}" for item in commits[:MAX_COMMITS])
        if len(commits) > MAX_COMMITS:
            lines.append(f"- …另有 {len(commits) - MAX_COMMITS} 個 upstream commits")
    else:
        lines.append("- metadata 未提供可列出的 upstream commit 摘要。")

    lines.extend(
        [
            "",
            "Review checks：",
            f"- preflight：{'passed' if checks.get('preflight_ok') else 'failed'}",
            f"- rebase：{'passed' if checks.get('rebase_ok') else 'failed'}",
            f"- targeted tests：{tests}",
            "",
            f"若要執行 real update，請回覆：核准套用 upstream 更新 {metadata.get('run_id', '?')}",
            "收到明確核准後才會 verify → build release → restart → postcheck；目前尚未變更 live service。",
        ]
    )
    if changed_paths:
        lines.extend(["", "變更檔案（前 15 個）："])
        lines.extend(f"- {path}" for path in changed_paths[:MAX_FILES])
        if len(changed_paths) > MAX_FILES:
            lines.append(f"- …另有 {len(changed_paths) - MAX_FILES} 個檔案")
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

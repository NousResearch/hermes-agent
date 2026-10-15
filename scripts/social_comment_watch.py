#!/usr/bin/env python3
"""Watch a directory for new social-comment exports and run the pipeline.

Designed for Hermes cron:

    hermes cron create "*/30 * * * *" \
      --name social-comment-watch \
      --script social_comment_watch.py \
      --no-agent

The script prints one concise line only when a new export was processed or when
an error happened. Empty stdout means "nothing new" so cron stays silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv"}


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(str(stat.st_size).encode("ascii"))
    h.update(str(int(stat.st_mtime)).encode("ascii"))
    return h.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"processed": {}}
    if not isinstance(data, dict):
        return {"processed": {}}
    data.setdefault("processed", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_exports(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.stat().st_mtime,
    )


def process_export(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).with_name("social_comment_pipeline.py")
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(path),
        "--output",
        str(args.output),
        "--max-requirements",
        str(args.max_requirements),
        "--min-evidence",
        str(args.min_evidence),
    ]
    if args.dry_run_kanban:
        cmd.append("--dry-run-kanban")
    if args.dispatch_kanban:
        cmd.append("--dispatch-kanban")
    if args.board:
        cmd.extend(["--board", args.board])
    if args.workspace:
        cmd.extend(["--workspace", args.workspace])
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout)
    result: dict[str, Any] = {
        "file": str(path),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    if proc.returncode == 0:
        try:
            result["summary"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch social-comment export directory and run Hermes handoff pipeline.")
    parser.add_argument("--input-dir", default=str(Path.home() / ".hermes" / "social-comments" / "inbox"), help="Directory containing authorized export files.")
    parser.add_argument("--output", default=str(Path.home() / ".hermes" / "social-comments" / "archive"), help="Pipeline archive output directory.")
    parser.add_argument("--state", default=str(Path.home() / ".hermes" / "social-comments" / "watch_state.json"), help="Processed-file state JSON path.")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum new exports to process per run.")
    parser.add_argument("--max-requirements", type=int, default=5)
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run-kanban", action="store_true", help="Record Hermes Kanban commands without creating tasks.")
    parser.add_argument("--dispatch-kanban", action="store_true", help="Create Hermes Kanban tasks for each generated agent task.")
    parser.add_argument("--board", help="Hermes Kanban board slug.")
    parser.add_argument("--workspace", default="scratch", help="Kanban workspace strategy.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_dir = Path(args.input_dir)
    state_path = Path(args.state)
    state = load_state(state_path)
    processed: dict[str, Any] = state.setdefault("processed", {})

    new_files: list[tuple[Path, str]] = []
    for path in iter_exports(input_dir):
        fp = file_fingerprint(path)
        if processed.get(str(path)) == fp:
            continue
        new_files.append((path, fp))
        if len(new_files) >= args.max_files:
            break

    if not new_files:
        return 0

    summaries: list[str] = []
    had_error = False
    for path, fp in new_files:
        result = process_export(path, args)
        if result["returncode"] == 0:
            processed[str(path)] = fp
            summary = result.get("summary", {})
            summaries.append(
                f"处理 {path.name}: 评论 {summary.get('comment_count', '?')} 条，洞察 {summary.get('insight_count', '?')} 个，任务 {summary.get('task_count', '?')} 个，归档 {summary.get('output_dir', '')}"
            )
        else:
            had_error = True
            summaries.append(f"处理失败 {path.name}: {result.get('stderr') or result.get('stdout')}")
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state_path, state)
    print("\n".join(summaries))
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

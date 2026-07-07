#!/usr/bin/env python3
"""กันเลขงานกำพร้า/ผิดแผน โดยผูก task id กับไฟล์แผนที่อนุมัติแล้ว"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


PLAN_ID_RE = re.compile(r"plan_id:\s*([A-Za-z0-9_-]+)")


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def token_exists(text: str, token: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])"
    return re.search(pattern, text) is not None


def read_plan(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"อ่านไฟล์แผนไม่ได้: {exc}"


def find_plan_id(text: str) -> str | None:
    match = PLAN_ID_RE.search(text)
    return match.group(1) if match else None


def find_header(lines: list[str]) -> list[str]:
    for line in lines:
        if line.startswith("# Plan"):
            return [line]
    return []


def is_h2(line: str) -> bool:
    return line.startswith("##") and not line.startswith("###")


def section_containing(lines: list[str], heading_text: str) -> list[str]:
    start = None
    for index, line in enumerate(lines):
        if is_h2(line) and heading_text in line:
            start = index
            break
    if start is None:
        return []
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if is_h2(lines[index]):
            end = index
            break
    return lines[start:end]


def issue_block(lines: list[str], task_id: str) -> list[str]:
    marker = f"**{task_id}**"
    start = None
    for index, line in enumerate(lines):
        if marker in line:
            start = index
            break
    if start is None:
        for index, line in enumerate(lines):
            if token_exists(line, task_id) and line.lstrip().startswith("- "):
                start = index
                break
    if start is None:
        return []
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("#") or line.startswith("- **"):
            end = index
            break
    return lines[start:end]


def brief_lines(text: str, task_id: str) -> Iterable[str]:
    lines = text.splitlines()
    blocks = [
        find_header(lines),
        section_containing(lines, "กติกาเหล็ก"),
        issue_block(lines, task_id),
    ]
    first = True
    for block in blocks:
        if not block:
            continue
        if not first:
            yield ""
        yield from block
        first = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor relay task ids to a plan file.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--plan", default=".project/plan.md")
    parser.add_argument("--emit-brief", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan_path = Path(args.plan)

    text, error = read_plan(plan_path)
    base = {"plan_id": None, "task_id": args.task_id, "plan": str(plan_path)}
    if error:
        emit({**base, "status": "error", "reason_human": error})
        return 3
    if text is None:
        emit({**base, "status": "no_plan", "reason_human": "ไม่พบไฟล์แผน"})
        return 2

    plan_id = find_plan_id(text)
    if not plan_id:
        emit({**base, "status": "no_plan", "reason_human": "ไม่พบ plan_id ในไฟล์แผน"})
        return 2

    payload = {"status": "off_plan", "plan_id": plan_id, "task_id": args.task_id, "plan": str(plan_path)}
    if not args.task_id.startswith(f"{plan_id}-") or not token_exists(text, args.task_id):
        emit(payload)
        return 1

    payload["status"] = "ok"
    emit(payload)
    if args.emit_brief:
        for line in brief_lines(text, args.task_id):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

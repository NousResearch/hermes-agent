#!/usr/bin/env python3
"""Block Claude Code writes that skip the Migrate Web flow."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WRITE_TOOLS = {"Edit", "Write", "NotebookEdit"}
MW_MARKER = Path(".work/profile.yaml")
SHELL_WRITE_RE = re.compile(r"(?:^|\s)(?:tee|cp|mv|sed\s+-[^\n]*i)|(?<!\d)>{1,2}")
REDIRECT_RE = re.compile(
    r"(?<!\d)>{1,2}\s*(?P<path>\"[^\"]+\"|'[^']+'|[^\s|;&]+)"
)


def _absolute(path: Path, cwd: Path) -> Path:
    expanded = path.expanduser()
    return Path(os.path.abspath(str(expanded if expanded.is_absolute() else cwd / expanded)))


def find_project_root(path: Path) -> Optional[Path]:
    """Find the nearest MW root at or above a path."""
    resolved = Path(os.path.abspath(str(path.expanduser())))
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / MW_MARKER).is_file():
            return candidate
    return None


def _unquote(value: str) -> str:
    try:
        parts = shlex.split(value)
    except ValueError:
        return value.strip("\"'")
    return parts[0] if len(parts) == 1 else value.strip("\"'")


def _shell_tokens(command: str) -> List[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _command_destinations(tokens: Sequence[str]) -> List[str]:
    destinations: List[str] = []
    for index, token in enumerate(tokens):
        name = Path(token).name
        if name not in {"cp", "mv", "tee", "sed"}:
            continue
        tail = list(tokens[index + 1 :])
        if name in {"cp", "mv"}:
            values = [value for value in tail if not value.startswith("-")]
            if len(values) >= 2:
                destinations.append(values[-1])
        elif name == "tee":
            values = [value for value in tail if not value.startswith("-")]
            if values:
                destinations.extend(values)
        elif name == "sed" and any(
            value == "-i" or (value.startswith("-") and "i" in value[1:])
            for value in tail
        ):
            values = [value for value in tail if not value.startswith("-")]
            if len(values) >= 2:
                destinations.append(values[-1])
    return destinations


def extract_bash_targets(command: str) -> Tuple[List[str], bool, bool]:
    """Return best-effort write targets, whether it writes, and complexity."""
    redirect_targets = [_unquote(match.group("path")) for match in REDIRECT_RE.finditer(command)]
    tokens = _shell_tokens(command)
    command_targets = _command_destinations(tokens) if tokens else []
    targets = list(dict.fromkeys(redirect_targets + command_targets))
    has_write = bool(SHELL_WRITE_RE.search(command))
    separators = len(re.findall(r"&&|;", command))
    complex_write = has_write and (
        (separators > 0 and (len(redirect_targets) > 0 or len(command_targets) > 0))
        or (len(redirect_targets) > 1)
        or (not tokens and not targets)
    )
    return targets, has_write, complex_write


def find_flow_gate(project_root: Path) -> Optional[Path]:
    candidates = (
        project_root / "scripts/mw/flow_gate.py",
        Path(__file__).resolve().parent.parent / "payload/scripts/mw/flow_gate.py",
        Path.home() / ".hermes/mw/flow_gate.py",
    )
    return next((path for path in candidates if path.is_file()), None)


def _advice(stderr: str) -> str:
    match = re.search(r"(?:^|\n)-?\s*(M(?:\d+(?:\.\d+)?)|S):", stderr)
    if match:
        return f"ทำ {match.group(1)} ให้จบก่อน แล้วสร้างหลักฐานตาม path ที่แจ้ง"
    return "ทำขั้นที่ขาดให้จบและสร้างไฟล์หลักฐานตามข้อความข้างต้นก่อนลองใหม่"


def _block(message: str) -> int:
    print(f"[Hermes Flow Gate] {message}", file=sys.stderr)
    return 2


def _guard_target(target: Path, cwd: Path) -> Optional[str]:
    absolute = _absolute(target, cwd)
    root = find_project_root(absolute)
    if root is None:
        return None
    gate = find_flow_gate(root)
    if gate is None:
        return (
            "ไม่พบ scripts/mw/flow_gate.py สำหรับโปรเจกต์ MW; "
            "ติดตั้งชุด MW flow gate ที่ project/scripts/mw หรือ ~/.hermes/mw ก่อนเขียนไฟล์"
        )
    try:
        proc = subprocess.run(
            [sys.executable, str(gate), "guard-write", str(absolute)],
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "flow_gate.py ใช้เวลาเกิน 15 วินาที จึงบล็อกการเขียนเพื่อความปลอดภัย"
    except OSError as exc:
        return f"เรียก flow_gate.py ไม่สำเร็จ จึงบล็อกการเขียน: {exc}"
    if proc.returncode == 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        return None
    detail = (proc.stderr or proc.stdout or f"guard-write exit={proc.returncode}").strip()
    return f"{detail}\n{_advice(detail)}"


def run(payload: Dict[str, object]) -> int:
    tool_name = payload.get("tool_name")
    if tool_name not in WRITE_TOOLS | {"Bash"}:
        return 0
    raw_input = payload.get("tool_input")
    tool_input = raw_input if isinstance(raw_input, dict) else {}
    raw_cwd = payload.get("cwd")
    cwd = _absolute(Path(str(raw_cwd)) if raw_cwd else Path.cwd(), Path.cwd())
    cwd_root = find_project_root(cwd)

    targets: Iterable[Path]
    if tool_name in WRITE_TOOLS:
        file_path = tool_input.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            if cwd_root is not None:
                return _block("คำสั่งเขียนไฟล์ไม่มี tool_input.file_path จึงตรวจลำดับงานไม่ได้")
            return 0
        targets = (Path(file_path),)
    else:
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            return 0
        extracted, has_write, complex_write = extract_bash_targets(command)
        if not has_write:
            return 0
        if cwd_root is not None and (complex_write or not extracted):
            return _block(
                "คำสั่ง shell เขียนไฟล์ในโปรเจกต์ MW ต้องแตกเป็นคำสั่งเดียวชัดๆ"
            )
        targets = (Path(value) for value in extracted)

    blockers = []
    for target in targets:
        blocker = _guard_target(target, cwd)
        if blocker:
            blockers.append(blocker)
    if blockers:
        return _block("\n".join(blockers))
    return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeError) as exc:
        return _block(f"อ่านข้อมูล PreToolUse ไม่ได้: {exc}")
    if not isinstance(payload, dict):
        return _block("ข้อมูล PreToolUse ต้องเป็น JSON object")
    return run(payload)


if __name__ == "__main__":
    raise SystemExit(main())

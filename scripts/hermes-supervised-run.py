#!/usr/bin/env python3
"""Run finite long-running commands with Hermes-friendly artifacts.

Writes a redacted combined log, status.json during execution, and manifest.json
on completion. Intended for local finite jobs that need resumable evidence.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

SECRET_PATTERNS = [
    re.compile(r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*)\s*=\s*([^\s]+)"),
    re.compile(r"(?i)\b(token)\s*=\s*([^\s]+)"),
    re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)([^\s]+)"),
    re.compile(r"\b(xox[baprs]-)[A-Za-z0-9-]+"),
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{8,})"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{12,})"),
    re.compile(r"\b(shpat_[A-Za-z0-9_]{8,})"),
]

MARKER_RE = re.compile(r"^\s*(STAGE|BLOCKED|NEEDS_USER_DECISION|AUTH_FAILED|FATAL)\s*:\s*(.+?)\s*$", re.IGNORECASE)
BLOCKER_MARKERS = {"BLOCKED", "NEEDS_USER_DECISION", "AUTH_FAILED", "FATAL"}
TAIL_CHARS = 4000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_text(text: str) -> str:
    """Redact obvious secret-bearing command/output fragments."""
    redacted = text
    for pattern in SECRET_PATTERNS:
        def repl(match: re.Match[str]) -> str:
            if match.lastindex and match.lastindex >= 2:
                return f"{match.group(1)}[REDACTED]"
            if match.lastindex == 1:
                prefix = match.group(1)
                if prefix.startswith(("xox", "gh", "sk-", "shpat_")):
                    return "[REDACTED]"
                return f"{prefix}[REDACTED]"
            return "[REDACTED]"

        redacted = pattern.sub(repl, redacted)
    return redacted


@dataclass
class RunState:
    name: str
    started_at: str
    status: str = "running"
    updated_at: str = ""
    exit_code: Optional[int] = None
    current_stage: str = ""
    last_output_tail: str = ""
    artifacts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at or utc_now(),
            "exit_code": self.exit_code,
            "current_stage": self.current_stage,
            "last_output_tail": self.last_output_tail,
            "artifacts": self.artifacts,
            "blockers": self.blockers,
        }


def atomic_json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def update_markers(text: str, state: RunState) -> None:
    for line in text.splitlines():
        match = MARKER_RE.match(line)
        if not match:
            continue
        marker = match.group(1).upper()
        value = match.group(2).strip()
        if marker == "STAGE":
            state.current_stage = value
        elif marker in BLOCKER_MARKERS and value not in state.blockers:
            state.blockers.append(value)


def snapshot_files(workdir: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not workdir.exists():
        return snapshot
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(workdir))
            stat = path.stat()
        except OSError:
            continue
        snapshot[rel] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def changed_files(workdir: Path, before: dict[str, tuple[int, int]], exclude: Iterable[Path]) -> tuple[list[str], dict[str, str]]:
    excluded = {p.resolve() for p in exclude}
    files: list[str] = []
    hashes: dict[str, str] = {}
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
            if resolved in excluded:
                continue
            rel = str(path.relative_to(workdir))
            stat = path.stat()
        except OSError:
            continue
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        if rel not in before or before[rel] != fingerprint:
            files.append(rel)
            try:
                hashes[rel] = sha256_file(path)
            except OSError:
                pass
    files.sort()
    return files, hashes


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command with durable Hermes status/log/manifest artifacts.")
    parser.add_argument("--name", required=True, help="Human-readable run name")
    parser.add_argument("--workdir", required=True, help="Directory to run in and scan for artifacts")
    parser.add_argument("--log", default="run.log", help="Log file path, relative to workdir unless absolute")
    parser.add_argument("--status", default="status.json", help="Status JSON path, relative to workdir unless absolute")
    parser.add_argument("--manifest", default="manifest.json", help="Manifest JSON path, relative to workdir unless absolute")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def resolve_artifact_path(workdir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workdir / path


def run(argv: list[str]) -> int:
    args = parse_args(argv)
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    log_path = resolve_artifact_path(workdir, args.log)
    status_path = resolve_artifact_path(workdir, args.status)
    manifest_path = resolve_artifact_path(workdir, args.manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    safe_command = redact_text(" ".join(subprocess.list2cmdline([part]) for part in args.command))
    state = RunState(name=args.name, started_at=started_at, updated_at=started_at)
    atomic_json_write(status_path, state.to_json())

    before = snapshot_files(workdir)
    exit_code = 1
    tail = ""

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"# hermes-supervised-run {args.name}\n")
        log.write(f"# started_at={started_at}\n")
        log.write(f"# command={safe_command}\n")
        process = subprocess.Popen(
            args.command,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = redact_text(raw_line)
            log.write(line)
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except BrokenPipeError:
                pass
            tail = (tail + line)[-TAIL_CHARS:]
            state.last_output_tail = tail
            update_markers(line, state)
            state.updated_at = utc_now()
            atomic_json_write(status_path, state.to_json())
        exit_code = process.wait()

    files, hashes = changed_files(workdir, before, exclude=[log_path, status_path, manifest_path])
    completed_at = utc_now()
    state.exit_code = exit_code
    state.artifacts = files
    if state.blockers:
        state.status = "blocked"
    else:
        state.status = "completed" if exit_code == 0 else "failed"
    state.updated_at = completed_at
    atomic_json_write(status_path, state.to_json())

    manifest = {
        "run_name": args.name,
        "command": safe_command,
        "workdir": str(workdir),
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": exit_code,
        "files_created_or_updated": files,
        "sha256": hashes,
        "notes": [
            "Command and output are redacted for obvious secret-like values before saving.",
            f"Status file: {status_path}",
            f"Log file: {log_path}",
        ],
    }
    atomic_json_write(manifest_path, manifest)
    return exit_code


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()

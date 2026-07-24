#!/usr/bin/env python3
"""Guarded Codex read-only review runner.

This wrapper is intentionally small and dependency-free. It gives Hermes a
single safe entry point for Codex review runs:

- always launches Codex with structured-output flags;
- writes raw stdout/stderr to a log file for manual audit;
- emits only bounded JSON to stdout;
- terminates the process when output starts flooding source/diff text;
- marks reviews without a final message file as unusable instead of passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["passed", "failed", "unusable"]},
        "summary": {"type": "string"},
        "must_fix": {"type": "array", "items": {"type": "string"}},
        "suggested_fixes": {"type": "array", "items": {"type": "string"}},
        "verification_commands": {"type": "array", "items": {"type": "string"}},
        "final_judgment": {"type": "string"},
    },
    "required": [
        "verdict",
        "summary",
        "must_fix",
        "suggested_fixes",
        "verification_commands",
        "final_judgment",
    ],
}

SOURCE_LINE_RE = re.compile(
    r"^\s*(from\s+\S+\s+import\s+|import\s+\S+|class\s+\w+|def\s+\w+|"
    r"async\s+def\s+\w+|(export\s+)?(async\s+)?function\s+\w+|"
    r"(const|let|var)\s+\w+\s*=|return\s+|if\s+|elif\s+|else:|for\s+|"
    r"while\s+|try:|except\s+|<div|<span|<template|<script|</|public\s+|private\s+)"
)
DIFF_LINE_RE = re.compile(r"^(diff --git |@@|--- a/|\+\+\+ b/|[+-][^+-])")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JSON_FLOOD_FIELD_NAMES = {"aggregated_output", "output", "content", "text"}


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _is_source_like(line: str) -> bool:
    stripped = _strip_ansi(line).strip()
    return bool(stripped and (SOURCE_LINE_RE.match(stripped) or stripped in {"{", "}", "};"}))


def _is_diff_like(line: str) -> bool:
    return bool(DIFF_LINE_RE.match(_strip_ansi(line).rstrip("\r")))


def _iter_json_text_fields(
    value: Any,
    *,
    active_field: str | None = None,
    path: str = "",
    depth: int = 0,
) -> list[tuple[str, str, str]]:
    """Return text leaves under flood-prone JSONL fields."""
    if depth > 12:
        return []
    if isinstance(value, str):
        if active_field:
            return [(active_field, path or active_field, value)]
        return []
    if isinstance(value, list):
        fields: list[tuple[str, str, str]] = []
        for index, item in enumerate(value[:100]):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            fields.extend(
                _iter_json_text_fields(item, active_field=active_field, path=item_path, depth=depth + 1)
            )
        return fields
    if isinstance(value, dict):
        fields = []
        for key, item in list(value.items())[:100]:
            key_text = str(key)
            child_field = key_text if key_text in JSON_FLOOD_FIELD_NAMES else active_field
            child_path = f"{path}.{key_text}" if path else key_text
            fields.extend(
                _iter_json_text_fields(item, active_field=child_field, path=child_path, depth=depth + 1)
            )
        return fields
    return []


def _json_field_flood(line: str, *, source_line_threshold: int, diff_line_threshold: int, char_threshold: int) -> dict[str, Any] | None:
    """Detect source/diff floods embedded inside one Codex JSONL field."""
    stripped = _strip_ansi(line).strip()
    if not stripped or not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    for field, path, text in _iter_json_text_fields(data):
        source_like = 0
        diff_like = 0
        for field_line in text.splitlines():
            if _is_source_like(field_line):
                source_like += 1
            if _is_diff_like(field_line):
                diff_like += 1
        chars = len(text)
        over_char_limit = chars >= char_threshold
        over_source_limit = source_like >= source_line_threshold
        over_diff_limit = diff_like >= diff_line_threshold
        if over_char_limit or over_source_limit or over_diff_limit:
            return {
                "reason": "aggregated_output_flood" if field == "aggregated_output" else "json_field_flood",
                "json_flood_field": field,
                "json_flood_path": path,
                "json_flood_chars": chars,
                "json_flood_source_like_lines": source_like,
                "json_flood_diff_like_lines": diff_like,
                "json_flood_limit": (
                    "char_threshold"
                    if over_char_limit
                    else "source_line_threshold"
                    if over_source_limit
                    else "diff_line_threshold"
                ),
            }
    return None


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Return a small JSON-serializable preview without raw flood content."""
    if depth > 4:
        return "[truncated]"
    if isinstance(value, str):
        return value if len(value) <= 1000 else value[:1000] + "...[truncated]"
    if isinstance(value, list):
        items = [_bounded(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            items.append(f"...[{len(value) - 20} more items truncated]")
        return items
    if isinstance(value, dict):
        return {str(k): _bounded(v, depth=depth + 1) for k, v in list(value.items())[:30]}
    return value


def _load_final_review(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists() or path.stat().st_size == 0:
        return None, "final_file_missing"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None, "final_file_empty"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": _bounded(text), "raw_text_preview": _bounded(text)}, "final_file_not_json"
    if not isinstance(data, dict):
        return {"summary": _bounded(data)}, "final_file_not_object"
    return _bounded(data), None


def _looks_like_review(value: Any) -> bool:
    return isinstance(value, dict) and bool(
        value.get("verdict")
        or value.get("final_judgment")
        or value.get("must_fix")
        or value.get("blockers")
    )


def _review_from_value(value: Any, *, depth: int = 0) -> dict[str, Any] | None:
    if depth > 8:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("{"):
            return None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return _review_from_value(decoded, depth=depth + 1)
    if isinstance(value, dict):
        if _looks_like_review(value):
            return _bounded(value)
        for key in ("text", "message", "content", "final", "review"):
            if key in value:
                review = _review_from_value(value[key], depth=depth + 1)
                if review is not None:
                    return review
        return None
    if isinstance(value, list):
        for item in reversed(value):
            review = _review_from_value(item, depth=depth + 1)
            if review is not None:
                return review
    return None


def _recover_review_from_raw_log(path: Path, *, max_bytes: int | None = None) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    latest_review: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            review = _review_from_json_line(line)
            if review is not None:
                latest_review = review
    return latest_review


def _review_from_json_line(line: str) -> dict[str, Any] | None:
    stripped = _strip_ansi(line).strip()
    if not stripped or not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return _review_from_value(data)


def _packet_review_prompt(prompt: str, packet: str) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "Review only the bounded packet below. Do not run shell commands. "
        "Do not inspect files directly. Do not request full source or full diffs.\n\n"
        "<bounded_review_packet>\n"
        f"{packet.rstrip()}\n"
        "</bounded_review_packet>"
    )


def _review_schema_error(review: dict[str, Any]) -> str | None:
    verdict = review.get("verdict") or review.get("status")
    if not isinstance(verdict, str) or not verdict.strip():
        return "missing_verdict"
    if verdict.strip().lower() not in {"passed", "pass", "ok", "can_continue", "可以继续", "failed", "fail", "blocked", "needs_fix", "需要先修", "unusable"}:
        return "unknown_verdict"
    for key in ("summary", "final_judgment"):
        if not isinstance(review.get(key), str):
            return f"missing_{key}"
    for key in ("must_fix", "suggested_fixes", "verification_commands"):
        if not isinstance(review.get(key), list):
            return f"missing_{key}"
        if any(not isinstance(item, str) for item in review[key]):
            return f"invalid_{key}"
    return None


def _status_from_review(review: dict[str, Any]) -> str:
    if _review_schema_error(review):
        return "unusable"
    verdict = str(review.get("verdict") or review.get("status") or "").strip().lower()
    if verdict in {"passed", "pass", "ok", "can_continue", "可以继续"}:
        return "passed"
    if verdict in {"failed", "fail", "blocked", "needs_fix", "需要先修"}:
        return "failed"
    must_fix = review.get("must_fix") or review.get("blockers") or []
    if isinstance(must_fix, list) and must_fix:
        return "failed"
    return "passed"


def _terminate(proc: subprocess.Popen[str], grace_seconds: float) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Codex review behind output flood guards.")
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex-yuna"))
    parser.add_argument("--workdir", default=".")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file")
    parser.add_argument("--schema-file")
    parser.add_argument("--final-file")
    parser.add_argument("--raw-log")
    parser.add_argument("--review-packet-file")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=2.0)
    parser.add_argument("--max-stdout-chars", type=int, default=200_000)
    parser.add_argument("--max-stdout-lines", type=int, default=4_000)
    parser.add_argument("--source-line-threshold", type=int, default=250)
    parser.add_argument("--diff-line-threshold", type=int, default=300)
    parser.add_argument("--json-field-char-threshold", type=int, default=40_000)
    return parser


def _emit(result: dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    status = result.get("status")
    if status == "passed":
        return 0
    if status == "failed":
        return 1
    return 2


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        return _emit({"status": "unusable", "reason": "workdir_not_found", "workdir": str(workdir)})

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8", errors="replace")
    if args.review_packet_file:
        packet_path = Path(args.review_packet_file).resolve()
        if not packet_path.exists():
            return _emit({
                "status": "unusable",
                "reason": "review_packet_file_missing",
                "review_packet_file": str(packet_path),
            })
        packet = packet_path.read_text(encoding="utf-8", errors="replace")
        prompt = _packet_review_prompt(prompt or "", packet)

    guard_dir = Path(tempfile.mkdtemp(prefix="codex-review-guard-"))
    schema_path = Path(args.schema_file).resolve() if args.schema_file else guard_dir / "schema.json"
    final_path = Path(args.final_file).resolve() if args.final_file else guard_dir / "final.json"
    raw_log_path = Path(args.raw_log).resolve() if args.raw_log else guard_dir / "raw.log"
    codex_cwd = workdir
    if args.review_packet_file:
        codex_cwd = guard_dir / "packet-only-workdir"
        codex_cwd.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.schema_file:
        schema_path.write_text(json.dumps(DEFAULT_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")

    # Use generic read-only exec instead of the `exec review` subcommand.
    # Current Codex CLI review subcommand rejects `--color` and also rejects
    # `--uncommitted` when a custom prompt is present. Generic read-only exec
    # supports the full guarded flag set and lets us pass a scoped prompt.
    cmd = [
        args.codex_bin,
        "exec",
    ]
    if args.review_packet_file:
        cmd.append("--skip-git-repo-check")
    cmd.extend([
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(final_path),
        "--color",
        "never",
        prompt or "",
    ])

    start = time.monotonic()
    stdout_chars = 0
    stdout_lines = 0
    source_lines = 0
    diff_lines = 0
    terminated = False
    process_exited_before_guard = False
    reason = ""
    raw_tail = ""
    json_flood: dict[str, Any] | None = None
    streamed_review: dict[str, Any] | None = None

    def _process_output_line(line: str) -> None:
        nonlocal source_lines, diff_lines, json_flood, streamed_review
        review = _review_from_json_line(line)
        if review is not None:
            streamed_review = review
        if _is_source_like(line):
            source_lines += 1
        if _is_diff_like(line):
            diff_lines += 1
        if json_flood is None:
            json_flood = _json_field_flood(
                line,
                source_line_threshold=args.source_line_threshold,
                diff_line_threshold=args.diff_line_threshold,
                char_threshold=args.json_field_char_threshold,
            )

    def _current_flood_reason() -> str | None:
        if json_flood:
            return str(json_flood["reason"])
        if stdout_chars > args.max_stdout_chars or stdout_lines > args.max_stdout_lines:
            return "stdout_limit_exceeded"
        if source_lines >= args.source_line_threshold:
            return "source_flood"
        if diff_lines >= args.diff_line_threshold:
            return "diff_flood"
        return None

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(codex_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError:
        return _emit({
            "status": "unusable",
            "reason": "codex_bin_not_found",
            "codex_bin": args.codex_bin,
            "raw_log_path": str(raw_log_path),
        })

    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    partial = ""

    with raw_log_path.open("w", encoding="utf-8", errors="replace") as raw_log:
        while True:
            if args.timeout_seconds > 0 and time.monotonic() - start > args.timeout_seconds:
                terminated = True
                reason = "timeout"
                _terminate(proc, args.kill_grace_seconds)
                break

            events = selector.select(timeout=0.05)
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                text = chunk.decode("utf-8", errors="replace")
                raw_log.write(text)
                raw_log.flush()
                raw_tail = (raw_tail + text)[-4000:]
                stdout_chars += len(text)
                stdout_lines += text.count("\n")

                combined = partial + text
                lines = combined.split("\n")
                partial = lines[-1]
                for line in lines[:-1]:
                    _process_output_line(line)

                current_reason = _current_flood_reason()
                if current_reason:
                    terminated = True
                    reason = current_reason

                if terminated:
                    _terminate(proc, args.kill_grace_seconds)
                    break

            if terminated:
                break
            if proc.poll() is not None:
                # Drain any buffered tail once after exit.
                while True:
                    try:
                        chunk = os.read(proc.stdout.fileno(), 8192)
                    except Exception:
                        chunk = b""
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    raw_log.write(text)
                    raw_log.flush()
                    raw_tail = (raw_tail + text)[-4000:]
                    stdout_chars += len(text)
                    stdout_lines += text.count("\n")
                    combined = partial + text
                    lines = combined.split("\n")
                    partial = lines[-1]
                    for line in lines[:-1]:
                        _process_output_line(line)
                break

    if not terminated:
        if partial:
            _process_output_line(partial)
            partial = ""
        current_reason = _current_flood_reason()
        if current_reason:
            terminated = True
            process_exited_before_guard = proc.poll() is not None
            reason = current_reason

    if terminated:
        recovered_review = streamed_review or _recover_review_from_raw_log(raw_log_path)
        if recovered_review is not None:
            recovered_schema_error = _review_schema_error(recovered_review)
            recovered_status = "unusable" if recovered_schema_error else _status_from_review(recovered_review)
            if recovered_status == "passed":
                recovered_status = "unusable"
            return _emit({
                "status": recovered_status,
                "reason": (
                    f"recovered_review_schema_invalid_after_{reason}"
                    if recovered_schema_error
                    else f"recovered_after_{reason}"
                ),
                "review_schema_error": recovered_schema_error,
                "flood_reason": reason,
                "review_recovered_from_flood": True,
                "terminated_by_guard": not process_exited_before_guard,
                "process_exited_before_guard": process_exited_before_guard,
                "source_flood_detected": reason == "source_flood" or source_lines >= args.source_line_threshold,
                "diff_flood_detected": reason == "diff_flood" or diff_lines >= args.diff_line_threshold,
                "json_field_flood_detected": json_flood is not None,
                "json_flood_field": (json_flood or {}).get("json_flood_field"),
                "json_flood_path": (json_flood or {}).get("json_flood_path"),
                "json_flood_chars": (json_flood or {}).get("json_flood_chars", 0),
                "json_flood_source_like_lines": (json_flood or {}).get("json_flood_source_like_lines", 0),
                "json_flood_diff_like_lines": (json_flood or {}).get("json_flood_diff_like_lines", 0),
                "json_flood_limit": (json_flood or {}).get("json_flood_limit"),
                "stdout_chars": stdout_chars,
                "stdout_lines": stdout_lines,
                "source_like_lines": source_lines,
                "diff_like_lines": diff_lines,
                "raw_log_path": str(raw_log_path),
                "final_file": str(final_path),
                "review": recovered_review,
            })
        return _emit({
            "status": "unusable",
            "reason": reason,
            "terminated_by_guard": not process_exited_before_guard,
            "process_exited_before_guard": process_exited_before_guard,
            "source_flood_detected": reason == "source_flood" or source_lines >= args.source_line_threshold,
            "diff_flood_detected": reason == "diff_flood" or diff_lines >= args.diff_line_threshold,
            "json_field_flood_detected": json_flood is not None,
            "json_flood_field": (json_flood or {}).get("json_flood_field"),
            "json_flood_path": (json_flood or {}).get("json_flood_path"),
            "json_flood_chars": (json_flood or {}).get("json_flood_chars", 0),
            "json_flood_source_like_lines": (json_flood or {}).get("json_flood_source_like_lines", 0),
            "json_flood_diff_like_lines": (json_flood or {}).get("json_flood_diff_like_lines", 0),
            "json_flood_limit": (json_flood or {}).get("json_flood_limit"),
            "stdout_chars": stdout_chars,
            "stdout_lines": stdout_lines,
            "source_like_lines": source_lines,
            "diff_like_lines": diff_lines,
            "raw_log_path": str(raw_log_path),
            "final_file": str(final_path),
        })

    exit_code = proc.wait(timeout=1)
    review, final_error = _load_final_review(final_path)
    if final_error:
        return _emit({
            "status": "unusable",
            "reason": final_error,
            "codex_exit_code": exit_code,
            "terminated_by_guard": False,
            "source_flood_detected": source_lines >= args.source_line_threshold,
            "diff_flood_detected": diff_lines >= args.diff_line_threshold,
            "json_field_flood_detected": json_flood is not None,
            "json_flood_field": (json_flood or {}).get("json_flood_field"),
            "json_flood_path": (json_flood or {}).get("json_flood_path"),
            "json_flood_chars": (json_flood or {}).get("json_flood_chars", 0),
            "json_flood_source_like_lines": (json_flood or {}).get("json_flood_source_like_lines", 0),
            "json_flood_diff_like_lines": (json_flood or {}).get("json_flood_diff_like_lines", 0),
            "json_flood_limit": (json_flood or {}).get("json_flood_limit"),
            "stdout_chars": stdout_chars,
            "stdout_lines": stdout_lines,
            "source_like_lines": source_lines,
            "diff_like_lines": diff_lines,
            "raw_log_path": str(raw_log_path),
            "final_file": str(final_path),
        })

    assert review is not None
    review_schema_error = _review_schema_error(review)
    status = "unusable" if review_schema_error else _status_from_review(review)
    if review_schema_error:
        final_reason = "review_schema_invalid"
    elif exit_code != 0 and status == "passed":
        status = "unusable"
        final_reason = "codex_exit_nonzero"
    else:
        final_reason = "ok"

    return _emit({
        "status": status,
        "reason": final_reason,
        "review_schema_error": review_schema_error,
        "codex_exit_code": exit_code,
        "terminated_by_guard": False,
        "source_flood_detected": source_lines >= args.source_line_threshold,
        "diff_flood_detected": diff_lines >= args.diff_line_threshold,
        "json_field_flood_detected": json_flood is not None,
        "json_flood_field": (json_flood or {}).get("json_flood_field"),
        "json_flood_path": (json_flood or {}).get("json_flood_path"),
        "json_flood_chars": (json_flood or {}).get("json_flood_chars", 0),
        "json_flood_source_like_lines": (json_flood or {}).get("json_flood_source_like_lines", 0),
        "json_flood_diff_like_lines": (json_flood or {}).get("json_flood_diff_like_lines", 0),
        "json_flood_limit": (json_flood or {}).get("json_flood_limit"),
        "stdout_chars": stdout_chars,
        "stdout_lines": stdout_lines,
        "source_like_lines": source_lines,
        "diff_like_lines": diff_lines,
        "raw_log_path": str(raw_log_path),
        "final_file": str(final_path),
        "schema_file": str(schema_path),
        "review": review,
    })


if __name__ == "__main__":
    raise SystemExit(run())

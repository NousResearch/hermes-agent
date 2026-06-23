"""Local JSONL sink for Hermes execution receipts.

The plugin is deliberately passive: it listens for already-redacted
``execution_receipt`` observer payloads and writes them to a local, profile-
scoped JSONL file. It never blocks or changes agent execution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import threading
from typing import Any

from hermes_constants import get_hermes_home

SCHEMA_VERSION = "hermes.execution_receipt.v0"
DEFAULT_RELATIVE_PATH = Path("execution-receipts") / "receipts.jsonl"
_MAX_RECEIPT_BYTES = 64 * 1024
_SAFE_PAYLOAD_METADATA_KEYS = {
    "redacted",
    "kind",
    "size_bytes",
    "field_names",
    "field_count",
    "item_count",
    "char_count",
    "unsafe_payload_dropped",
    "oversized_payload_dropped",
}
_WRITE_LOCK = threading.Lock()
_writer_errors = 0

_HELP_TEXT = """Usage: /receipts [status|tail [N]|gaps|help]

Inspect local Hermes execution receipts.

Commands:
  status     Summarize receipt counts, corrupt lines, sequence gaps, writer errors
  tail [N]   Show the latest N receipt summaries without args/results
  gaps       Show missing sequence numbers, corrupt lines, and evidence-gap codes
""".strip()


def _receipt_path() -> Path:
    return get_hermes_home() / DEFAULT_RELATIVE_PATH


def _ensure_storage(path: Path) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    if not path.exists():
        path.touch(mode=0o600, exist_ok=True)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _sanitize_payload_metadata(value: Any) -> dict[str, Any]:
    if not (isinstance(value, Mapping) and value.get("redacted") is True):
        return {"redacted": True, "unsafe_payload_dropped": True}

    safe: dict[str, Any] = {"redacted": True}
    for key in _SAFE_PAYLOAD_METADATA_KEYS:
        if key == "redacted" or key not in value:
            continue
        item = value.get(key)
        if key == "field_names" and isinstance(item, list):
            safe[key] = [str(name) for name in item[:50]]
        elif isinstance(item, (str, int, float, bool)) or item is None:
            safe[key] = item
    return safe


def _sanitize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(receipt)
    sanitized.setdefault("schema_version", SCHEMA_VERSION)
    sanitized.setdefault("receipt_type", "tool_complete")
    sanitized.setdefault("links", [])
    sanitized.setdefault("evidence_gaps", [])
    sanitized.setdefault("redaction_status", "ok")
    sanitized.setdefault("redaction_policy_version", "execution_receipts.v0")

    # The receipt builder emits metadata-only payload descriptors. Defense in
    # depth: whitelist descriptor keys so a buggy caller cannot smuggle raw
    # args/results in a mapping that merely sets ``redacted: true``.
    for field in ("args", "result"):
        before = sanitized.get(field)
        sanitized[field] = _sanitize_payload_metadata(before)
        if sanitized[field].get("unsafe_payload_dropped"):
            sanitized["redaction_status"] = "unsafe_payload_dropped"
    return sanitized


def _write_receipt(receipt: Mapping[str, Any]) -> bool:
    global _writer_errors
    try:
        if not isinstance(receipt, Mapping):
            raise TypeError("receipt must be a mapping")
        sanitized = _sanitize_receipt(receipt)
        encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8", errors="replace")) > _MAX_RECEIPT_BYTES:
            minimal = {
                "schema_version": SCHEMA_VERSION,
                "receipt_type": sanitized.get("receipt_type", "tool_complete"),
                "receipt_id": sanitized.get("receipt_id", ""),
                "trace_id": sanitized.get("trace_id", ""),
                "span_id": sanitized.get("span_id", ""),
                "sequence_number": sanitized.get("sequence_number", 0),
                "timestamp": sanitized.get("timestamp", ""),
                "session_id": sanitized.get("session_id", ""),
                "task_id": sanitized.get("task_id", ""),
                "tool_call_id": sanitized.get("tool_call_id", ""),
                "tool_name": sanitized.get("tool_name", ""),
                "status": sanitized.get("status", ""),
                "redaction_status": "oversized_minimal",
                "redaction_policy_version": "execution_receipts.v0",
                "args": {"redacted": True, "oversized_payload_dropped": True},
                "result": {"redacted": True, "oversized_payload_dropped": True},
                "links": [],
                "evidence_gaps": ["receipt_oversized"],
            }
            encoded = json.dumps(minimal, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        path = _receipt_path()
        with _WRITE_LOCK:
            _ensure_storage(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        return True
    except Exception:
        _writer_errors += 1
        return False


def _on_execution_receipt(*, receipt: Mapping[str, Any] | None = None, **_: Any) -> None:
    if not isinstance(receipt, Mapping):
        _write_receipt({} if receipt is None else receipt)  # type: ignore[arg-type]
        return
    _write_receipt(receipt)


def _read_receipt_file() -> tuple[list[dict[str, Any]], int]:
    path = _receipt_path()
    if not path.exists():
        return [], 0
    receipts: list[dict[str, Any]] = []
    corrupt = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    corrupt += 1
                    continue
                if isinstance(parsed, dict):
                    receipts.append(parsed)
                else:
                    corrupt += 1
    except OSError:
        return [], 1
    return receipts, corrupt


def _sequence_gaps(receipts: list[dict[str, Any]]) -> list[int]:
    seqs: list[int] = []
    for receipt in receipts:
        raw_sequence = receipt.get("sequence_number")
        if isinstance(raw_sequence, int):
            seqs.append(raw_sequence)
        elif isinstance(raw_sequence, str) and raw_sequence.isdigit():
            seqs.append(int(raw_sequence))
    seqs.sort()
    if len(seqs) < 2:
        return []
    missing: list[int] = []
    seen = set(seqs)
    for expected in range(seqs[0], seqs[-1] + 1):
        if expected not in seen:
            missing.append(expected)
    return missing


def _evidence_gap_counts(receipts: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for receipt in receipts:
        gaps = receipt.get("evidence_gaps") or []
        if isinstance(gaps, list):
            counts.update(str(gap) for gap in gaps if gap)
    return counts


def _format_status() -> str:
    receipts, corrupt = _read_receipt_file()
    gaps = _sequence_gaps(receipts)
    statuses = Counter(str(receipt.get("status", "unknown")) for receipt in receipts)
    status_bits = ", ".join(f"{key}:{value}" for key, value in sorted(statuses.items())) or "none"
    return (
        "Execution receipts: "
        f"total={len(receipts)} corrupt={corrupt} gaps={len(gaps)} "
        f"writer_errors={_writer_errors} statuses={status_bits} "
        f"path={_receipt_path()}"
    )


def _format_tail(limit: int = 10) -> str:
    receipts, corrupt = _read_receipt_file()
    if not receipts:
        return f"Execution receipts: no receipts recorded (corrupt={corrupt}, writer_errors={_writer_errors})."
    limit = max(1, min(int(limit), 100))
    lines = [f"Latest {min(limit, len(receipts))} execution receipts:"]
    for receipt in receipts[-limit:]:
        gaps = receipt.get("evidence_gaps") or []
        gap_text = ",".join(str(gap) for gap in gaps[:3]) if isinstance(gaps, list) and gaps else "-"
        lines.append(
            "# {seq} {tool} {status} session={session} call={call} duration_ms={duration} gaps={gaps}".format(
                seq=receipt.get("sequence_number", "?"),
                tool=receipt.get("tool_name", "?"),
                status=receipt.get("status", "?"),
                session=receipt.get("session_id", ""),
                call=receipt.get("tool_call_id", ""),
                duration=receipt.get("duration_ms", ""),
                gaps=gap_text,
            )
        )
    if corrupt:
        lines.append(f"corrupt_lines={corrupt}")
    return "\n".join(lines)


def _format_gaps() -> str:
    receipts, corrupt = _read_receipt_file()
    missing = _sequence_gaps(receipts)
    evidence_counts = _evidence_gap_counts(receipts)
    lines: list[str] = []
    if missing:
        lines.extend(f"missing sequence {seq}" for seq in missing[:100])
    else:
        lines.append("no sequence gaps detected")
    if corrupt:
        lines.append(f"corrupt lines: {corrupt}")
    if evidence_counts:
        lines.append("evidence gaps: " + ", ".join(f"{gap}:{count}" for gap, count in sorted(evidence_counts.items())))
    lines.append(f"writer_errors={_writer_errors}")
    return "\n".join(lines)


def _handle_slash(raw_args: str = "") -> str:
    argv = (raw_args or "").strip().split()
    command = argv[0].lower() if argv else "status"
    if command in {"help", "-h", "--help"}:
        return _HELP_TEXT
    if command == "status":
        return _format_status()
    if command == "tail":
        limit = 10
        if len(argv) > 1:
            try:
                limit = int(argv[1])
            except ValueError:
                return "Usage: /receipts tail [N]"
        return _format_tail(limit)
    if command == "gaps":
        return _format_gaps()
    return _HELP_TEXT


def register(ctx) -> None:
    ctx.register_hook("execution_receipt", _on_execution_receipt)
    ctx.register_command(
        "receipts",
        handler=_handle_slash,
        description="Inspect local Hermes execution receipt summaries.",
    )


__all__ = [
    "register",
]

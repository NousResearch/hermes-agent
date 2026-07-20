from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _read_events(ledger_file: Path, quarantine_dir: Path) -> tuple[List[Dict[str, Any]], int, int]:
    events: List[Dict[str, Any]] = []
    if not ledger_file.exists():
        return events, 0, 0

    raw = ledger_file.read_bytes()
    lines = raw.splitlines(keepends=True)
    quarantine_from: int | None = None

    for idx, raw_line in enumerate(lines):
        if not raw_line.endswith(b"\n"):
            quarantine_from = idx
            break
        try:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            events.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            quarantine_from = idx
            break

    if quarantine_from is None:
        return events, 0, 0

    _mkdir_private(quarantine_dir)
    suffix = b"".join(lines[quarantine_from:])
    quarantined = quarantine_dir / f"projection-corrupt-tail-{ledger_file.stem}-{int(time.time())}.jsonl"
    with quarantined.open("wb") as fh:
        fh.write(suffix)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.chmod(quarantined, 0o600)
    except OSError:
        pass

    return events, len(lines) - quarantine_from, 1


def _event_fact(event: Mapping[str, Any]) -> Mapping[str, Any]:
    fact = event.get("fact")
    if isinstance(fact, Mapping):
        return fact
    return event


def rebuild_current_view(root: Path) -> Dict[str, Any]:
    root = Path(root)
    ledger_dir = root / "ledger"
    views_dir = root / "views"
    errors_dir = root / "errors"
    _mkdir_private(views_dir)

    active_by_logical: Dict[str, Dict[str, Any]] = {}
    fact_to_logical: Dict[str, str] = {}
    applied = 0
    invalid_source_records = 0
    quarantined_files = 0

    for ledger_file in sorted(ledger_dir.glob("*.jsonl")):
        events, invalid_count, quarantined_count = _read_events(ledger_file, errors_dir)
        invalid_source_records += invalid_count
        quarantined_files += quarantined_count

        for event in events:
            applied += 1
            op = event.get("operation") or event.get("event")
            fact_id = str(event.get("fact_id", ""))
            fact = _event_fact(event)
            logical = f"{fact.get('scope', '')}|{fact.get('subject', '')}|{fact.get('key', '')}"

            if op in {"assert", "confirm", "supersede"}:
                if op == "supersede":
                    prev = str(event.get("supersedes", ""))
                    if prev:
                        old_key = fact_to_logical.pop(prev, None)
                        if old_key and old_key in active_by_logical:
                            active_by_logical.pop(old_key, None)
                active_by_logical[logical] = event
                if fact_id:
                    fact_to_logical[fact_id] = logical
            elif op == "retract":
                target_fact = str(event.get("supersedes") or event.get("retracts") or "")
                if target_fact:
                    old_key = fact_to_logical.pop(target_fact, None)
                    if old_key:
                        active_by_logical.pop(old_key, None)

    current_path = views_dir / "current.jsonl"
    fd, tmp_name = tempfile.mkstemp(prefix=".current-", suffix=".jsonl", dir=str(views_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for event in sorted(active_by_logical.values(), key=lambda e: (e.get("occurred_at", ""), e.get("event_id", ""))):
                fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, current_path)
        try:
            os.chmod(current_path, 0o600)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "applied": applied,
        "active": len(active_by_logical),
        "path": str(current_path),
        "invalid_source_records": invalid_source_records,
        "quarantined_files": quarantined_files,
    }

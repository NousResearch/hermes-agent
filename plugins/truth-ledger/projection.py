from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _read_events(ledger_file: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not ledger_file.exists():
        return events
    for line in ledger_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return events


def rebuild_current_view(root: Path) -> Dict[str, Any]:
    root = Path(root)
    ledger_dir = root / "ledger"
    views_dir = root / "views"
    _mkdir_private(views_dir)

    active_by_logical: Dict[str, Dict[str, Any]] = {}
    fact_to_logical: Dict[str, str] = {}
    applied = 0

    for ledger_file in sorted(ledger_dir.glob("*.jsonl")):
        for event in _read_events(ledger_file):
            applied += 1
            op = event.get("operation") or event.get("event")
            fact_id = str(event.get("fact_id", ""))
            logical = f"{event.get('scope', '')}|{event.get('subject', '')}|{event.get('key', '')}"

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

    return {"applied": applied, "active": len(active_by_logical), "path": str(current_path)}

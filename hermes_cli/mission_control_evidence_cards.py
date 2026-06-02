"""Local inert Evidence Card persistence for Mission Control.

Evidence Cards store caller-supplied evidence as bounded, redacted data only.
This module does not resolve paths, expand ``~``, stat files, inspect repos,
fetch URLs, execute commands, run tests, scan secrets, parse artifacts, or
integrate with runtime approval, command, gateway, goal, preflight, or
enforcement systems.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli.mission_control import redact_text, redact_value


EVIDENCE_CARD_SCHEMA = "mission-control.evidence-card.v1"
EVIDENCE_KINDS = {
    "repo_state",
    "diff_summary",
    "validation",
    "secret_scan",
    "safety_scan",
    "dirty_worktree",
    "file_locality",
    "capability_state",
    "reviewer_verdict",
    "limitation",
    "commit_report",
}
MAX_TEXT_CHARS = 100_000
MAX_LIST_ITEMS = 100
MAX_LIST_ITEM_CHARS = 4_000
MAX_STRUCTURED_KEYS = 100
MAX_STRUCTURED_DEPTH = 8
MAX_STRUCTURED_LIST_ITEMS = 100
_LOCK = threading.RLock()


class EvidenceCardError(ValueError):
    """Raised for invalid Evidence Card requests."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control" / "evidence-cards"


def audit_path() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control" / "evidence-cards-audit.jsonl"


def _new_card_id(created_at: str) -> str:
    stamp = re.sub(r"[^0-9TZ]", "", created_at.replace("+00:00", "Z"))
    return f"card_{stamp}_{secrets.token_hex(6)}"


def _card_path(card_id: str) -> Path:
    if not re.fullmatch(r"card_[0-9TZ]+_[a-f0-9]{12}", card_id):
        raise EvidenceCardError("Invalid evidence card id")
    return state_dir() / f"{card_id}.json"


def _bounded_text(value: Any, *, field: str, required: bool = False) -> str:
    if value is None:
        if required:
            raise EvidenceCardError(f"Missing required field: {field}")
        return ""
    text = str(value)
    if required and not text.strip():
        raise EvidenceCardError(f"Missing required field: {field}")
    return redact_text(text[:MAX_TEXT_CHARS])


def _string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvidenceCardError(f"{field} must be a list of strings")
    items: list[str] = []
    for item in value[:MAX_LIST_ITEMS]:
        if not isinstance(item, str):
            raise EvidenceCardError(f"{field} must be a list of strings")
        items.append(redact_text(item[:MAX_LIST_ITEM_CHARS]))
    return items


def _kind(value: Any) -> str:
    kind = str(value or "")
    if kind not in EVIDENCE_KINDS:
        raise EvidenceCardError("kind must be one of: " + ", ".join(sorted(EVIDENCE_KINDS)))
    return kind


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_STRUCTURED_DEPTH:
        return redact_text(str(value)[:MAX_TEXT_CHARS])
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_STRUCTURED_KEYS:
                break
            bounded_key = str(key)[:MAX_LIST_ITEM_CHARS]
            bounded[bounded_key] = _bounded_json_value(item, depth=depth + 1)
        return redact_value(bounded)
    if isinstance(value, list):
        return [
            _bounded_json_value(item, depth=depth + 1)
            for item in value[:MAX_STRUCTURED_LIST_ITEMS]
        ]
    if isinstance(value, tuple):
        return [
            _bounded_json_value(item, depth=depth + 1)
            for item in value[:MAX_STRUCTURED_LIST_ITEMS]
        ]
    if isinstance(value, str):
        return redact_text(value[:MAX_TEXT_CHARS])
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_text(str(value)[:MAX_TEXT_CHARS])


def _structured_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EvidenceCardError("structured_payload must be an object")
    bounded = _bounded_json_value(value)
    if not isinstance(bounded, dict):
        raise EvidenceCardError("structured_payload must be an object")
    return bounded


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(redact_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _append_audit(event: str, card: dict[str, Any] | None = None, *, result: str = "ok") -> None:
    record = {
        "timestamp": _now_iso(),
        "event": event,
        "actor": "dashboard",
        "surface": "dashboard",
        "card_id": (card or {}).get("id"),
        "kind": (card or {}).get("kind"),
        "title": (card or {}).get("title"),
        "summary": (card or {}).get("summary"),
        "trusted_for_execution": False,
        "inert_context_only": True,
        "authorizing": False,
        "result": redact_text(result),
    }
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact_value(record), sort_keys=True) + "\n")


def _summary(card: dict[str, Any]) -> dict[str, Any]:
    return redact_value(
        {
            "id": card["id"],
            "schema": card["schema"],
            "kind": card["kind"],
            "title": card["title"],
            "summary": card["summary"],
            "source": card.get("source", ""),
            "created_by": card.get("created_by", ""),
            "created_from": card.get("created_from", ""),
            "created_at": card["created_at"],
            "updated_at": card["updated_at"],
            "trusted_for_execution": False,
            "inert_context_only": True,
            "authorizing": False,
        }
    )


def _read_card_unlocked(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvidenceCardError("Evidence Card file is invalid")
    return data


def create_card(data: dict[str, Any]) -> dict[str, Any]:
    created_at = _now_iso()
    card = {
        "id": _new_card_id(created_at),
        "schema": EVIDENCE_CARD_SCHEMA,
        "kind": _kind(data.get("kind")),
        "title": _bounded_text(data.get("title"), field="title", required=True),
        "summary": _bounded_text(data.get("summary"), field="summary", required=True),
        "details": _bounded_text(data.get("details"), field="details"),
        "structured_payload": _structured_payload(data.get("structured_payload")),
        "limitations": _string_list(data.get("limitations"), field="limitations"),
        "redaction_notes": _string_list(data.get("redaction_notes"), field="redaction_notes"),
        "source": _bounded_text(data.get("source"), field="source"),
        "created_by": _bounded_text(data.get("created_by") or "dashboard", field="created_by"),
        "created_from": _bounded_text(data.get("created_from") or "dashboard_api", field="created_from"),
        "created_at": created_at,
        "updated_at": created_at,
        "trusted_for_execution": False,
        "inert_context_only": True,
        "authorizing": False,
    }
    with _LOCK:
        path = _card_path(card["id"])
        _atomic_write_json(path, card)
        _append_audit("evidence_card_created", card)
    return redact_value(card)


def list_cards() -> dict[str, Any]:
    with _LOCK:
        directory = state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        cards: list[dict[str, Any]] = []
        for path in sorted(directory.glob("card_*.json")):
            try:
                cards.append(_read_card_unlocked(path))
            except Exception:
                continue
    cards.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"items": [_summary(card) for card in cards], "warnings": []}


def get_card(card_id: str) -> dict[str, Any]:
    with _LOCK:
        path = _card_path(card_id)
        try:
            card = _read_card_unlocked(path)
        except FileNotFoundError:
            raise
    return {"card": redact_value(card)}

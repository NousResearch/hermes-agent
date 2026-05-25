"""Profile-local approval inbox storage for the Hermes dashboard.

This module intentionally records decisions only. It does not execute the
approved action; the dashboard can surface a generated Jenny command for a human
or agent to run in the normal chat/tool flow.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hermes_cli.config import get_hermes_home


ALLOWED_RISK_LABELS = {
    "Read-only",
    "Draft-only",
    "Local-build",
    "Private-send",
    "Live-service",
    "External-side-effect",
    "Money/customer",
    "Credential/auth",
    "Destructive",
    "Security boundary",
}

FINAL_STATUSES = {"approved", "rejected", "expired"}
DECISION_STATUSES = {"approved", "rejected", "clarification_requested", "snoozed"}


class ApprovalError(ValueError):
    """Raised when an approval request or transition is invalid."""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception as exc:
        raise ApprovalError(f"Invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _default_expiry(risk_label: str, created_at: datetime) -> datetime:
    if risk_label in {"Live-service", "Credential/auth", "Destructive", "Security boundary"}:
        return created_at + timedelta(hours=2)
    if risk_label in {"External-side-effect", "Money/customer"}:
        return created_at + timedelta(hours=24)
    return created_at + timedelta(days=7)


def _required_text(data: Dict[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ApprovalError(f"Missing required field: {key}")
    return value


def _optional_text(data: Dict[str, Any], key: str, default: str = "") -> str:
    return str(data.get(key) or default).strip()


def _string_list(data: Dict[str, Any], key: str) -> list[str]:
    raw = data.get(key)
    if raw is None or raw == "":
        return []
    if not isinstance(raw, list):
        raise ApprovalError(f"{key} must be a list of strings")
    return [str(item).strip() for item in raw if str(item).strip()]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class ApprovalStore:
    """JSON/JSONL approval inbox under the active HERMES_HOME."""

    def __init__(self, base_dir: Optional[Path] = None):
        root = Path(base_dir) if base_dir else Path(get_hermes_home()) / "state" / "ops-center"
        self.base_dir = root
        self.inbox_path = root / "approval-inbox.json"
        self.audit_path = root / "approval-audit.jsonl"

    def _load_items(self) -> list[Dict[str, Any]]:
        if not self.inbox_path.exists():
            return []
        try:
            data = json.loads(self.inbox_path.read_text() or "[]")
        except json.JSONDecodeError as exc:
            raise ApprovalError(f"Approval inbox is not valid JSON: {self.inbox_path}") from exc
        if not isinstance(data, list):
            raise ApprovalError("Approval inbox must contain a JSON list")
        return [dict(item) for item in data if isinstance(item, dict)]

    def _save_items(self, items: Iterable[Dict[str, Any]]) -> None:
        _atomic_write_json(self.inbox_path, list(items))

    def _append_audit(self, event: str, approval: Dict[str, Any], *, actor: Optional[str] = None, note: Optional[str] = None) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "approval_id": approval.get("id"),
            "status": approval.get("status"),
            "actor": actor,
            "note": note,
            "timestamp": _iso(_now_utc()),
        }
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _with_expiry(self, item: Dict[str, Any], now: Optional[datetime]) -> Dict[str, Any]:
        out = dict(item)
        if out.get("status") in FINAL_STATUSES:
            return out
        check_time = now or _now_utc()
        expires_at = out.get("expires_at")
        if expires_at and _parse_iso(str(expires_at)) <= check_time.astimezone(timezone.utc):
            out["status"] = "expired"
            out["expired_at"] = _iso(check_time)
        return out

    def list(self, *, now: Optional[datetime] = None) -> list[Dict[str, Any]]:
        items = [self._with_expiry(item, now) for item in self._load_items()]
        # Persist expiry transitions only when using real current time, not a test/read simulation.
        if now is None and any(item.get("status") == "expired" for item in items):
            self._save_items(items)
        return items

    def get(self, approval_id: str) -> Dict[str, Any]:
        for item in self.list():
            if item.get("id") == approval_id:
                return item
        raise ApprovalError("Approval not found")

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        created_at = _now_utc()
        risk_label = _required_text(data, "risk_label")
        if risk_label not in ALLOWED_RISK_LABELS:
            raise ApprovalError(f"Invalid risk_label: {risk_label}")
        approval = {
            "id": f"appr_{created_at.strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}",
            "created_at": _iso(created_at),
            "created_by": _required_text(data, "created_by"),
            "project": _required_text(data, "project"),
            "profile": str(data.get("profile") or "default").strip() or "default",
            "risk_label": risk_label,
            "title": _required_text(data, "title"),
            "proposed_action": _required_text(data, "proposed_action"),
            "target": _required_text(data, "target"),
            "preview": _required_text(data, "preview"),
            "reason": _required_text(data, "reason"),
            "rollback_or_verification": _required_text(data, "rollback_or_verification"),
            "blocked_until_approved": True,
            "status": "pending",
            "expires_at": str(data.get("expires_at") or _iso(_default_expiry(risk_label, created_at))),
            "decided_at": None,
            "decided_by": None,
            "decision_note": None,
            "execution_allowed": False,
            "execution_result": None,
            "generated_command": None,
            "proposal_kind": str(data.get("proposal_kind") or "manual").strip() or "manual",
            "source_surface": _optional_text(data, "source_surface"),
            "source_ref": _optional_text(data, "source_ref"),
            "conversation_excerpt": _optional_text(data, "conversation_excerpt"),
            "related_paths": _string_list(data, "related_paths"),
        }
        # Validate provided expiry if present.
        _parse_iso(approval["expires_at"])
        items = self._load_items()
        items.append(approval)
        self._save_items(items)
        self._append_audit("created", approval, actor=approval["created_by"])
        return approval

    def propose_from_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a gated-action approval from a chat/tool workflow.

        This is the ingestion path Jenny can use when normal work discovers a
        side-effect boundary. It records source context and still creates only a
        pending decision record; it never executes the proposed action.
        """
        payload = dict(data)
        payload["proposal_kind"] = "gated_action"
        proposal = self.create(payload)
        self._append_audit(
            "proposed_from_context",
            proposal,
            actor=proposal.get("created_by"),
            note=proposal.get("source_ref"),
        )
        return proposal

    def decide(self, approval_id: str, status: str, *, decided_by: str, decision_note: Optional[str] = None) -> Dict[str, Any]:
        if status not in DECISION_STATUSES:
            raise ApprovalError(f"Invalid decision status: {status}")
        actor = str(decided_by or "").strip()
        if not actor:
            raise ApprovalError("decided_by is required")
        items = self.list()
        for idx, item in enumerate(items):
            if item.get("id") != approval_id:
                continue
            current = str(item.get("status") or "")
            if current != "pending":
                raise ApprovalError(f"Cannot change approval in status '{current}'")
            updated = dict(item)
            updated["status"] = status
            updated["decided_at"] = _iso(_now_utc())
            updated["decided_by"] = actor
            updated["decision_note"] = decision_note
            updated["execution_allowed"] = False
            if status == "approved":
                updated["generated_command"] = self._generated_command(updated)
            items[idx] = updated
            self._save_items(items)
            self._append_audit(status, updated, actor=actor, note=decision_note)
            return updated
        raise ApprovalError("Approval not found")

    def _generated_command(self, approval: Dict[str, Any]) -> str:
        return (
            "Jenny, approved: "
            f"{approval['title']} for {approval['project']} targeting {approval['target']}. "
            f"Scope: {approval['proposed_action']}. "
            "Verify after completion: "
            f"{approval['rollback_or_verification']}. "
            "Do not perform any broader action."
        )


def _load_json_payload(path: Optional[str]) -> Dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text())
    raw = sys.stdin.read()
    if not raw.strip():
        raise ApprovalError("No proposal JSON provided on stdin or --json-file")
    return json.loads(raw)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes Ops approval ledger helper")
    sub = parser.add_subparsers(dest="command")

    propose = sub.add_parser("propose", help="Create a pending gated-action approval from JSON")
    propose.add_argument("--json-file", help="Path to a JSON proposal payload; defaults to stdin")
    propose.add_argument("--json", action="store_true", help="Print the created approval as JSON")

    args = parser.parse_args(argv)
    if args.command == "propose":
        try:
            created = ApprovalStore().propose_from_context(_load_json_payload(args.json_file))
        except Exception as exc:
            print(f"approval proposal failed: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(created, sort_keys=True))
        else:
            print(f"created approval {created['id']} ({created['status']})")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

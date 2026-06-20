"""AgentCyber break-glass approval records.

Break-glass approvals are scoped, expiring records that allow a single
high-impact AgentCyber tool call after explicit human approval.  They never
persist raw tool arguments; only a stable SHA-256 fingerprint and a redacted
preview are stored.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_replace

_APPROVAL_ARG_KEYS = frozenset({
    "agentcyber_breakglass_approval",
    "approval_token",
    "breakglass_approval_id",
})
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|authorization|bearer|credential|password|secret|token)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|[A-Za-z0-9_=-]{32,}|ghp_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]+)\b"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _without_approval_args(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _without_approval_args(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k) not in _APPROVAL_ARG_KEYS
        }
    if isinstance(value, list):
        return [_without_approval_args(v) for v in value]
    return value


def _redact_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_redact_value(key, v) for v in value]
    if _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("[REDACTED]", value)
    return value


def redacted_args(function_args: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = _without_approval_args(function_args or {})
    redacted = _redact_value("", cleaned)
    return redacted if isinstance(redacted, dict) else {}


def fingerprint_tool_call(tool_name: str, function_args: dict[str, Any] | None) -> str:
    payload = {
        "tool_name": str(tool_name or ""),
        "args": _without_approval_args(function_args or {}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BreakGlassApproval:
    approval_id: str
    created_at: str
    expires_at: str
    operator: str
    reason: str
    gate: str
    tool_name: str
    args_fingerprint: str
    asset_matches: tuple[str, ...]
    revoked: bool = False
    redacted_args: dict[str, Any] | None = None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        return _parse_time(self.expires_at) <= (now or utc_now())

    def to_json(self) -> str:
        data = asdict(self)
        data["asset_matches"] = list(self.asset_matches)
        return json.dumps(data, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BreakGlassApproval":
        return cls(
            approval_id=str(data.get("approval_id") or ""),
            created_at=str(data.get("created_at") or ""),
            expires_at=str(data.get("expires_at") or ""),
            operator=str(data.get("operator") or ""),
            reason=str(data.get("reason") or ""),
            gate=str(data.get("gate") or ""),
            tool_name=str(data.get("tool_name") or ""),
            args_fingerprint=str(data.get("args_fingerprint") or ""),
            asset_matches=tuple(str(x) for x in data.get("asset_matches") or ()),
            revoked=bool(data.get("revoked", False)),
            redacted_args=data.get("redacted_args") if isinstance(data.get("redacted_args"), dict) else None,
        )


def default_store_path() -> Path:
    return get_hermes_home() / "agentcyber" / "breakglass.jsonl"


class BreakGlassStore:
    def __init__(self, path: Path | None = None):
        self.path = path or default_store_path()

    def list(self) -> list[BreakGlassApproval]:
        if not self.path.exists():
            return []
        approvals: list[BreakGlassApproval] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            approvals.append(BreakGlassApproval.from_dict(json.loads(line)))
        return approvals

    def get(self, approval_id: str) -> BreakGlassApproval | None:
        for approval in reversed(self.list()):
            if approval.approval_id == approval_id:
                return approval
        return None

    def create(
        self,
        *,
        tool_name: str,
        function_args: dict[str, Any] | None,
        gate: str,
        asset_matches: tuple[str, ...],
        operator: str,
        reason: str,
        ttl_minutes: int = 15,
        now: datetime | None = None,
    ) -> BreakGlassApproval:
        now = now or utc_now()
        approval = BreakGlassApproval(
            approval_id=f"bg_{now.strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}",
            created_at=_iso(now),
            expires_at=_iso(now + timedelta(minutes=max(1, int(ttl_minutes)))),
            operator=operator,
            reason=reason,
            gate=gate,
            tool_name=tool_name,
            args_fingerprint=fingerprint_tool_call(tool_name, function_args),
            asset_matches=tuple(asset_matches),
            revoked=False,
            redacted_args=redacted_args(function_args),
        )
        self._append(approval)
        return approval

    def revoke(self, approval_id: str, *, now: datetime | None = None) -> BreakGlassApproval:
        existing = self.get(approval_id)
        if existing is None:
            raise KeyError(approval_id)
        revoked = BreakGlassApproval(
            approval_id=existing.approval_id,
            created_at=existing.created_at,
            expires_at=_iso(now or utc_now()),
            operator=existing.operator,
            reason=existing.reason,
            gate=existing.gate,
            tool_name=existing.tool_name,
            args_fingerprint=existing.args_fingerprint,
            asset_matches=existing.asset_matches,
            revoked=True,
            redacted_args=existing.redacted_args,
        )
        self._append(revoked)
        return revoked

    def _append(self, approval: BreakGlassApproval) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(existing + approval.to_json() + "\n", encoding="utf-8")
        atomic_replace(tmp, self.path)


@dataclass(frozen=True)
class ApprovalCheck:
    allowed: bool
    reason: str
    approval: BreakGlassApproval | None = None


def extract_approval_id(function_args: dict[str, Any] | None) -> str:
    args = function_args or {}
    for key in _APPROVAL_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def validate_approval(
    *,
    approval_id: str,
    tool_name: str,
    function_args: dict[str, Any] | None,
    gate: str,
    asset_matches: tuple[str, ...],
    store: BreakGlassStore | None = None,
    now: datetime | None = None,
) -> ApprovalCheck:
    if not approval_id:
        return ApprovalCheck(False, "missing break-glass approval")
    store = store or BreakGlassStore()
    approval = store.get(approval_id)
    if approval is None:
        return ApprovalCheck(False, "unknown break-glass approval")
    if approval.revoked:
        return ApprovalCheck(False, "break-glass approval is revoked", approval)
    if approval.is_expired(now=now):
        return ApprovalCheck(False, "break-glass approval is expired", approval)
    if approval.tool_name != tool_name:
        return ApprovalCheck(False, "break-glass approval tool mismatch", approval)
    if approval.gate != gate:
        return ApprovalCheck(False, "break-glass approval gate mismatch", approval)
    if approval.args_fingerprint != fingerprint_tool_call(tool_name, function_args):
        return ApprovalCheck(False, "break-glass approval action fingerprint mismatch", approval)
    current_assets = set(asset_matches)
    approved_assets = set(approval.asset_matches)
    if not approved_assets or not approved_assets.issubset(current_assets):
        return ApprovalCheck(False, "break-glass approval asset scope mismatch", approval)
    return ApprovalCheck(True, "break-glass approval accepted", approval)

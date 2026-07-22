"""MCP tool audit log — correlation_id, params_hash, latency, status."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_trader.audit.redact import hash_params, redact_for_log, redact_mapping
from hermes_trader.config import TRADER_HOME_SUBDIR


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_mcp_audit_path() -> Path:
    import os

    override = os.environ.get("HERMES_TRADER_MCP_AUDIT", "").strip()
    if override:
        return Path(override)
    return _hermes_home() / TRADER_HOME_SUBDIR / "mcp_audit.jsonl"


@dataclass
class McpAuditLog:
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = default_mcp_audit_path()

    def append(self, record: dict[str, Any]) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **record,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")

    def list_recent(self, *, limit: int = 100) -> list[dict[str, Any]]:
        assert self.path is not None
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def audit_mcp_call(
    *,
    server_name: str,
    tool_name: str,
    args: Mapping[str, Any] | None,
    result_status: str,
    latency_ms: float,
    correlation_id: Optional[str] = None,
    error_message: str = "",
    audit_log: Optional[McpAuditLog] = None,
    debug_redaction: bool = False,
) -> str:
    """Append one MCP audit row; returns correlation_id."""
    cid = correlation_id or uuid.uuid4().hex[:12]
    log = audit_log or McpAuditLog()
    log.append(
        {
            "correlation_id": cid,
            "server": server_name,
            "tool": tool_name,
            "params_hash": hash_params(args),
            "params_redacted": redact_mapping(args, debug=debug_redaction),
            "result_status": result_status,
            "latency_ms": round(latency_ms, 2),
            "error": redact_for_log(error_message, debug=debug_redaction) if error_message else "",
        }
    )
    return cid
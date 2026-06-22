"""SOC audit hook — always-on structured logging for cybersecurity operations.

Activated when the environment variable HERMES_CYBER_AUDIT=true is set.
Writes a newline-delimited JSON (NDJSON) audit log to:

    $HERMES_HOME/logs/cyber_audit.jsonl

Every agent:step and agent:end event is recorded with:
  - ISO-8601 timestamp (UTC)
  - session_id / session_key (gateway sessions) or task description
  - platform / user identity (gateway context only)
  - event_type
  - tool call name + truncated input (agent:step)
  - tool result summary (agent:step)
  - completion reason + token counts (agent:end)

The log is append-only and never truncated by this module.
Rotate externally with logrotate or equivalent.

Security notes:
  - Credential-like values in tool inputs are redacted (env vars that match
    _REDACT_RE are replaced with ***).
  - The log file is created with mode 0600 (owner-read-write only).
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Match common secret-bearing key names inside tool argument dicts. Live USB
# approval keys are exact after `-`/`_` normalization to avoid redacting benign
# metadata such as `approval_status` or larger names like `not_operator_approval`.
_REDACT_RE = re.compile(
    r"api[_-]?key|token|password|secret|credential|auth|bearer",
    re.IGNORECASE,
)
_APPROVAL_REDACT_KEYS = {"operator_approval", "approval_token", "live_usb_approval"}


def _should_redact_key(key: Any) -> bool:
    key_text = str(key)
    if _REDACT_RE.search(key_text):
        return True
    return key_text.lower().replace("-", "_") in _APPROVAL_REDACT_KEYS


def _redact_text(value: str) -> str:
    """Redact secret-bearing strings before they enter the cyber audit log."""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    except Exception:
        return value


def _is_enabled() -> bool:
    return os.environ.get("HERMES_CYBER_AUDIT", "").lower() in ("1", "true", "yes")


def _audit_log_path() -> Path:
    try:
        from hermes_cli.config import get_hermes_home
        logs_dir = get_hermes_home() / "logs"
    except Exception:
        logs_dir = Path.home() / ".hermes" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "cyber_audit.jsonl"
    # Ensure restrictive permissions on creation
    if not log_path.exists():
        try:
            log_path.touch(mode=0o600)
        except Exception:
            pass
    return log_path


def _redact(obj: Any, depth: int = 0) -> Any:
    if depth > 4:
        return obj
    if isinstance(obj, dict):
        return {
            k: "***" if _should_redact_key(k) else _redact(v, depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(item, depth + 1) for item in obj[:20]]
    if isinstance(obj, str):
        redacted = _redact_text(obj)
        if len(redacted) > 500:
            return redacted[:500] + "…"
        return redacted
    return obj


# ---------------------------------------------------------------------------
# HMAC-chain helpers — tamper-evident audit log
# ---------------------------------------------------------------------------

def _audit_key_path() -> Path:
    try:
        from hermes_cli.config import get_hermes_home
        return get_hermes_home() / "audit.key"
    except Exception:
        return Path.home() / ".hermes" / "audit.key"


def _load_or_create_audit_key() -> bytes:
    """Load or generate the per-install HMAC signing key (256-bit, hex-encoded)."""
    key_path = _audit_key_path()
    try:
        if key_path.exists():
            raw = key_path.read_text(encoding="utf-8").strip()
            if len(raw) == 64:  # 32 bytes hex
                return bytes.fromhex(raw)
        # Generate a fresh key and persist it with mode 0600
        key_bytes = secrets.token_bytes(32)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key_bytes.hex())
        return key_bytes
    except Exception as exc:
        logger.warning("cyber_audit: could not load/create audit.key: %s — HMAC disabled", exc)
        return b""


def _last_log_line(path: Path) -> bytes:
    """Return the raw bytes of the last non-empty line in the audit log."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return b""
            # Walk back to find the second-to-last newline (last complete line)
            pos = size - 1
            # Skip trailing newline of the last line
            if fh.read(1) == b"\n":
                pos -= 1
            fh.seek(pos)
            buf = b""
            while pos >= 0:
                fh.seek(pos)
                ch = fh.read(1)
                if ch == b"\n":
                    break
                buf = ch + buf
                pos -= 1
            return buf
    except Exception:
        return b""


def _hmac_of(data: bytes, key: bytes) -> str:
    """Return hex HMAC-SHA256 of data under key, or '' if key is empty."""
    if not key:
        return ""
    return _hmac.new(key, data, hashlib.sha256).hexdigest()


def _write(record: dict) -> None:
    try:
        path   = _audit_log_path()
        key    = _load_or_create_audit_key()
        # Chain: include HMAC of the previous line so tampering breaks the chain
        prev_line = _last_log_line(path)
        record["prev_hmac"] = _hmac_of(prev_line, key) if key else None
        # Sign this record itself (without the hmac field, to avoid circularity)
        body  = json.dumps(record, default=str, separators=(",", ":"), sort_keys=True)
        record["hmac"] = _hmac_of(body.encode(), key) if key else None
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:
        logger.warning("cyber_audit: failed to write audit log: %s", exc)


# ---------------------------------------------------------------------------
# Public hook entry point
# ---------------------------------------------------------------------------

async def handle(event_type: str, context: dict) -> None:
    if not _is_enabled():
        return

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record: dict[str, Any] = {
        "ts": ts,
        "event": event_type,
    }

    # Attach session identity when available (gateway context)
    for key in ("session_id", "session_key", "platform", "user_id", "user_name", "chat_id"):
        val = context.get(key)
        if val is not None:
            record[key] = str(val)

    cyber_route = context.get("cyber_route")
    if isinstance(cyber_route, dict):
        record["cyber_route"] = {
            key: cyber_route.get(key)
            for key in (
                "route",
                "provider_preference",
                "reason",
                "requires_hosted_secret_confirmation",
                "explicit_override",
            )
        }

    gate_metadata = context.get("agentcyber_gate") or context.get("cyber_execution_gate")
    if isinstance(gate_metadata, dict):
        record["agentcyber_gate"] = {
            key: gate_metadata.get(key)
            for key in (
                "gate",
                "allowed",
                "reason",
                "asset_matches",
                "candidates",
                "breakglass_approval_id",
            )
            if key in gate_metadata
        }

    if event_type == "agent:step":
        tool_call = context.get("tool_call") or {}
        tool_result = context.get("tool_result") or {}
        record["tool"] = tool_call.get("name") or tool_call.get("function", {}).get("name")
        raw_input = tool_call.get("input") or tool_call.get("function", {}).get("arguments") or {}
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except Exception:
                raw_input = {"_raw": raw_input[:300]}
        record["tool_input"]  = _redact(raw_input)
        # Truncate result to keep log manageable
        redacted_result = _redact(tool_result)
        result_str = _redact_text(str(redacted_result))
        record["tool_result_preview"] = result_str[:300] + ("…" if len(result_str) > 300 else "")

    elif event_type == "agent:end":
        record["stop_reason"]  = context.get("stop_reason")
        record["input_tokens"] = context.get("input_tokens")
        record["output_tokens"] = context.get("output_tokens")
        record["iterations"]   = context.get("iterations")

    elif event_type in ("session:start", "session:end", "gateway:startup"):
        # Minimal record — just timestamp + event type + identity already added
        pass

    _write(record)

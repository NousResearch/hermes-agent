"""Redaction applied to telemetry data on export.

Two independent controls:

  * Secrets are always redacted, on every export and in every mode; no setting
    disables this. Wraps ``agent/redact.py::redact_sensitive_text(force=True)``.

  * Whether message bodies, reasoning, and raw tool arguments are exportable at all is
    governed by the trajectories setting (``telemetry.trajectories.enabled``, default
    off, admin-pinnable), not by a redaction mode. With trajectories off, content is
    dropped. With it on, content is exportable and ``content_redaction`` (none|pii)
    controls how much is scrubbed; secrets are still always stripped.

This applies to the local and trajectory export paths. It is unrelated to any
aggregate-metrics path.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Content-redaction strengths for any content that IS exported.
CONTENT_NONE = "none"   # drop content entirely (structural telemetry only)
CONTENT_PII = "pii"     # codec-aware PII redaction on exported content
CONTENT_MODES = {CONTENT_NONE, CONTENT_PII}

# ── PII patterns (applied only in CONTENT_PII mode, on content that is exported) ──
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# E.164-ish and common separators; conservative to avoid nuking code/IDs.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\w)"
)
# Long opaque hex/uuid-ish user identifiers.
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def _secret_redact(text: Optional[str]) -> Optional[str]:
    """Always-on secret redaction. force=True so user config can't disable it."""
    if text is None:
        return None
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(str(text), force=True)
    except Exception:
        # Fail CLOSED: if the redactor can't run, do not emit the raw string.
        return "[redaction-unavailable]"


def _pii_redact(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _UUID_RE.sub("[id]", text)
    text = _PHONE_RE.sub("[phone]", text)
    return text


def redact_for_export(
    text: Optional[str],
    *,
    content_mode: str = CONTENT_NONE,
) -> Optional[str]:
    """Redact a single content string for export.

    Secrets are ALWAYS stripped. Then PII is stripped when content_mode is 'pii'.
    Callers gate *whether content is exported at all* via telemetry.trajectories
    (see ``content_export_enabled``); this function only scrubs content that the
    caller has already decided to export.
    """
    redacted = _secret_redact(text)
    if redacted is None:
        return None
    if content_mode == CONTENT_PII:
        redacted = _pii_redact(redacted)
    return redacted


def content_export_enabled(config: Optional[Dict[str, Any]]) -> bool:
    """True only when telemetry.trajectories is explicitly enabled.

    This is the consent gate for exporting message bodies / reasoning / raw tool
    args. Default off. Admin-pinnable via managed scope (telemetry.trajectories.enabled).
    """
    try:
        tel = (config or {}).get("telemetry") or {}
        traj = tel.get("trajectories") or {}
        return bool(traj.get("enabled", False))
    except Exception:
        return False


def content_mode_for(config: Optional[Dict[str, Any]]) -> str:
    try:
        tel = (config or {}).get("telemetry") or {}
        mode = tel.get("content_redaction", CONTENT_NONE)
        return mode if mode in CONTENT_MODES else CONTENT_NONE
    except Exception:
        return CONTENT_NONE


# ── Codec-aware message redaction (NeMo pattern) ─────────────────────────────
# Redact the right fields of a provider message shape rather than regex-blasting
# the whole blob. Structure (roles, names, counts) is preserved; only the
# free-text content fields are scrubbed.

def redact_message(
    msg: Dict[str, Any],
    *,
    content_mode: str = CONTENT_NONE,
    include_content: bool = False,
) -> Dict[str, Any]:
    """Redact one chat message dict for export.

    When include_content is False (trajectories off), content/reasoning/tool-arg
    fields are dropped — only structural fields (role, tool name, counts) remain.
    When True, those fields are kept but passed through redact_for_export.
    """
    role = msg.get("role")
    out: Dict[str, Any] = {"role": role}

    # Always-structural fields.
    if msg.get("tool_name") is not None:
        out["tool_name"] = msg.get("tool_name")
    if msg.get("name") is not None:
        out["name"] = msg.get("name")

    if not include_content:
        # Structural only: record presence/size, not bytes.
        c = msg.get("content")
        if c is not None:
            out["content_chars"] = len(str(c))
        if msg.get("reasoning_content"):
            out["reasoning_chars"] = len(str(msg["reasoning_content"]))
        if msg.get("tool_calls"):
            out["tool_call_count"] = _count_tool_calls(msg["tool_calls"])
        return out

    # Content included (trajectories enabled): scrub then keep.
    if msg.get("content") is not None:
        out["content"] = redact_for_export(msg["content"], content_mode=content_mode)
    if msg.get("reasoning_content"):
        out["reasoning_content"] = redact_for_export(
            msg["reasoning_content"], content_mode=content_mode
        )
    if msg.get("tool_calls"):
        out["tool_calls"] = _redact_tool_calls(msg["tool_calls"], content_mode=content_mode)
    return out


def _count_tool_calls(tool_calls: Any) -> int:
    try:
        import json
        tc = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
        return len(tc) if isinstance(tc, list) else (1 if tc else 0)
    except Exception:
        return 0


def _redact_tool_calls(tool_calls: Any, *, content_mode: str) -> Any:
    """Redact raw tool-call arguments (free text) while keeping function names."""
    import json
    try:
        tc = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
    except Exception:
        return "[unparseable-tool-calls]"
    if not isinstance(tc, list):
        return []
    out: List[Dict[str, Any]] = []
    for call in tc:
        if not isinstance(call, dict):
            continue
        fn = (call.get("function") or {}) if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") or call.get("name")
        args = fn.get("arguments")
        red_args = redact_for_export(args, content_mode=content_mode) if args is not None else None
        out.append({"name": name, "arguments": red_args})
    return out


__all__ = [
    "CONTENT_NONE",
    "CONTENT_PII",
    "CONTENT_MODES",
    "redact_for_export",
    "content_export_enabled",
    "content_mode_for",
    "redact_message",
]

"""Profile route advisory helpers for safe Hermes routing surfaces.

R1 is advisory-only: this module classifies a prompt with the local
``hermes-route`` command, normalizes the result, and writes a metadata-only audit
record. It never switches profiles or executes work in a specialist lane.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

DEFAULT_ROUTE_COMMAND = "/usr/local/bin/hermes-route"
DEFAULT_TIMEOUT_SECS = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _route_command() -> str | None:
    configured = os.environ.get("HERMES_ROUTE_COMMAND", "").strip()
    if configured:
        return configured
    discovered = shutil.which("hermes-route")
    if discovered:
        return discovered
    if Path(DEFAULT_ROUTE_COMMAND).exists():
        return DEFAULT_ROUTE_COMMAND
    return None


def _prompt_fingerprint(prompt: str) -> dict[str, str | int]:
    """Return metadata-only prompt fingerprints without retaining prompt text."""
    text = str(prompt or "")
    return {
        "prompt_sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        "prompt_length": len(text),
    }


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _fallback_advisory(
    *,
    surface: str,
    error: str | None = None,
    reason: str = "Route advisory unavailable; default to Main Hermes/macOS.",
) -> dict[str, Any]:
    return {
        "timestamp": _utc_now(),
        "surface": str(surface or "unknown"),
        "route_id": "main-hermes",
        "route_name": "Main Hermes / Orchestration",
        "owner": "macos",
        "profile": "macos",
        "prompt_class": "orchestration",
        "action": "Main Hermes handles",
        "blocked": False,
        "blocked_actions": [],
        "requires_approval": False,
        "reason": reason,
        "confidence": 0.0,
        "advisory_mode": True,
        "auto_execute": False,
        "is_live": True,
        "error": error,
    }


def normalize_route_payload(payload: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    route_id = str(payload.get("route_id") or "main-hermes")
    profile = str(payload.get("profile") or ("macos" if route_id == "main-hermes" else route_id))
    return {
        "timestamp": _utc_now(),
        "surface": str(surface or "unknown"),
        "route_id": route_id,
        "route_name": str(payload.get("route_name") or route_id),
        "owner": str(payload.get("owner") or profile or "macos"),
        "profile": profile,
        "prompt_class": str(payload.get("prompt_class") or "orchestration"),
        "action": str(payload.get("action") or "Route advisory only"),
        "blocked": _coerce_bool(payload.get("blocked"), False),
        "blocked_actions": _coerce_string_list(payload.get("blocked_actions")),
        "requires_approval": _coerce_bool(payload.get("requires_approval"), False),
        "reason": str(payload.get("reason") or ""),
        "confidence": _coerce_float(payload.get("confidence"), 0.0),
        "advisory_mode": True,
        "auto_execute": False,
        "is_live": _coerce_bool(payload.get("is_live"), route_id != "vault-local"),
        "error": payload.get("error"),
    }


def log_route_decision(advisory: Mapping[str, Any], prompt: str, *, log_path: Path | None = None) -> Path:
    """Append a metadata-only JSONL route-decision record and return its path."""
    target = log_path or (get_hermes_home() / "logs" / "routing_decisions.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": advisory.get("timestamp") or _utc_now(),
        "surface": advisory.get("surface") or "unknown",
        "route_id": advisory.get("route_id"),
        "profile": advisory.get("profile"),
        "owner": advisory.get("owner"),
        "confidence": advisory.get("confidence"),
        "blocked": advisory.get("blocked"),
        "requires_approval": advisory.get("requires_approval"),
        "blocked_actions": advisory.get("blocked_actions") or [],
        "advisory_mode": True,
        "auto_execute": False,
        "error": advisory.get("error"),
    }
    record.update(_prompt_fingerprint(prompt))
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def classify_route_advisory(
    prompt: str,
    *,
    surface: str = "unknown",
    timeout: float = DEFAULT_TIMEOUT_SECS,
    log: bool = True,
    command: str | None = None,
) -> dict[str, Any]:
    """Return an advisory-only route decision for ``prompt``.

    Failures are intentionally safe: no route command, timeout, parse error, or
    non-JSON output falls back to Main Hermes/macOS and can still be logged for
    observability. The helper never mutates session profile or job execution.
    """
    prompt_text = str(prompt or "")
    if not prompt_text.strip():
        advisory = _fallback_advisory(
            surface=surface,
            reason="Empty prompt; default to Main Hermes/macOS.",
        )
    else:
        route_command = command or _route_command()
        if not route_command:
            advisory = _fallback_advisory(surface=surface, error="hermes-route command not found")
        else:
            try:
                completed = subprocess.run(
                    [route_command, "--json", "--stdin"],
                    input=prompt_text,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                advisory = _fallback_advisory(surface=surface, error="hermes-route timeout")
            except OSError as exc:
                advisory = _fallback_advisory(surface=surface, error=f"hermes-route failed: {exc}")
            else:
                try:
                    payload = json.loads(completed.stdout or "{}")
                except json.JSONDecodeError:
                    advisory = _fallback_advisory(
                        surface=surface,
                        error=f"hermes-route non-json output rc={completed.returncode}",
                    )
                else:
                    if not isinstance(payload, dict):
                        advisory = _fallback_advisory(
                            surface=surface,
                            error=f"hermes-route invalid payload rc={completed.returncode}",
                        )
                    elif completed.returncode not in {0, 1} and payload.get("error"):
                        advisory = _fallback_advisory(
                            surface=surface,
                            error=str(payload.get("error")),
                        )
                    else:
                        advisory = normalize_route_payload(payload, surface=surface)
                        if completed.returncode not in {0, 1}:
                            advisory["error"] = f"hermes-route rc={completed.returncode}"
    if log:
        try:
            log_route_decision(advisory, prompt_text)
        except OSError as exc:
            advisory = dict(advisory)
            advisory["log_error"] = str(exc)
    return advisory


def format_route_advisory(advisory: Mapping[str, Any]) -> str:
    """Render a concise user-facing advisory line."""
    route_id = advisory.get("route_id") or "main-hermes"
    profile = advisory.get("profile") or "macos"
    confidence = _coerce_float(advisory.get("confidence"), 0.0)
    approval = "; approval required before specialist execution" if advisory.get("requires_approval") else ""
    blocked = advisory.get("blocked_actions") or []
    blocked_text = f"; guarded actions: {', '.join(blocked)}" if blocked else ""
    return (
        f"Route advisory: {route_id} -> profile {profile} "
        f"(confidence {confidence:.2f}; advisory-only, no auto-switch{approval}{blocked_text})."
    )

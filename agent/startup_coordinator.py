"""Hermes startup coordination and authoritative v3 receipts.

The expensive discovery work is started before ``AIAgent`` construction.  This
module owns the final, bounded startup checkpoint immediately before the system
prompt is built and the first model request can be made.  It deliberately does
not mutate tool registrations: capabilities that missed the earlier ready
barrier are recorded as pending/degraded for this session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA = "hermes.startup-receipt.v3"
RECEIPT_DIRNAME = "startup"
MAX_STARTUP_CONTEXT_BYTES = 4096
READY_BARRIER_SECONDS = 1.5
CAPABILITY_STATES = frozenset(
    {"configured", "ready", "pending", "degraded", "disabled"}
)
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|token)", re.I
)


@dataclass(frozen=True)
class StartupOutcome:
    """The one-shot context and receipt produced for a new session."""

    context: str
    receipt: Dict[str, Any]
    receipt_path: Optional[Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(128 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _json_safe_redacted(value: Any) -> Any:
    """Return JSON-compatible data without secret-looking values."""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            result[key] = "<redacted>" if _SECRET_KEY_RE.search(key) else _json_safe_redacted(raw_value)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe_redacted(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _merge_patch(target: Dict[str, Any], patch: Mapping[str, Any]) -> None:
    """Merge a redacted hook patch without allowing core receipt overrides."""

    protected = {
        "schema",
        "surface",
        "profile",
        "session_id",
        "generation",
        "generated_at",
    }
    for raw_key, raw_value in patch.items():
        key = str(raw_key)
        if key in protected:
            continue
        value = _json_safe_redacted(raw_value)
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_patch(target[key], value)
        else:
            target[key] = value


def _bounded_utf8(text: str, limit: int = MAX_STARTUP_CONTEXT_BYTES) -> tuple[str, bool]:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, False
    marker = "\n[Hermes startup context truncated to 4 KB]"
    marker_bytes = marker.encode("utf-8")
    budget = max(0, limit - len(marker_bytes))
    prefix = raw[:budget]
    while prefix:
        try:
            return prefix.decode("utf-8") + marker, True
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return marker_bytes[:limit].decode("utf-8", errors="ignore"), True


def _profile_name(home: Path) -> str:
    if home.parent.name == "profiles":
        return home.name
    return "default"


def _contract_path(home: Path) -> Path:
    profile_contract = home / "canon" / "startup-contract.yaml"
    if profile_contract.exists():
        return profile_contract
    return get_default_hermes_root() / "canon" / "startup-contract.yaml"


def _configured_mcp_names(config: Mapping[str, Any]) -> set[str]:
    raw = config.get("mcp_servers") or config.get("mcpServers") or config.get("mcp") or {}
    if not isinstance(raw, Mapping):
        return set()
    return {
        str(name)
        for name, value in raw.items()
        if not isinstance(value, Mapping) or value.get("enabled", True) is not False
    }


def _semantic_capabilities(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Describe startup capabilities by role, not provider/server spelling."""

    names = _configured_mcp_names(config)
    in_flight = False
    statuses: Dict[str, Mapping[str, Any]] = {}
    if names:
        try:
            from hermes_cli.mcp_startup import mcp_discovery_in_flight

            in_flight = bool(mcp_discovery_in_flight())
        except Exception:
            pass
        try:
            from tools.mcp_tool import get_mcp_status

            statuses = {
                str(row.get("name")): row
                for row in get_mcp_status()
                if isinstance(row, Mapping) and row.get("name")
            }
        except Exception:
            logger.debug("startup MCP status read failed", exc_info=True)

    def mcp_role(candidates: Iterable[str]) -> Dict[str, Any]:
        selected = next((name for name in candidates if name in names or name in statuses), None)
        if selected is None:
            return {"state": "disabled"}
        status = statuses.get(selected, {})
        if status.get("disabled"):
            state = "disabled"
        elif status.get("connected"):
            state = "ready"
        elif in_flight or str(status.get("status", "")).lower() in {"connecting", "pending"}:
            state = "pending"
        elif status:
            state = "degraded"
        else:
            state = "configured"
        return {"state": state, "provider": selected}

    memory_cfg = config.get("memory")
    memory_enabled = not (
        memory_cfg is False
        or (isinstance(memory_cfg, Mapping) and memory_cfg.get("enabled") is False)
    )
    memory_provider = "hindsight"
    if isinstance(memory_cfg, Mapping):
        memory_provider = str(memory_cfg.get("provider") or memory_provider)

    plugin_state = "configured"
    # The lifecycle hook imports the plugin runtime before this coordinator in
    # real sessions. Avoid turning a diagnostic receipt into a cold import of
    # the multi-thousand-line plugin module for direct/minimal callers.
    if "hermes_cli.plugins" in sys.modules:
        try:
            from hermes_cli.plugins import get_plugin_manager

            plugin_state = "ready" if get_plugin_manager()._discovered else "pending"
        except Exception:
            plugin_state = "degraded"

    capabilities = {
        "plugins": {"state": plugin_state},
        "memory": {
            "state": "configured" if memory_enabled else "disabled",
            "provider": memory_provider if memory_enabled else None,
        },
        "memory_mcp": mcp_role(("fleet-memory", "hindsight")),
        "obsidian": mcp_role(("obsidian", "obsidian-read", "obsidian_read")),
    }
    for detail in capabilities.values():
        if detail["state"] not in CAPABILITY_STATES:
            detail["state"] = "degraded"
    return capabilities


def _hook_trust(config: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from agent.shell_hooks import (
            allowlist_entry_for,
            iter_configured_hooks,
            script_mtime_iso,
        )

        entries = []
        for spec in iter_configured_hooks(dict(config)):
            approval = allowlist_entry_for(spec.event, spec.command)
            current_mtime = script_mtime_iso(spec.command)
            approved_mtime = approval.get("script_mtime") if isinstance(approval, Mapping) else None
            trusted = bool(approval) and (
                not approved_mtime or not current_mtime or approved_mtime == current_mtime
            )
            entries.append(
                {
                    "event": spec.event,
                    "command_hash": hashlib.sha256(spec.command.encode("utf-8")).hexdigest(),
                    "trusted": trusted,
                    "reason": None if trusted else ("modified" if approval else "not_allowlisted"),
                }
            )
        return {
            "state": "ready" if all(row["trusted"] for row in entries) else "degraded",
            "configured": len(entries),
            "entries": entries,
        }
    except Exception as exc:
        return {"state": "degraded", "configured": 0, "reason": type(exc).__name__}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_startup_receipt(home: Path, receipt: Dict[str, Any]) -> Path:
    """Write an immutable per-session receipt and atomically replace latest."""

    safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(receipt["session_id"])).strip("._")
    if not safe_session:
        safe_session = str(receipt["generation"])
    receipt_dir = home / "state" / RECEIPT_DIRNAME
    immutable = receipt_dir / f"{safe_session}.json"
    if immutable.exists():
        immutable = receipt_dir / f"{safe_session}-{receipt['generation']}.json"
    _atomic_json(immutable, receipt)
    pointer = {
        "schema": RECEIPT_SCHEMA,
        "session_id": receipt["session_id"],
        "generation": receipt["generation"],
        "generated_at": receipt["generated_at"],
        "receipt": immutable.name,
    }
    _atomic_json(receipt_dir / "latest.json", pointer)
    return immutable


def coordinate_session_start(
    agent: Any,
    hook_results: Iterable[Any],
    *,
    elapsed_ms: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    home: Optional[Path] = None,
) -> StartupOutcome:
    """Collect exactly one startup injector and persist its v3 receipt."""

    started = time.monotonic()
    hermes_home = Path(home) if home is not None else get_hermes_home()
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    if not isinstance(config, Mapping):
        config = {}

    context = ""
    injector: Optional[str] = None
    injector_count = 0
    degraded: list[str] = []
    hook_patch: Dict[str, Any] = {}
    for index, result in enumerate(hook_results or []):
        if isinstance(result, str):
            piece = result.strip()
            patch = None
            identity = f"plugin-result-{index + 1}"
        elif isinstance(result, Mapping):
            raw_piece = result.get("context")
            piece = raw_piece.strip() if isinstance(raw_piece, str) else ""
            patch = result.get("receipt_patch")
            identity = str(result.get("_hermes_hook_identity") or f"plugin-result-{index + 1}")
        else:
            continue
        if isinstance(patch, Mapping):
            _merge_patch(hook_patch, patch)
        if not piece:
            continue
        injector_count += 1
        if not context:
            context = piece
            injector = identity
        else:
            degraded.append(
                f"duplicate startup injector rejected: {identity}; retained {injector}"
            )

    context, truncated = _bounded_utf8(context)
    if truncated:
        degraded.append("startup context exceeded 4096 bytes and was truncated")

    config_path = hermes_home / "config.yaml"
    contract_path = _contract_path(hermes_home)
    capabilities = _semantic_capabilities(config)
    for role, detail in capabilities.items():
        if detail.get("state") == "degraded":
            degraded.append(f"{role} capability degraded")

    trust = _hook_trust(config)
    if trust.get("state") == "degraded":
        degraded.append("one or more configured hooks are untrusted or modified")

    generation = os.urandom(8).hex()
    receipt: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "surface": str(getattr(agent, "platform", None) or "cli"),
        "profile": _profile_name(hermes_home),
        "session_id": str(getattr(agent, "session_id", "") or generation),
        "generation": generation,
        "generated_at": _utc_now(),
        "config_hash": _sha256_file(config_path),
        "contract_hash": _sha256_file(contract_path),
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
        "injector": {
            "identity": injector,
            "count": injector_count,
            "accepted": 1 if context else 0,
        },
        "hook_trust": trust,
        "capabilities": capabilities,
        "timings_ms": {
            "ready_barrier_budget": int(READY_BARRIER_SECONDS * 1000),
            "session_hooks": round(float(elapsed_ms or 0.0), 3),
        },
        "degraded_reasons": degraded,
    }
    if hook_patch:
        _merge_patch(receipt, hook_patch)
    receipt["timings_ms"]["coordinator"] = round((time.monotonic() - started) * 1000, 3)

    receipt_path: Optional[Path] = None
    try:
        receipt_path = write_startup_receipt(hermes_home, receipt)
    except OSError as exc:
        logger.warning("startup receipt write failed for session=%s: %s", receipt["session_id"], exc)
    return StartupOutcome(context=context, receipt=receipt, receipt_path=receipt_path)

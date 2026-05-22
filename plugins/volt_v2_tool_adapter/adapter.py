"""Minimal Volt V2 tool adapter proof.

This module deliberately uses existing Hermes plugin hooks.  It does not
replace core dispatch, override tools, send network traffic, or touch memory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .audit import build_audit_event, write_audit_event
from .config import VoltV2ToolAdapterConfig, load_adapter_config

logger = logging.getLogger(__name__)

MUTATION_TOOLS = frozenset(
    {
        "write_file",
        "patch",
        "terminal",
        "cronjob",
        "send_message",
        "memory",
        "skill_manage",
    }
)
READ_ONLY_TOOLS = frozenset({"read_file", "search_files", "vision_analyze"})


@dataclass(frozen=True)
class AdapterDecision:
    active: bool
    allowlisted: bool
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return ",".join(self.reasons) if self.reasons else "ok"


def evaluate_call(
    tool_name: str,
    args: Mapping[str, Any] | None,
    config: VoltV2ToolAdapterConfig | None = None,
) -> AdapterDecision:
    cfg = config or load_adapter_config()
    if not cfg.enabled:
        return AdapterDecision(active=False, allowlisted=False, reasons=("disabled",))

    reasons: list[str] = []
    if tool_name in cfg.denylist_tools:
        reasons.append("denylisted_tool")
    if cfg.allowlist_tools and tool_name not in cfg.allowlist_tools:
        reasons.append("tool_not_allowlisted")
    if _args_have_sensitive_material(args):
        reasons.append("sensitive_args")

    path_values = _extract_path_values(args)
    if path_values:
        for value in path_values:
            if _is_under_any(value, cfg.denylist_path_prefixes):
                reasons.append("denylisted_path")
                break
        if cfg.allowlist_paths:
            for value in path_values:
                if not _is_under_any(value, cfg.allowlist_paths):
                    reasons.append("path_not_allowlisted")
                    break

    # If config has path allowlists and the call has no path values, keep the
    # gate tool-only. Some safe tools like search_files may route by pattern.
    allowlisted = not reasons
    return AdapterDecision(active=True, allowlisted=allowlisted, reasons=tuple(dict.fromkeys(reasons)))


def on_post_tool_call(
    tool_name: str = "",
    args: Mapping[str, Any] | None = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int | float | None = None,
    config: VoltV2ToolAdapterConfig | None = None,
    **_: Any,
) -> None:
    cfg = config or load_adapter_config()
    if not cfg.enabled or not cfg.verification.write_audit_jsonl:
        return
    try:
        decision = evaluate_call(tool_name, args, cfg)
        if not decision.allowlisted:
            return
        event = build_audit_event(
            tool_name=tool_name,
            args=args,
            result=result,
            mode=cfg.mode,
            decision="observed" if cfg.mode == "observe" else cfg.mode,
            allowlisted=True,
            reason=decision.reason,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            duration_ms=duration_ms,
        )
        write_audit_event(cfg.audit_path, event)
    except Exception as exc:
        # Observe/transform hooks must be fail-open: never break the core tool
        # pipeline because an audit observer failed.
        handle_adapter_exception(tool_name, exc, cfg)
        return


def transform_tool_result(
    tool_name: str = "",
    args: Mapping[str, Any] | None = None,
    result: Any = None,
    config: VoltV2ToolAdapterConfig | None = None,
    **_: Any,
) -> str | None:
    cfg = config or load_adapter_config()
    if not cfg.enabled or cfg.mode != "transform":
        return None
    try:
        decision = evaluate_call(tool_name, args, cfg)
        if not decision.allowlisted or not isinstance(result, str):
            return None
        transformed = _with_result_marker(result, tool_name=tool_name, mode=cfg.mode)
        if cfg.verification.write_audit_jsonl:
            event = build_audit_event(
                tool_name=tool_name,
                args=args,
                result=result,
                mode=cfg.mode,
                decision="transformed",
                allowlisted=True,
                reason=decision.reason,
            )
            write_audit_event(cfg.audit_path, event)
        return transformed
    except Exception as exc:
        return handle_adapter_exception(tool_name, exc, cfg)


def handle_adapter_exception(
    tool_name: str,
    exc: Exception,
    config: VoltV2ToolAdapterConfig | None = None,
) -> str | None:
    cfg = config or load_adapter_config()
    logger.warning("Volt V2 adapter fallback for %s: %s", tool_name, exc)

    if cfg.mode in {"observe", "transform"}:
        return None
    if cfg.mode in {"route", "override"} and tool_name in MUTATION_TOOLS:
        return json.dumps(
            {
                "error": f"Volt V2 adapter failed closed for mutation tool '{tool_name}'",
                "adapter": "volt_v2_tool_adapter",
                "mode": cfg.mode,
            },
            ensure_ascii=False,
        )
    return None


def _with_result_marker(result: str, *, tool_name: str, mode: str) -> str:
    marker = {
        "adapter": "volt_v2_tool_adapter",
        "mode": mode,
        "decision": "transformed",
        "tool_name": tool_name,
    }
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            parsed.setdefault("_volt_v2_adapter", marker)
            return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass
    return result + "\n\n" + json.dumps({"_volt_v2_adapter": marker}, ensure_ascii=False)


def _extract_path_values(args: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(args, Mapping):
        return []
    values: list[str] = []
    for key, value in args.items():
        lowered = str(key).lower()
        if any(part in lowered for part in ("path", "file", "dir", "root")):
            if isinstance(value, (list, tuple, set)):
                values.extend(str(item) for item in value if item is not None)
            elif value is not None:
                values.append(str(value))
    return values


def _args_have_sensitive_material(args: Mapping[str, Any] | None) -> bool:
    if not isinstance(args, Mapping):
        return False
    sensitive_parts = ("api_key", "apikey", "authorization", "credential", "password", "secret", "token")
    for key, value in args.items():
        if any(part in str(key).lower() for part in sensitive_parts):
            return True
        if isinstance(value, Mapping) and _args_have_sensitive_material(value):
            return True
    return False


def _is_under_any(path_value: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return False
    value = _normalise_path(path_value)
    for prefix in prefixes:
        if not prefix:
            continue
        norm_prefix = _normalise_path(prefix)
        if value == norm_prefix or value.startswith(norm_prefix.rstrip("/") + "/"):
            return True
    return False


def _normalise_path(path_value: str) -> str:
    try:
        return str(Path(path_value).expanduser().resolve(strict=False))
    except Exception:
        return str(path_value)

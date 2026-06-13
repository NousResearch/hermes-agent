"""Import-independent fallback helpers for the generic runtime guard.

These helpers avoid importing ``agent.runtime_guard`` so callers can still
fail closed when the main guard module or a configured guard cannot be
resolved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _runtime_guard_cfg(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    block = config.get("runtime_guard")
    return block if isinstance(block, Mapping) else config


def _load_runtime_guard_cfg_readonly() -> tuple[Mapping[str, Any], bool]:
    try:
        from hermes_cli.config import load_config_readonly

        return _runtime_guard_cfg(load_config_readonly()), True
    except Exception:
        return {}, False


def _agent_runtime_guard_cfg(agent: Any) -> tuple[Mapping[str, Any], bool]:
    for attr in ("_runtime_guard_config", "runtime_guard_config"):
        config = getattr(agent, attr, None)
        if isinstance(config, Mapping):
            return _runtime_guard_cfg(config), True
    return _load_runtime_guard_cfg_readonly()


def _enabled_and_fail_closed(cfg: Mapping[str, Any]) -> bool:
    return bool(cfg.get("enabled", False)) and bool(cfg.get("fail_closed", True))


def runtime_guard_enabled_for_agent(agent: Any) -> bool:
    cfg, _ = _agent_runtime_guard_cfg(agent)
    return bool(cfg.get("enabled", False))


def runtime_guard_enabled_for_current_context() -> bool:
    cfg, _ = _load_runtime_guard_cfg_readonly()
    return bool(cfg.get("enabled", False))


def runtime_guard_guard_error_reason(exc: BaseException, *, guard_name: str) -> str:
    detail = str(exc) or exc.__class__.__name__
    return f"runtime_guard {guard_name} guard internal error: {detail}"


def runtime_guard_block_result(
    reason: str,
    *,
    guard_name: str,
    code: str = "runtime_guard_error",
    metadata: Mapping[str, Any] | None = None,
) -> str:
    block = {
        "allowed": False,
        "reason": reason,
        "message": reason,
        "code": code,
        "context": {"guard_name": guard_name, "action": guard_name},
    }
    if metadata:
        block["metadata"] = dict(metadata)
    return json.dumps(
        {
            "error": reason,
            "status": "blocked",
            "blocked_by": "runtime_guard",
            "code": code,
            "runtime_guard_block": block,
        },
        ensure_ascii=False,
    )


def runtime_guard_tool_block_for_agent_guard_error(
    agent: Any,
    exc: BaseException,
    *,
    guard_name: str,
) -> tuple[str | None, str | None]:
    cfg, _ = _agent_runtime_guard_cfg(agent)
    if not _enabled_and_fail_closed(cfg):
        return None, None
    reason = runtime_guard_guard_error_reason(exc, guard_name=guard_name)
    return runtime_guard_block_result(reason, guard_name=guard_name), reason


def runtime_guard_tool_block_for_current_guard_error(
    exc: BaseException,
    *,
    guard_name: str,
) -> tuple[str | None, str | None]:
    cfg, _ = _load_runtime_guard_cfg_readonly()
    if not _enabled_and_fail_closed(cfg):
        return None, None
    reason = runtime_guard_guard_error_reason(exc, guard_name=guard_name)
    return runtime_guard_block_result(reason, guard_name=guard_name), reason


def runtime_guard_blocked_text(reason: str, *, guard_name: str) -> str:
    return (
        "Response blocked by runtime_guard.\n\n"
        f"Guard: {guard_name}\n"
        f"Reason: {reason}"
    )


def runtime_guard_final_text_for_agent_guard_error(
    agent: Any,
    exc: BaseException,
    *,
    guard_name: str,
) -> str | None:
    cfg, _ = _agent_runtime_guard_cfg(agent)
    if not _enabled_and_fail_closed(cfg):
        return None
    return runtime_guard_blocked_text(
        runtime_guard_guard_error_reason(exc, guard_name=guard_name),
        guard_name=guard_name,
    )

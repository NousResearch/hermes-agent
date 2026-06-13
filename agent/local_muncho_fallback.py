"""Import-independent Local Muncho guard fallback helpers.

These helpers intentionally avoid importing ``agent.local_muncho``.  They are
used only when the real runtime guard import or helper resolution failed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


def _runtime_cfg(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    block = config.get("muncho_runtime")
    return block if isinstance(block, Mapping) else config


def _load_runtime_cfg_readonly() -> tuple[Mapping[str, Any], bool]:
    try:
        from hermes_cli.config import load_config_readonly

        return _runtime_cfg(load_config_readonly()), True
    except Exception:
        return {}, False


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _agent_runtime_cfg(agent: Any) -> tuple[Mapping[str, Any], bool]:
    config = getattr(agent, "_local_muncho_config", None)
    if isinstance(config, Mapping):
        return _runtime_cfg(config), True
    return _load_runtime_cfg_readonly()


def _agent_lane(agent: Any) -> str:
    context = getattr(agent, "_local_muncho_context", None)
    lane = _get_value(context, "lane")
    return str(
        lane
        or getattr(agent, "local_muncho_lane", None)
        or os.getenv("HERMES_MUNCHO_RUNTIME_LANE")
        or ""
    )


def _current_lane() -> str:
    return str(os.getenv("HERMES_MUNCHO_RUNTIME_LANE") or "")


def _plausibly_enabled_from_cfg(
    cfg: Mapping[str, Any],
    *,
    cfg_known: bool,
    lane: str,
) -> bool:
    if bool(cfg.get("enabled", False)):
        expected_lane = str(cfg.get("lane") or "internal-support")
        return not lane or lane == expected_lane
    if cfg_known:
        return False
    return bool(lane)


def local_muncho_plausibly_enabled_for_agent(agent: Any) -> bool:
    cfg, cfg_known = _agent_runtime_cfg(agent)
    return _plausibly_enabled_from_cfg(
        cfg,
        cfg_known=cfg_known,
        lane=_agent_lane(agent),
    )


def local_muncho_plausibly_enabled_for_current_context() -> bool:
    cfg, cfg_known = _load_runtime_cfg_readonly()
    return _plausibly_enabled_from_cfg(
        cfg,
        cfg_known=cfg_known,
        lane=_current_lane(),
    )


def local_muncho_guard_error_reason(exc: BaseException, *, guard_name: str) -> str:
    detail = str(exc) or exc.__class__.__name__
    return f"Local Muncho runtime {guard_name} guard internal error: {detail}"


def local_muncho_tool_block_result(
    reason: str,
    *,
    code: str = "local_muncho_guard_error",
) -> str:
    return json.dumps(
        {
            "error": reason,
            "status": "blocked",
            "blocked_by": "local_muncho_runtime",
            "code": code,
        },
        ensure_ascii=False,
    )


def local_muncho_tool_block_for_agent_guard_error(
    agent: Any,
    exc: BaseException,
    *,
    guard_name: str,
) -> tuple[str | None, str | None]:
    if not local_muncho_plausibly_enabled_for_agent(agent):
        return None, None
    reason = local_muncho_guard_error_reason(exc, guard_name=guard_name)
    return local_muncho_tool_block_result(reason), reason


def local_muncho_tool_block_for_current_guard_error(
    exc: BaseException,
    *,
    guard_name: str,
) -> tuple[str | None, str | None]:
    if not local_muncho_plausibly_enabled_for_current_context():
        return None, None
    reason = local_muncho_guard_error_reason(exc, guard_name=guard_name)
    return local_muncho_tool_block_result(reason), reason


def local_muncho_blocked_text(reason: str) -> str:
    return "\n".join(
        [
            "VERDICT: BLOCKED",
            "TL;DR: Local Muncho runtime blocked the response before delivery.",
            "CATEGORY: runtime_guard",
            "EVIDENCE_CHECKED: active-lease",
            f"EVIDENCE_GAP: {reason}",
            "STATUS: blocked",
            "NEXT_ACTION: restore local lease or hand off to cloud runtime",
            "APPROVAL_NEEDED: yes",
            "RISK: unsafe-runtime-transition",
        ]
    )


def local_muncho_final_text_for_agent_guard_error(
    agent: Any,
    exc: BaseException,
    *,
    guard_name: str,
) -> str | None:
    if not local_muncho_plausibly_enabled_for_agent(agent):
        return None
    return local_muncho_blocked_text(
        local_muncho_guard_error_reason(exc, guard_name=guard_name)
    )

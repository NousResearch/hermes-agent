"""Feature flags and harness linkage for Kynver-first Forge operating tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .agentos_bridge import agentos_enabled, load_kynver_agentos_config


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def kynver_operating_tools_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Default-on when AgentOS credentials exist; opt out via KYNVER_OPERATING_TOOLS=false."""

    merged = dict(os.environ)
    if env:
        merged.update(env)
    if not agentos_enabled(merged):
        return False
    return _truthy(merged.get("KYNVER_OPERATING_TOOLS"), default=True)


@dataclass(frozen=True)
class OperatingLinkage:
    plan_id: str | None
    task_id: str | None
    session_id: str | None
    executor_ref: str

    @property
    def linked(self) -> bool:
        return bool(self.plan_id or self.task_id)


def load_operating_linkage(env: Mapping[str, str] | None = None) -> OperatingLinkage:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    cfg = load_kynver_agentos_config(merged)
    return OperatingLinkage(
        plan_id=(merged.get("KYNVER_PLAN_ID") or "").strip() or None,
        task_id=(merged.get("KYNVER_TASK_ID") or "").strip() or None,
        session_id=(merged.get("HERMES_SESSION_ID") or merged.get("KYNVER_SESSION_ID") or "").strip() or None,
        executor_ref=f"hermes:{cfg.slug or 'forge'}",
    )

"""Target-issued execution authority for RoomLink member turns."""

from __future__ import annotations

import hashlib
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from gateway.hosted_rooms_common import bounded_int, compact_json, identifier

POLICY_VERSION = 1
MAX_POLICY_TOOLSETS = 128
MAX_POLICY_ITERATIONS = (1 << 53) - 1
_POLICY_FIELDS = {"version", "target_profile", "enabled_toolsets", "approval_mode", "max_iterations", "policy_digest"}


class RoomExecutionPolicyError(ValueError): """A RoomLink execution policy is malformed or no longer current."""


def _policy_digest(unsigned: Mapping[str, Any]) -> str:
    return hashlib.sha256(compact_json(unsigned).encode("ascii")).hexdigest()


def _identifier(value: Any, *, field: str) -> str:
    # Stringifies first (``None`` -> ""), so non-strings also fail as "is invalid".
    return identifier(str(value or ""), label=field, error=RoomExecutionPolicyError, invalid=f"{field} is invalid")


@dataclass(frozen=True)
class RoomExecutionPolicy:
    """Immutable target policy applied at the agent and approval boundaries."""
    version: int
    target_profile: str
    enabled_toolsets: tuple[str, ...]
    approval_mode: str
    max_iterations: int
    policy_digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoomExecutionPolicy":
        if not isinstance(value, Mapping) or set(value) != _POLICY_FIELDS:
            raise RoomExecutionPolicyError("execution policy fields are invalid")
        if value["version"] != POLICY_VERSION:
            raise RoomExecutionPolicyError("execution policy version is unsupported")
        target_profile = _identifier(value["target_profile"], field="target_profile")
        raw_toolsets = value["enabled_toolsets"]
        if not isinstance(raw_toolsets, list) or not 1 <= len(raw_toolsets) <= MAX_POLICY_TOOLSETS:
            raise RoomExecutionPolicyError("enabled_toolsets are invalid")
        toolsets = tuple(sorted(_identifier(item, field="enabled_toolset") for item in raw_toolsets))
        if len(set(toolsets)) != len(toolsets) or "bot_room" not in toolsets:
            raise RoomExecutionPolicyError("enabled_toolsets are invalid")
        approval_mode = str(value["approval_mode"] or "").strip().lower()
        if approval_mode not in {"manual", "smart", "off"}:
            raise RoomExecutionPolicyError("approval_mode is invalid")
        max_iterations = bounded_int(
            value["max_iterations"], error=RoomExecutionPolicyError, message="max_iterations is invalid", low=1,
            high=MAX_POLICY_ITERATIONS)
        unsigned = {
            "version": POLICY_VERSION, "target_profile": target_profile, "enabled_toolsets": list(toolsets),
            "approval_mode": approval_mode, "max_iterations": max_iterations}
        supplied = str(value["policy_digest"] or "").strip().lower()
        if supplied != _policy_digest(unsigned):
            raise RoomExecutionPolicyError("policy_digest does not match the execution policy")
        return cls(**unsigned, policy_digest=supplied)

    def as_mapping(self) -> dict[str, Any]:
        return {**asdict(self), "enabled_toolsets": list(self.enabled_toolsets)}


def execution_policy_mapping(*, target_profile: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the effective API-server policy from the target's own config."""
    if config is None:
        from gateway.run import _load_gateway_config
        config = _load_gateway_config()
    if not isinstance(config, Mapping):
        raise RoomExecutionPolicyError("gateway config is invalid")
    from hermes_cli.config import resolve_turn_limit
    from hermes_cli.tools_config import _get_platform_tools
    from tools.approval import _YOLO_MODE_FROZEN
    from tools.approval_context import _normalize_approval_mode
    toolsets = sorted({*_get_platform_tools(dict(config), "api_server"), "bot_room"})
    agent = config.get("agent") if isinstance(config.get("agent"), Mapping) else {}
    approvals = config.get("approvals") if isinstance(config.get("approvals"), Mapping) else {}
    unsigned = {
        "version": POLICY_VERSION, "target_profile": _identifier(target_profile, field="target_profile"),
        "enabled_toolsets": toolsets,
        "approval_mode": ("off" if _YOLO_MODE_FROZEN else _normalize_approval_mode(approvals.get("mode", "manual"))),
        "max_iterations": min(resolve_turn_limit(agent.get("max_turns")), MAX_POLICY_ITERATIONS)}
    value = {**unsigned, "policy_digest": _policy_digest(unsigned)}
    return RoomExecutionPolicy.from_mapping(value).as_mapping()


_CURRENT_POLICY: ContextVar[RoomExecutionPolicy | None] = ContextVar("hosted_room_execution_policy", default=None)


def bind_room_execution_policy(policy: RoomExecutionPolicy) -> Token:
    return _CURRENT_POLICY.set(policy)


def reset_room_execution_policy(token: Token) -> None:
    _CURRENT_POLICY.reset(token)


def current_room_execution_policy() -> RoomExecutionPolicy | None:
    return _CURRENT_POLICY.get()


__all__ = [
    "MAX_POLICY_ITERATIONS", "POLICY_VERSION", "RoomExecutionPolicy", "RoomExecutionPolicyError",
    "bind_room_execution_policy", "current_room_execution_policy", "execution_policy_mapping",
    "reset_room_execution_policy"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

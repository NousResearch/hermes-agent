"""Per-platform slash command access control.

A second axis beside ``allow_from``: of the users allowed to talk to the gateway, which may run
which slash commands. Two lists per scope (DM vs group, mirroring ``allow_from`` /
``group_allow_from``): ``allow_admin_from`` (user IDs that get every registered command, built-in
and plugin) and ``user_allowed_commands`` (names non-admins may run; empty/unset -> only the
``_ALWAYS_ALLOWED_FOR_USERS`` floor). No ``allow_admin_from`` for a scope => gating disabled there,
so existing installs are unaffected until an operator lists an admin. Applied at the dispatch
site in ``gateway/run.py`` via the live registry; never affects plain chat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, Optional

logger = logging.getLogger(__name__)

# Read-only floor every allowed user keeps under gating, so a non-admin can still discover what
# they can do. ``user_allowed_commands`` only adds to this set, never restricts it.
_ALWAYS_ALLOWED_FOR_USERS: frozenset[str] = frozenset({"help", "whoami"})

_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})

# scope -> (admin list key, user command list key)
_SCOPE_KEYS = {
    "dm": ("allow_admin_from", "user_allowed_commands"),
    "group": ("group_allow_admin_from", "group_user_allowed_commands"),
}


@dataclass(frozen=True)
class SlashAccessPolicy:
    """Resolved access policy for one (platform, scope) pair; scope is ``"dm"`` or ``"group"``."""

    enabled: bool  # gating active for this scope?
    admin_user_ids: frozenset[str]
    user_allowed_commands: frozenset[str]

    def is_admin(self, user_id: Optional[str]) -> bool:
        # Gating disabled -> everyone is admin so callers can use is_admin/can_run uniformly.
        if not self.enabled:
            return True
        return bool(user_id) and str(user_id) in self.admin_user_ids

    def can_run(self, user_id: Optional[str], canonical_cmd: str) -> bool:
        if self.is_admin(user_id):
            return True
        return bool(canonical_cmd) and (
            canonical_cmd in _ALWAYS_ALLOWED_FOR_USERS or canonical_cmd in self.user_allowed_commands
        )


_DISABLED_POLICY = SlashAccessPolicy(
    enabled=False, admin_user_ids=frozenset(), user_allowed_commands=frozenset()
)
# Gated with nobody on the admin list: only the ``_ALWAYS_ALLOWED_FOR_USERS`` floor runs.
_FAIL_CLOSED_POLICY = SlashAccessPolicy(
    enabled=True, admin_user_ids=frozenset(), user_allowed_commands=frozenset()
)


def _coerce_list(raw: Any, normalize: Callable[[str], str] = str) -> frozenset[str]:
    """Normalize a YAML-loaded value (None, list/tuple/set, comma string, or scalar) into a
    frozenset of stripped, non-empty strings, applying ``normalize`` to each."""
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        items = (raw,)  # single scalar (int user id, etc.)
    return frozenset(s for s in (normalize(str(it).strip()) for it in items) if s)


def _coerce_id_list(raw: Any) -> frozenset[str]:
    """Normalize an admin/user id list into a frozenset of strings."""
    return _coerce_list(raw)


def _coerce_command_list(raw: Any) -> frozenset[str]:
    """Command allowlist: strip leading slashes (``/help`` or ``help``) and lowercase to match
    how ``resolve_command()`` stores names."""
    return _coerce_list(raw, lambda s: s.lstrip("/").lower())


def policy_from_extra(extra: dict, scope: str) -> SlashAccessPolicy:
    """Build a policy from a platform's ``extra`` dict for one scope.

    DM scope falls back to ``group_user_allowed_commands`` ONLY for the command list, and only
    when DM didn't set its own. Admin lists are NOT cross-scope: a DM admin is not a group admin.
    """
    admin_key, cmd_key = _SCOPE_KEYS.get(scope, _SCOPE_KEYS["dm"])
    admin_ids = _coerce_id_list(extra.get(admin_key))
    cmds = _coerce_command_list(extra.get(cmd_key))
    if scope == "dm" and not cmds:
        cmds = _coerce_command_list(extra.get("group_user_allowed_commands"))
    return SlashAccessPolicy(enabled=bool(admin_ids), admin_user_ids=admin_ids, user_allowed_commands=cmds)


def policy_for_source(gateway_config: Any, source: Any) -> SlashAccessPolicy:
    """Resolve the slash-gating policy for a SessionSource.

    Disabled (allow-everything) when gateway_config/source is None, the platform has no
    PlatformConfig, or no admin list is set for the scope. Gates slash commands only, never chat.
    """
    if gateway_config is None or source is None:
        return _DISABLED_POLICY
    platforms = getattr(gateway_config, "platforms", None)
    platform_config = None
    if platforms is not None:
        try:
            platform_config = platforms.get(source.platform)
        except Exception:
            platform_config = None
    # ``extra`` from a PlatformConfig-like object, or a bare dict as some test harnesses pass.
    extra = getattr(platform_config, "extra", None)
    if not isinstance(extra, dict):
        extra = platform_config if isinstance(platform_config, dict) else {}
    chat_type = getattr(source, "chat_type", None)
    normalized = str(chat_type).strip().lower() if chat_type is not None else ""
    if normalized:
        scope = "dm" if normalized in _DM_CHAT_TYPES else "group"
        return policy_from_extra(extra, scope)
    # Blank/None chat_type is ambiguous (relay frames may send "" / null; restored rows keep a
    # stored empty value verbatim). _DM_CHAT_TYPES lists "" but a truthiness guard made it
    # unreachable: resolve to whichever scope is gated so an ambiguous source never lands in an
    # ungated allow-everything scope; keep the historical group scope on a tie.
    dm_policy = policy_from_extra(extra, "dm")
    group_policy = policy_from_extra(extra, "group")
    return dm_policy if dm_policy.enabled and not group_policy.enabled else group_policy


__all__ = ["SlashAccessPolicy", "policy_for_source", "policy_from_extra"]



def policy_for_runner_source(runner: Any, source: Any) -> SlashAccessPolicy:
    """Slash-gating policy from the profile whose bot received ``source``.

    Under ``multiplex_profiles`` the runner's ``config`` is the launch profile's alone, so a platform
    configured only in a secondary profile had no PlatformConfig there and every caller came back
    unrestricted (#121705). Authorization follows the transport (bot-owning) profile, the owner
    ``RoutingIdentity.authorization_home`` already names; a served profile whose config is not cached
    fails closed instead of inheriting the launch profile's open policy.
    """
    config = getattr(runner, "config", None)
    if not getattr(config, "multiplex_profiles", False) or source is None:
        return policy_for_source(config, source)
    from gateway.session_identity import identity_of

    identity = identity_of(source)
    owner = identity.transport_profile if identity is not None else getattr(source, "profile", None)
    primary = getattr(runner, "_primary_profile_name", None) or "default"
    if owner and owner != primary:
        config = (getattr(runner, "_profile_configs", None) or {}).get(owner)
        if config is None:
            return _FAIL_CLOSED_POLICY
    return policy_for_source(config, source)


def resolve_exec_approval_admin_gate(config_extra: Any) -> tuple[bool, set]:
    """Resolve the exec-approval button admin gate from a platform's ``extra``; returns
    ``(require_admin, admin_user_ids)``. Default OFF (any admitted user may resolve an approval
    button). When ``require_admin_for_exec_approval`` is true only ``allow_admin_from`` ids may;
    on with no admins -> ``(True, set())`` (fail closed — the caller logs it).
    """
    extra = config_extra if isinstance(config_extra, dict) else {}
    raw_toggle = extra.get("require_admin_for_exec_approval", False)
    require_admin = str(raw_toggle).strip().lower() in {"true", "1", "yes"}
    if not require_admin:
        return (False, set())
    return (True, set(_coerce_id_list(extra.get("allow_admin_from"))))


def exec_approval_tap_allowed(config_extra: Any, user_id: Any, platform: str) -> bool:
    """Whether *user_id* may resolve an exec-approval button under the admin gate above (call after
    the platform's own admission check). Logs when the gate is on with no admins configured."""
    require_admin, admin_ids = resolve_exec_approval_admin_gate(config_extra)
    if not require_admin:
        return True
    uid = str(user_id or "").strip()
    if uid and uid in admin_ids:
        return True
    if not admin_ids:
        logger.warning(
            "[%s] require_admin_for_exec_approval is enabled but no admins are configured "
            "(allow_admin_from is empty) — exec approval buttons are disabled for everyone. Add admin "
            "user IDs under the %s platform's allow_admin_from, or disable the toggle.",
            platform, platform.lower(),
        )
    return False

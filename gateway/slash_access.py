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

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, Optional

# Read-only floor every allowed user keeps under gating, so a non-admin can still discover what
# they can do. ``user_allowed_commands`` only adds to this set, never restricts it.
_ALWAYS_ALLOWED_FOR_USERS: FrozenSet[str] = frozenset({"help", "whoami"})

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
    admin_user_ids: FrozenSet[str]
    user_allowed_commands: FrozenSet[str]

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


def _coerce_list(raw: Any, normalize: Callable[[str], str] = str) -> FrozenSet[str]:
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


def _coerce_id_list(raw: Any) -> FrozenSet[str]:
    """Normalize an admin/user id list into a frozenset of strings."""
    return _coerce_list(raw)


def _coerce_command_list(raw: Any) -> FrozenSet[str]:
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
    scope = "dm" if chat_type and chat_type.lower() in _DM_CHAT_TYPES else "group"
    return policy_from_extra(extra, scope)


__all__ = ["SlashAccessPolicy", "policy_from_extra", "policy_for_source"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Tuple  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----

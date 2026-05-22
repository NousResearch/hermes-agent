"""Per-platform slash command access control.

This module sits beside the existing per-platform allowlist (``allow_from``)
and adds a second axis: of the users who are *allowed to talk to the
gateway*, which ones can run *which slash commands*.

Two lists per platform scope (DM vs group, mirroring ``allow_from`` vs
``group_allow_from``):

  - ``allow_admin_from``      — user IDs that get every registered slash
                                command (built-in + plugin-registered).
  - ``user_allowed_commands`` — slash command names non-admin users may
                                run. Empty / unset → non-admins get no
                                slash commands.

Backward compatibility:

  If ``allow_admin_from`` is not set for a scope, slash command gating
  is disabled entirely for that scope. Every allowed user can run every
  slash command, exactly like before. This means existing installs are
  unaffected until an operator opts in by listing at least one admin.

The gate is applied at the slash command dispatch site in
``gateway/run.py`` so it covers BOTH built-in and plugin-registered
commands via the live registry. Gating slash commands does not affect
plain chat — non-admin users can still talk to the agent normally,
they just can't trigger commands outside ``user_allowed_commands``.

Authored as a slimmed-down salvage of PR #4443's permission tiers
(co-authored by @ReqX). The full tier system, audit log, usage
tracking, rate limiting, and tool filtering from that PR are not
included here — only the slash-command access split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Optional, Tuple


# Slash commands that MUST stay reachable for any allowed user, even when
# slash gating is enabled and the user has no commands listed. Without this
# carve-out, a non-admin user has no way to discover what they can or
# can't do (``/help``, ``/whoami``) and no way to see what state the agent
# is in (``/status``). These mirror the smallest set of read-only commands
# we'd hand to a guest. Operators can still narrow this further by writing
# their own ``user_allowed_commands`` (this set is only the implicit
# fallback floor — anything in ``user_allowed_commands`` overrides it
# additively, never restrictively).
_ALWAYS_ALLOWED_FOR_USERS: FrozenSet[str] = frozenset({
    "help",
    "whoami",
})


@dataclass(frozen=True)
class SlashAccessPolicy:
    """Resolved access policy for a single (platform, scope) pair.

    ``scope`` is ``"dm"`` for direct messages and ``"group"`` for groups,
    channels, threads, and any other multi-user context. The mapping from
    SessionSource.chat_type → scope happens in ``policy_for_source``.
    """

    enabled: bool                      # gating active for this scope?
    admin_user_ids: FrozenSet[str]
    user_allowed_commands: FrozenSet[str]

    def is_admin(self, user_id: Optional[str]) -> bool:
        if not self.enabled:
            # Gating disabled → treat every allowed user as admin so
            # downstream code can keep using ``is_admin`` / ``can_run``
            # uniformly.
            return True
        if not user_id:
            return False
        return str(user_id) in self.admin_user_ids

    def can_run(self, user_id: Optional[str], canonical_cmd: str) -> bool:
        if not self.enabled:
            return True
        if self.is_admin(user_id):
            return True
        if not canonical_cmd:
            return False
        if canonical_cmd in _ALWAYS_ALLOWED_FOR_USERS:
            return True
        return canonical_cmd in self.user_allowed_commands


_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})


def _coerce_id_list(raw: Any) -> FrozenSet[str]:
    """Normalize a YAML-loaded admin/user list into a frozenset of strings.

    Accepts ``None``, list, tuple, or comma-separated string. Stringifies
    each entry and strips whitespace; empty entries are dropped.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        # single scalar (int user id, etc.)
        items = (raw,)
    out: list[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return frozenset(out)


def _coerce_command_list(raw: Any) -> FrozenSet[str]:
    """Normalize a slash command allowlist.

    Strips leading slashes so YAML can read either ``["help", "status"]``
    or ``["/help", "/status"]``. Lowercase canonicalization matches how
    ``resolve_command()`` stores names.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        items = (raw,)
    out: list[str] = []
    for it in items:
        s = str(it).strip().lstrip("/").lower()
        if s:
            out.append(s)
    return frozenset(out)


def _scope_for_chat_type(chat_type: Optional[str]) -> str:
    if chat_type and chat_type.lower() in _DM_CHAT_TYPES:
        return "dm"
    return "group"


def _platform_extra(platform_config: Any) -> dict:
    """Return the ``extra`` dict from a PlatformConfig-like object.

    Defensively handles None and non-PlatformConfig shapes so calling
    code can stay simple.
    """
    if platform_config is None:
        return {}
    extra = getattr(platform_config, "extra", None)
    if isinstance(extra, dict):
        return extra
    if isinstance(platform_config, dict):
        # Some test harnesses pass dicts directly.
        return platform_config
    return {}


def _keys_for_scope(scope: str) -> Tuple[str, str]:
    """Return (admin_key, user_cmd_key) names for a scope."""
    if scope == "group":
        return ("group_allow_admin_from", "group_user_allowed_commands")
    return ("allow_admin_from", "user_allowed_commands")


def policy_from_extra(extra: dict, scope: str) -> SlashAccessPolicy:
    """Build a policy from a platform's ``extra`` dict for one scope.

    DM scope falls back to group scope keys ONLY for ``user_allowed_commands``
    when the DM scope didn't specify its own. This keeps the common case
    (operator wants the same command set DM and group) ergonomic without
    forcing duplication. Admin lists are NOT cross-scope: an admin in
    DMs is not implicitly an admin in a group.
    """
    admin_key, cmd_key = _keys_for_scope(scope)
    admin_ids = _coerce_id_list(extra.get(admin_key))
    cmds = _coerce_command_list(extra.get(cmd_key))

    if scope == "dm" and not cmds:
        # DM didn't specify — let group's user_allowed_commands fall through
        # so operators only need to list it once if it's the same.
        cmds = _coerce_command_list(extra.get("group_user_allowed_commands"))

    enabled = bool(admin_ids)
    return SlashAccessPolicy(
        enabled=enabled,
        admin_user_ids=admin_ids,
        user_allowed_commands=cmds,
    )


def policy_for_source(gateway_config: Any, source: Any) -> SlashAccessPolicy:
    """Resolve the access policy for a SessionSource.

    Returns a "disabled" policy (gating off, allow everything) when:
      - gateway_config is None
      - the platform has no PlatformConfig
      - the platform's PlatformConfig has no admin list set for the scope

    Callers should treat the returned policy as authoritative for slash
    command gating only. It does not gate plain chat messages.
    """
    if gateway_config is None or source is None:
        return SlashAccessPolicy(
            enabled=False,
            admin_user_ids=frozenset(),
            user_allowed_commands=frozenset(),
        )
    platforms = getattr(gateway_config, "platforms", None)
    platform_config = None
    if platforms is not None:
        try:
            platform_config = platforms.get(source.platform)
        except Exception:
            platform_config = None
    extra = _platform_extra(platform_config)
    scope = _scope_for_chat_type(getattr(source, "chat_type", None))
    return policy_from_extra(extra, scope)


__all__ = [
    "SlashAccessPolicy",
    "policy_from_extra",
    "policy_for_source",
]

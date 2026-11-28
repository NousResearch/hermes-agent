"""Permission tiers for gateway users, mapped to tools.

Issue #527 asks for Owner/Admin/User/Guest roles. The gateway already has two
axes: ``allow_from`` decides *who* may talk, and ``gateway/slash_access.py``
decides which slash commands an admitted user may run. What was missing is a
third, optional axis: which tools an admitted user may invoke through chat.

Design constraints:

* **Opt-in.** With no role configured the policy answers "allowed" for every
  tool, so existing installs are untouched. The tiers only take effect when an
  operator actually assigns a role to a user or sets a default tier.
* **Deny by default inside a tier.** An unknown tool is refused for any tier
  below ``owner``. A new tool stays unavailable to restricted users until an
  operator adds it, matching ``tools/hosted_tool_policy.py``.
* **Owner is the full key.** ``owner`` (and an unconfigured install) allow
  every tool, including terminal and file writes.

The module is deliberately pure: it resolves a tier from a config mapping and
answers one question, ``can_use(role, tool)``. Wiring it into the dispatch path
is a separate, reviewable change.
"""

from __future__ import annotations

from typing import Mapping, Optional

OWNER = "owner"
ADMIN = "admin"
USER = "user"
GUEST = "guest"

ROLES = frozenset({OWNER, ADMIN, USER, GUEST})

# Tools every admitted user keeps, regardless of tier. Read-only or self-scoped:
# discovering what the agent can do never needs write or host access.
_BASE_ALLOWED = frozenset({
    "help", "whoami",
})

# Tools an admin may use in addition to the base set. Admins operate the agent
# (cron, sessions, skills) but do not get raw host shell or arbitrary writes.
_ADMIN_ALLOWED = _BASE_ALLOWED | frozenset({
    "cron", "sessions", "skills", "memory", "todo",
})

# Tools a regular user may use: base plus ordinary conversation tools.
_USER_ALLOWED = _BASE_ALLOWED | frozenset({
    "web_search", "web_extract", "read_file", "search_files",
})

# Guests keep only the base floor.
_GUEST_ALLOWED = _BASE_ALLOWED

_TIER_TOOLS = {
    OWNER: None,  # None means unrestricted.
    ADMIN: _ADMIN_ALLOWED,
    USER: _USER_ALLOWED,
    GUEST: _GUEST_ALLOWED,
}


def normalize_role(raw: Optional[object]) -> Optional[str]:
    """Normalize a configured role. Returns None when no role is configured."""
    if raw is None:
        return None
    role = str(raw).strip().lower()
    if not role:
        return None
    aliases = {
        "owner": OWNER, "admin": ADMIN, "administrator": ADMIN,
        "user": USER, "member": USER, "guest": GUEST, "visitor": GUEST,
    }
    return aliases.get(role, role)


def can_use(role: Optional[str], tool: str) -> bool:
    """Whether a user with ``role`` may invoke ``tool``.

    ``role=None`` means no tier configured at all: unrestricted, for backward
    compatibility. A configured but unknown role is refused (fail closed).
    """
    if role is None:
        return True
    normalized = normalize_role(role)
    if normalized not in _TIER_TOOLS:
        return False
    allowed = _TIER_TOOLS[normalized]
    if allowed is None:
        return True
    return bool(tool) and tool in allowed


def resolve_role(
    user_id: Optional[str],
    roles: Mapping[str, str],
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """Resolve one user's role from a config mapping.

    Roles are keyed by platform-qualified user id as used by ``allow_from``;
    callers pass the same normalized id. A missing entry falls back to
    ``default``. No entry and no default means unrestricted (None).
    """
    if user_id and str(user_id) in roles:
        return normalize_role(roles[str(user_id)])
    return normalize_role(default)


def refine_allowed_tools(role: Optional[str], tools: list[str]) -> list[str]:
    """Filter a tool list down to what ``role`` may use.

    Owner/unconfigured returns the list unchanged so ordering is preserved.
    """
    if role is None or normalize_role(role) == OWNER:
        return list(tools)
    return [tool for tool in tools if can_use(role, tool)]

"""Per-user tool RBAC for messaging platforms.

Sits beside the chat allowlist and the slash-command tiers (see
``gateway/slash_access.py``) and adds a third axis: which *tools* an
identified platform user may invoke, expressed as named roles that grant
toolset categories.

Two config keys in a platform's ``extra`` block:

  - ``roles``       — optional map ``{role_name: {toolsets: [...]}}``. Merges
                      over the built-in defaults; a custom role with a
                      built-in name overrides it. ``"*"`` grants all toolsets;
                      ``[]`` grants none. Patterns containing ``*`` (e.g.
                      ``mcp-*``) are glob-matched against concrete toolsets.
  - ``user_roles``  — map ``{user_id: role_name}``. Its presence ACTIVATES
                      RBAC for the platform. When active it is the sole
                      authorization source: a user with a role may chat and
                      gets that role's toolsets; a user with no role is denied
                      entirely.

Backward compatibility: when ``user_roles`` is absent/empty, the policy is
disabled and every method defers (``is_authorized`` → True, ``allowed_toolsets``
→ everything). Existing installs are unaffected until an operator opts in.

Fail-closed: a ``user_roles`` entry naming an undefined role denies that user
and logs a config error.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

logger = logging.getLogger(__name__)

# Built-in roles. Operators get these without writing a ``roles:`` block.
BUILTIN_ROLES: Dict[str, FrozenSet[str]] = {
    "admin": frozenset({"*"}),
    # ``operator`` intentionally excludes ``terminal``: notion/file/web/browser
    # cover its work without a shell, and dropping the shell stops an operator
    # (including channel-role posters in issue-tracking channels) from reading
    # host secrets via ``printenv`` / ``cat ~/.hermes/.env``. ``session_search``
    # is granted so operators can recall prior work in their own session scope.
    # ``stripe`` / ``mcp-stripe``: grant BOTH forms of the Stripe MCP toolset.
    # MCP servers register their canonical toolset as ``mcp-{server}`` (here
    # ``mcp-stripe``) and a bare-name alias (``stripe``) via
    # ``registry.register_toolset_alias`` — see ``tools/mcp_tool.py``. The
    # toolset FILTER (``filter_enabled_toolsets`` / ``allowed_toolsets``)
    # typically sees whatever name ``enabled_toolsets`` config uses, commonly
    # the alias ``stripe``. But the execution BACKSTOP
    # (``denial_for_current_tool`` -> ``_toolset_for_tool`` ->
    # ``registry.get_toolset_for_tool``) reads ``ToolEntry.toolset`` directly,
    # which is set to the CANONICAL ``mcp-stripe`` at registration time — it
    # does NOT resolve through the alias table. Granting only ``stripe`` would
    # pass the filter but fail the backstop for real Stripe MCP tool calls, so
    # both names are granted here (harmless — an unregistered MCP toolset
    # string in a frozenset is inert until that server exists).
    "operator": frozenset(
        {"file", "web", "browser", "vision", "memory", "delegation",
         "notion", "session_search", "stripe", "mcp-stripe"}
    ),
    "readonly": frozenset({"web", "vision", "session_search", "memory"}),
    "chat_only": frozenset(),
}

# Toolsets every VALID-role user gets regardless of their role's grant, so
# even a restricted (e.g. chat_only) user's agent can still ask clarifying
# questions, track its work, and acknowledge/close out a Slack turn.
# Mirrors slash_access._ALWAYS_ALLOWED_FOR_USERS.
#
# ``slack`` (slack_react + turn_end) is a floor capability, not a privilege:
# reacting to a message and ending a turn silently is how the bot acknowledges
# ANY user's message in a quiet channel — including a readonly/view-only
# user's. Gating it behind a role makes the bot fall back to posting text for
# everyone but admins, which defeats quiet channels. On non-Slack platforms the
# ``slack`` toolset isn't in the enabled set, so listing it here is inert there
# (allowed_toolsets only ever intersects with what the platform actually offers).
#
# ``ownership`` (the ``ownership`` tool) is a floor for the same reason: the
# ownership registry is administered by ``hermes own``, a CLI, and reaching a
# CLI needs ``terminal`` — which only ``admin`` holds. That made claiming an
# unowned automation, handing off your own, or even asking who owns something
# admin-only, so the claim nudge told ordinary users to run a command they
# cannot run. The tool self-enforces owner-or-admin on every mutation
# (agent/automation_ownership._require_owner_or_admin) and refuses to act
# without a human identity, so granting it broadly costs nothing. Gating it
# behind a role instead would reproduce the bug the moment an operator writes a
# custom role with ``cronjob`` or ``skills`` and forgets ``ownership``.
#
# Note: this does NOT apply to users with no role / an undefined role — they
# get nothing (deny-until-assigned).
FLOOR_TOOLSETS: FrozenSet[str] = frozenset({"clarify", "todo", "slack", "ownership"})


def _coerce_str(value: Any) -> str:
    return str(value).strip()


def _coerce_user_roles(raw: Any) -> Dict[str, str]:
    """Normalize ``{user_id: role_name}`` from YAML (int ids, padding, case)."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        uid = _coerce_str(k)
        role = _coerce_str(v).lower()
        if uid and role:
            out[uid] = role
    return out


def _coerce_channel_roles(raw: Any) -> Dict[str, str]:
    """Normalize ``{chat_id: role_name}`` — chat→role bindings that grant a
    fixed role to EVERY poster in that channel. The chat id is kept verbatim
    (Slack/Discord ids are case-sensitive); the role name is lowercased to match
    the ``roles`` table.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        cid = _coerce_str(k)
        role = _coerce_str(v).lower()
        if cid and role:
            out[cid] = role
    return out


def _coerce_extends(raw: Any) -> List[str]:
    """Normalize a role's ``extends`` value to a list of parent role names.

    Accepts a single name, a comma-separated string, or a sequence. Names are
    lowercased to match the ``roles`` table. Anything else logs and yields no
    parents — an unusable ``extends`` must never silently widen a grant.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [s for s in (p.strip().lower() for p in raw.split(",")) if s]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return [n for n in (_coerce_str(p).lower() for p in raw) if n]
    logger.warning(
        "tool_access: unexpected extends type %s — ignoring",
        type(raw).__name__,
    )
    return []


def _coerce_roles(raw: Any) -> Dict[str, FrozenSet[str]]:
    """Normalize a ``roles`` block, merged over the built-in defaults.

    Two passes: collect each role's OWN toolsets, then resolve ``extends``
    against the fully-populated map so a role may extend one defined later in
    the YAML.
    """
    own: Dict[str, FrozenSet[str]] = dict(BUILTIN_ROLES)
    parents: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return own

    for name, body in raw.items():
        role_name = _coerce_str(name).lower()
        if not role_name:
            continue
        toolsets: Any = None
        if isinstance(body, dict):
            toolsets = body.get("toolsets")
            parents[role_name] = _coerce_extends(body.get("extends"))
        elif isinstance(body, (list, tuple)):
            toolsets = body
        if isinstance(toolsets, str):
            items = [s for s in (s.strip() for s in toolsets.split(",")) if s]
        elif isinstance(toolsets, (list, tuple, set, frozenset)):
            items = toolsets
        elif toolsets is not None:
            logger.warning(
                "tool_access: unexpected toolsets type %s for role '%s' — ignoring",
                type(toolsets).__name__, role_name,
            )
            items = []
        else:
            items = []
        own[role_name] = frozenset(
            _coerce_str(t).lower() for t in items if _coerce_str(t)
        )

    if not any(parents.values()):
        return own

    return {
        role: _resolve_role_extends(role, own, parents, set())
        for role in own
    }


def _resolve_role_extends(
    role: str,
    own: Dict[str, FrozenSet[str]],
    parents: Dict[str, List[str]],
    resolving: set,
) -> FrozenSet[str]:
    """Union of *role*'s own toolsets and everything it transitively extends.

    ``resolving`` carries the current DFS path so a cycle breaks instead of
    recursing forever. Deliberately unmemoized: role graphs are a handful of
    entries, and a memo populated mid-cycle would depend on which role the
    caller happened to start from.
    """
    if role in resolving:
        logger.warning(
            "tool_access: role inheritance cycle at '%s' — ignoring this edge",
            role,
        )
        return frozenset()
    resolving.add(role)
    try:
        acc = set(own.get(role, frozenset()))
        for parent in parents.get(role, []):
            if parent not in own:
                logger.warning(
                    "tool_access: role '%s' extends undefined role '%s' — ignoring",
                    role, parent,
                )
                continue
            acc |= _resolve_role_extends(parent, own, parents, resolving)
        return frozenset(acc)
    finally:
        resolving.discard(role)


def _granted(role_toolsets: FrozenSet[str], toolset: str) -> bool:
    """True if ``toolset`` is granted by ``role_toolsets`` (exact, ``*``, glob)."""
    toolset = toolset.lower()
    if "*" in role_toolsets:
        return True
    if toolset in role_toolsets:
        return True
    for pattern in role_toolsets:
        if "*" in pattern and fnmatch.fnmatchcase(toolset, pattern):
            return True
    return False


@dataclass(frozen=True)
class ToolAccessPolicy:
    """Resolved RBAC policy for a single platform.

    When ``enabled`` is False the policy defers entirely so legacy auth and
    the unfiltered toolset apply unchanged.
    """

    enabled: bool
    user_roles: Mapping[str, str]
    roles: Mapping[str, FrozenSet[str]]
    # chat_id → role: grants a fixed role to EVERY poster in that channel, so
    # the bot can serve issue-tracking channels where any teammate may report.
    # Additive — UNIONed with the poster's own role; never reduces access.
    # Honored only while RBAC is active (it does not by itself enable RBAC).
    channel_roles: Mapping[str, str] = field(
        default_factory=lambda: types.MappingProxyType({})
    )

    def role_for(self, user_id: Optional[str]) -> Optional[str]:
        if not self.enabled or not user_id:
            return None
        return self.user_roles.get(str(user_id))

    def _grant_for_role(
        self, role: Optional[str], *, kind: str, owner: Optional[str]
    ) -> Optional[FrozenSet[str]]:
        """Resolve a role name to its toolset grant, or None when unassigned or
        undefined. Logs an ERROR (once per call) when *role* is named but not
        defined in ``self.roles`` so every entry point logs consistently.
        """
        if role is None:
            return None
        if role not in self.roles:
            logger.error(
                "tool_access: %s %s assigned undefined role '%s' — denying",
                kind, owner, role,
            )
            return None
        return self.roles[role]

    def _effective_grant(
        self, user_id: Optional[str], chat_id: Optional[str] = None
    ) -> Optional[FrozenSet[str]]:
        """Union of the user's own role grant and, when *chat_id* names a
        ``channel_roles`` channel, that channel's role grant. None when neither
        axis grants anything (deny-until-assigned). A channel role authorizes
        and equips ANY poster in that channel; a user with their own role keeps
        it everywhere and simply gains the channel grant on top while in-channel.
        """
        grants = []
        user_grant = self._grant_for_role(
            self.user_roles.get(str(user_id)) if user_id else None,
            kind="user", owner=user_id,
        )
        if user_grant is not None:
            grants.append(user_grant)
        channel_grant = self._grant_for_role(
            self.channel_roles.get(str(chat_id)) if chat_id else None,
            kind="channel", owner=chat_id,
        )
        if channel_grant is not None:
            grants.append(channel_grant)
        if not grants:
            return None
        if len(grants) == 1:
            return grants[0]
        return frozenset().union(*grants)

    def grant_for(
        self, user_id: Optional[str], chat_id: Optional[str] = None
    ) -> Optional[FrozenSet[str]]:
        """Public accessor for the user's effective toolset grant.

        Returns None when the user resolves to no role (roleless / undefined
        role) — distinct from a defined role that grants nothing (e.g.
        ``chat_only`` returns an empty frozenset). Used by the cron toolset
        ceiling to decide whether a cap applies at all.
        """
        return self._effective_grant(user_id, chat_id)

    def is_authorized(
        self, user_id: Optional[str], chat_id: Optional[str] = None
    ) -> bool:
        if not self.enabled:
            return True  # defer to legacy auth
        return self._effective_grant(user_id, chat_id) is not None

    def allowed_toolsets(
        self,
        user_id: Optional[str],
        all_toolsets: FrozenSet[str],
        chat_id: Optional[str] = None,
    ) -> FrozenSet[str]:
        if not self.enabled:
            return frozenset(all_toolsets)
        grant = self._effective_grant(user_id, chat_id)
        if grant is None:
            return frozenset()
        return frozenset(
            t for t in all_toolsets if _granted(grant, t) or t in FLOOR_TOOLSETS
        )

    def can_use_tool(
        self,
        user_id: Optional[str],
        toolset: Optional[str],
        chat_id: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return True
        grant = self._effective_grant(user_id, chat_id)
        if grant is None or not toolset:
            return False
        return _granted(grant, toolset) or toolset in FLOOR_TOOLSETS


def policy_from_extra(extra: Any) -> ToolAccessPolicy:
    """Build a policy from a platform's ``extra`` dict."""
    if not isinstance(extra, dict):
        extra = {}
    user_roles = _coerce_user_roles(extra.get("user_roles"))
    roles = _coerce_roles(extra.get("roles"))
    channel_roles = _coerce_channel_roles(extra.get("channel_roles"))
    return ToolAccessPolicy(
        # Activation is still keyed to user_roles alone — channel_roles is an
        # additive carve-out within an already-active policy and must not by
        # itself flip RBAC on (which would deny roleless users everywhere).
        enabled=bool(user_roles),
        user_roles=types.MappingProxyType(user_roles),
        roles=types.MappingProxyType(roles),
        channel_roles=types.MappingProxyType(channel_roles),
    )


def _platform_extra(platform_config: Any) -> dict:
    if platform_config is None:
        return {}
    extra = getattr(platform_config, "extra", None)
    if isinstance(extra, dict):
        return extra
    # Some test harnesses pass dicts directly.
    if isinstance(platform_config, dict):
        return platform_config
    return {}


def policy_for_source(gateway_config: Any, source: Any) -> ToolAccessPolicy:
    """Resolve the policy for a SessionSource's platform."""
    if gateway_config is None or source is None:
        return ToolAccessPolicy(
            enabled=False,
            user_roles=types.MappingProxyType({}),
            roles=types.MappingProxyType(dict(BUILTIN_ROLES)),
            channel_roles=types.MappingProxyType({}),
        )
    platforms = getattr(gateway_config, "platforms", None)
    platform_config = None
    if platforms is not None:
        try:
            platform_config = platforms.get(source.platform)
        except Exception:
            platform_config = None
    return policy_from_extra(_platform_extra(platform_config))


# ---------------------------------------------------------------------------
# Dispatch-backstop helper. Reads identity from session contextvars, resolves
# the policy for that platform (cached on config mtime), maps the tool to its
# toolset via the registry, and returns a denial message or None.
# ---------------------------------------------------------------------------

# Keyed by the RESOLVED config.yaml path (multiplex fix 1): one entry per
# profile home, each holding ``(fp, config)`` with ``fp = (mtime_ns, size)``.
# A single global slot keyed only on ``fp`` served one profile's RBAC to
# another, because ``hermes profile create --clone`` preserves mtime.
_config_cache: Dict[str, Any] = {}
_config_cache_lock = threading.Lock()


def _current_identity():
    """Return (user_id, platform_name) from session contextvars, or (None, None)."""
    try:
        from gateway.session_context import get_session_env
        uid = get_session_env("HERMES_SESSION_USER_ID", "") or None
        plat = get_session_env("HERMES_SESSION_PLATFORM", "") or None
        return uid, plat
    except Exception as err:
        logger.debug("tool_access: _current_identity failed: %s", err)
        return None, None


def _current_chat_id() -> Optional[str]:
    """Return the active chat/channel id from session contextvars, or None.

    Lets the execution backstop honor ``channel_roles`` (the channel a tool
    call runs in determines the effective role for delegated/sandboxed calls).
    """
    try:
        from gateway.session_context import get_session_env
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or None
    except Exception as err:
        logger.debug("tool_access: _current_chat_id failed: %s", err)
        return None


def _toolset_for_tool(tool_name: str) -> Optional[str]:
    try:
        from tools.registry import registry
        return registry.get_toolset_for_tool(tool_name)
    except Exception as err:
        logger.debug("tool_access: _toolset_for_tool failed: %s", err)
        return None


def _load_config_cached():
    """Load gateway config, memoized per config.yaml path on (mtime, size)."""
    try:
        from gateway.config import load_gateway_config, get_hermes_home
        cfg_file = get_hermes_home() / "config.yaml"
        try:
            st = cfg_file.stat()
            fp = (st.st_mtime_ns, st.st_size)
        except OSError:
            fp = None
        try:
            key = str(cfg_file.resolve())
        except OSError:
            key = str(cfg_file)
        with _config_cache_lock:
            entry = _config_cache.get(key)
            if entry is None or entry[0] != fp or entry[1] is None:
                entry = (fp, load_gateway_config())
                _config_cache[key] = entry
            return entry[1]
    except Exception:
        return None


def _policy_for_current_platform(platform_name: str) -> Optional[ToolAccessPolicy]:
    config = _load_config_cached()
    if config is None:
        return None
    try:
        from gateway.config import Platform
        platform = Platform(platform_name)
    except Exception:
        return None
    platforms = getattr(config, "platforms", {}) or {}
    return policy_from_extra(_platform_extra(platforms.get(platform)))


def policy_for_platform(platform_name: str) -> Optional[ToolAccessPolicy]:
    """Resolve the RBAC policy for a named platform from the cached gateway
    config. Returns None when config is unavailable or the platform is invalid.

    Public wrapper over the dispatch backstop's resolver, for callers (e.g. the
    cron toolset ceiling) that resolve a policy outside a live inbound request.
    """
    return _policy_for_current_platform(platform_name)


def rbac_active_anywhere() -> bool:
    """True if RBAC is active for at least one configured platform.

    Used to suppress cron ownerless-elevated audit noise on installs that never
    opted into RBAC: when no platform sets ``user_roles``, the cron toolset
    ceiling is inert and behavior must be byte-for-byte unchanged (no new audit
    writes). Cached via ``_load_config_cached`` (config-mtime memoized).
    """
    config = _load_config_cached()
    if config is None:
        return False
    platforms = getattr(config, "platforms", {}) or {}
    for pconfig in platforms.values():
        try:
            if policy_from_extra(_platform_extra(pconfig)).enabled:
                return True
        except Exception:
            continue
    return False


def denial_for_current_tool(tool_name: str) -> Optional[str]:
    """Return a denial message if the current user may not use ``tool_name``,
    else None. Fail-open on any internal error (RBAC is a backstop; the
    toolset filter is the primary control)."""
    try:
        user_id, platform_name = _current_identity()
        if not user_id or not platform_name:
            return None  # CLI / system / cron context — no gating
        policy = _policy_for_current_platform(platform_name)
        if policy is None or not policy.enabled:
            return None
        toolset = _toolset_for_tool(tool_name)
        if toolset is None:
            return None  # tool not in registry → not gated by toolset RBAC
        chat_id = _current_chat_id()
        if policy.can_use_tool(user_id, toolset, chat_id):
            return None
        logger.info(
            "tool_access: denied tool '%s' (toolset '%s') for %s on %s (chat %s)",
            tool_name, toolset, user_id, platform_name, chat_id,
        )
        return (
            f"⛔ You are not permitted to use '{tool_name}' here. "
            "Ask an admin to adjust your role if you need this capability."
        )
    except Exception as err:  # pragma: no cover - defensive
        logger.debug("tool_access backstop error: %s", err)
        return None


def filter_enabled_toolsets(source, enabled_toolsets, gateway_config=None):
    """Intersect ``enabled_toolsets`` with the source user's role grant.

    Returns the input (as a sorted list) unchanged when RBAC is disabled for
    the platform. Resolves the typed gateway config itself when not provided.
    Fail-open: on any error returns the input list so a resolution failure
    can't silently strip a user's tools (the dispatch backstop remains the
    hard control).
    """
    base = list(enabled_toolsets or [])
    try:
        cfg = gateway_config if gateway_config is not None else _load_config_cached()
        if cfg is None:
            return sorted(base)
        policy = policy_for_source(cfg, source)
        if not policy.enabled:
            return sorted(base)
        user_id = getattr(source, "user_id", None)
        chat_id = getattr(source, "chat_id", None)
        allowed = policy.allowed_toolsets(user_id, frozenset(base), chat_id)
        return sorted(allowed)
    except Exception as err:  # pragma: no cover - defensive
        logger.debug("tool_access filter error: %s", err)
        return sorted(base)


__all__ = [
    "BUILTIN_ROLES",
    "FLOOR_TOOLSETS",
    "ToolAccessPolicy",
    "policy_from_extra",
    "policy_for_source",
    "policy_for_platform",
    "rbac_active_anywhere",
    "denial_for_current_tool",
    "filter_enabled_toolsets",
]

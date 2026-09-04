"""User-authorization methods for ``GatewayRunner``.

Extracted from ``gateway/run.py`` as part of the god-file decomposition campaign
(``~/.hermes/plans/god-file-decomposition.md``, Phase 3 mechanical mixin lifts).
This mixin holds the inbound-message authorization cluster: whether a user/chat
is allowed to talk to the agent, the per-adapter DM policy, and the
unauthorized-DM behavior.

Behavior-neutral: every method is lifted verbatim from ``GatewayRunner``.
``self.*`` calls resolve unchanged via the MRO. Neutral dependencies import at
module top; the module-level ``logger`` is imported lazily inside the one method
that uses it (``from gateway.run import logger`` resolves at call time, when
``gateway.run`` is fully loaded) so this module never imports ``gateway.run`` at
import time -> no import cycle. The lazy import preserves the exact logger name
(``"gateway.run"``) so log records are unchanged.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.whatsapp_identity import (
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)


def _platform_gate_env(name: str, default: str = "") -> str:
    """Read a platform allow/deny gate env var with per-profile isolation.

    When a profile secret scope is installed AND multiplexing is active, a
    key absent from the scope returns ``default`` instead of falling through
    to ``os.environ``. Under multiplex the process env may hold ANOTHER
    profile's first-writer-bridged value (the YAML→env bridges in the
    Discord/Telegram adapters' ``_apply_yaml_config`` are first-writer-wins),
    so falling through would leak profile A's allowlist into profile B
    (issue #72348). Single-profile deployments — no scope installed, or
    multiplex off — behave exactly like the legacy ``os.getenv`` read.
    """
    if not name:
        return default
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        if scope is not None and is_multiplex_active():
            val = scope.get(name)
            if val is None:
                return default
            return str(val).strip()
    except Exception:
        pass
    return (os.getenv(name) or default).strip()


def _auth_env(name: str, default: str = "") -> str:
    """Read allowlist/auth env with per-profile isolation under multiplex.

    Same rules as ``_platform_gate_env``: a scoped miss under multiplex
    returns ``default`` and does not fall through to ``os.environ``. The
    process env may hold another profile's first-writer-bridged value, so
    a fallthrough would leak allowlists and allow-all flags across profiles
    (issue #72348). Single-profile deployments keep the legacy
    ``os.getenv`` read.
    """
    return _platform_gate_env(name, default)


def _platform_declares_allowed_users_env(platform) -> bool:
    """Whether a plugin platform's registry entry declares ``allowed_users_env``.

    Such platforms (Buzz, DingTalk, …) document ``PlatformConfig.extra
    .allowed_users`` as the config-file spelling of that env allowlist, so
    the live adapter's extra is a valid authorization source when the env
    var is absent (#98738 / #82871). Built-in platforms and unknown entries
    return False.
    """
    if platform is None:
        return False
    try:
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get(platform.value)
        return bool(entry and entry.allowed_users_env)
    except Exception:
        return False


def _coerce_allow_set(raw) -> set[str]:
    """Parse allowlist values from config or env var into a set of strings.

    Handles both list inputs (YAML sequences) and comma-separated string
    inputs (env vars or scalar YAML values).  A scalar string is split on
    commas so ``allow_from: "123,456"`` yields ``{"123", "456"}``, not
    ``{"1", "2", "3", ",", ...}``.
    """
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}
def _no_adapter_is_upstream(platform: Optional[Platform], profile: Optional[str]) -> bool:
    return False


def _no_adapter_enforces_own_policy(platform: Optional[Platform], profile: Optional[str]) -> bool:
    return False


def _no_adapter_dm_policy(platform: Optional[Platform], profile: Optional[str]) -> str:
    return ""


def _no_adapter_group_policy(platform: Optional[Platform], profile: Optional[str]) -> str:
    return ""


def _no_adapter_group_sender_allowlist(
    platform: Optional[Platform], chat_id: Optional[str], profile: Optional[str]
) -> bool:
    return False


def _no_adapter_group_allowed_chats(platform: Optional[Platform], profile: Optional[str]) -> set[str]:
    return set()


def _no_adapter_allow_from(platform: Optional[Platform], profile: Optional[str], is_group: bool) -> set[str]:
    return set()


def _no_adapter_dm_is_allowed(
    platform: Optional[Platform], profile: Optional[str], user_id: str
) -> Optional[bool]:
    return None


def _no_adapter_resolved_allowlist_user_ids(
    platform: Optional[Platform], profile: Optional[str]
):
    return None


def is_authorized(
    source: SessionSource,
    *,
    pairing_is_approved: Callable[[str, str], bool],
    allow_adapter_delegation: bool = True,
    adapter_authorization_is_upstream: Callable[[Optional[Platform], Optional[str]], bool] = _no_adapter_is_upstream,
    adapter_enforces_own_access_policy: Callable[[Optional[Platform], Optional[str]], bool] = _no_adapter_enforces_own_policy,
    adapter_dm_policy: Callable[[Optional[Platform], Optional[str]], str] = _no_adapter_dm_policy,
    adapter_group_policy: Callable[[Optional[Platform], Optional[str]], str] = _no_adapter_group_policy,
    adapter_group_has_sender_allowlist: Callable[[Optional[Platform], Optional[str], Optional[str]], bool] = _no_adapter_group_sender_allowlist,
    adapter_group_allowed_chats: Callable[[Optional[Platform], Optional[str]], "set[str]"] = _no_adapter_group_allowed_chats,
    adapter_allow_from: Callable[[Optional[Platform], Optional[str], bool], "set[str]"] = _no_adapter_allow_from,
    adapter_dm_is_allowed: Callable[[Optional[Platform], Optional[str], str], Optional[bool]] = _no_adapter_dm_is_allowed,
    adapter_resolved_allowlist_user_ids: Callable[[Optional[Platform], Optional[str]], Optional[object]] = _no_adapter_resolved_allowlist_user_ids,
    on_legacy_group_users_warning: Optional[Callable[[str], None]] = None,
    env_get: Callable[[str, str], str] = os.getenv,
    platform_gate_env: Callable[[str, str], str] = _platform_gate_env,
) -> bool:
    """Pure authorization decision for an inbound (or Mini-App-asserted) sender.

    Mechanically lifted out of ``GatewayAuthorizationMixin._is_user_authorized``
    — same checks, same order, same env vars — with every ``self._adapter_*``
    call replaced by an injected callable so this function has no dependency on
    a live ``GatewayRunner``/adapter registry. The five ``adapter_*`` callables
    default to "no live adapter for this platform" (mirrors
    ``_authorization_adapter`` returning ``None``), so a caller with no adapter
    to consult — e.g. the Telegram Mini App dashboard, which authorizes a
    ``initData``-verified user_id with no inbound-message adapter in the
    picture — can omit them entirely and get the same env-allowlist /
    pairing-store decision a live Telegram adapter's DM traffic would get.

    Deliberately does NOT read ``_HERMES_HOME_OVERRIDE`` or resolve
    config/profile itself — every profile-scoped fact (adapter policy, the
    pairing store lookup) is passed in by the caller. Authorization here is
    process-global by construction, not because of a check that could be
    forgotten; there is nothing profile-aware left to accidentally add.

    ``pairing_is_approved`` is a callable, not a ``PairingStore`` instance,
    so it is only invoked (and any attribute on the caller's store only
    touched) once the checks above it actually require a pairing-store
    lookup — matching the original method, which never touched
    ``self.pairing_store`` for a request an earlier branch already resolved
    (e.g. the chat-scoped ``TELEGRAM_GROUP_ALLOWED_CHATS`` allowlist above).

    ``on_legacy_group_users_warning``, if given, is invoked at most once per
    call with the comma-joined legacy chat-ID string when the
    ``TELEGRAM_GROUP_ALLOWED_USERS`` backward-compat shim (#15027) fires; the
    caller owns any "warn once" state (was ``self._warned_telegram_group_users_legacy``).

    ``env_get`` replaces every internal ``os.getenv`` call (default: real
    ``os.getenv``, so the live gateway's behavior is bit-for-bit unchanged).
    A caller with its own process — one whose ``os.environ`` was populated
    once at import time and never refreshed, e.g. a long-lived dashboard
    process checking Telegram Mini App tier access — can pass a callable
    that re-reads the relevant vars fresh per call instead of trusting a
    stale process-wide snapshot, without this function mutating
    ``os.environ`` itself or knowing anything about where those fresh
    values come from.

    ``platform_gate_env`` reads the two boolean/allowlist "gate" vars
    (chat-scoped allowlist, ``{PLATFORM}_ALLOW_BOTS``) upstream hardened to
    ``_platform_gate_env`` rather than the plain ``env_get`` every other read
    here still uses: under multiplex, a key absent from the profile's secret
    scope must return ``default`` rather than falling through to
    ``os.environ``, which can hold ANOTHER profile's first-writer-bridged
    value for the same var name (#72348) -- a false affirmative on a gate
    check leaks profile A's allowlist into profile B, unlike the other
    ``env_get`` reads here where that fallthrough is comparatively benign.
    Defaults to the real ``_platform_gate_env``, matching ``env_get``'s
    real-``os.getenv`` default, so live behavior is unchanged.
   

    ``adapter_group_allowed_chats`` and ``adapter_allow_from`` cover the two
    config.yaml-only allowlist fallbacks some adapters (e.g. Telegram) support
    via ``platforms.<platform>.extra.group_allowed_chats`` / ``allow_from`` /
    ``group_allow_from`` — configured access that has no env var equivalent.
    Both default to "nothing configured" (empty set) when there is no live
    adapter to consult, same rationale as the other ``adapter_*`` defaults.

    ``adapter_dm_is_allowed`` re-checks a DM sender against the *live* adapter's
    allowlist before honoring an ``allowlist`` intake policy (#34515): a pairing
    revoke can clear the env allowlist while a construction-time snapshot on the
    adapter would otherwise keep authorizing until restart. It returns ``None``
    for an adapter that exposes no such helper (and by default, when there is no
    live adapter at all), which preserves the historical "reached the gateway
    under allowlist policy is enough" behavior for those adapters rather than
    failing closed on callers that have no adapter to consult.

    ``adapter_resolved_allowlist_user_ids`` unions the live adapter's own
    resolved numeric allowlist (e.g. Discord's ``resolved_allowlist_user_ids()``)
    into the env-derived allowlist. Adapters that resolve username-shaped
    allowlist entries to numeric IDs at connect time keep the authoritative
    resolved set in adapter memory and mirror it into the process env, but a
    per-turn env hot-reload can restore the raw username strings from disk —
    from the second turn onward the env-derived allowlist then holds usernames
    while ``source.user_id`` is numeric, and the operator is wrongly dropped as
    unauthorized. Only consulted when an env allowlist is configured (never a
    widening of the empty-allowlist fail-closed default) and duck-typed +
    isinstance-guarded so a caller with no adapter, or a mock one, gets the
    historical env-only behavior unchanged.
    """
    # Home Assistant events are system-generated (state changes), not
    # user-initiated messages.  The HASS_TOKEN already authenticates the
    # connection, so HA events are always authorized.
    # Webhook events are authenticated via HMAC signature validation in
    # the adapter itself — no user allowlist applies.
    if source.platform in {Platform.HOMEASSISTANT, Platform.WEBHOOK}:
        return True

    # Relay (and any adapter whose authorization is enforced by a trusted
    # authenticated upstream): the Team Gateway connector authenticates this
    # gateway's WS with a per-instance secret and resolves owner-only author
    # bindings BEFORE delivering, so an inbound relay event was already
    # authorized as this instance's bound user (the author id is the one the
    # connector observed, never gateway-asserted). There is no local
    # RELAY_ALLOWED_USERS env allowlist to consult, and default-denying for
    # its absence is the bug this branch fixes. This is delegation to a
    # trusted upstream, NOT a fail-open: it fires only for an event that was
    # actually delivered over the authenticated relay WS (the transport
    # stamps ``delivered_via_upstream_relay``), or whose platform's adapter
    # explicitly declares ``authorization_is_upstream=True``; every direct
    # network-exposed adapter leaves the flag False and its events unmarked,
    # so the env-allowlist default-deny below still applies unchanged.
    #
    # The delivery marker is the PRIMARY signal: a relay *message* inbound
    # carries the UNDERLYING platform (``source.platform`` == discord/…),
    # NOT ``Platform.RELAY``, because that's what session-keying and egress
    # need — so keying authz off ``source.platform`` would miss (the relay
    # adapter is registered under ``Platform.RELAY``) and default-deny the
    # user ("Unauthorized user <id> on discord"). The adapter-flag check is
    # retained for events whose ``source.platform`` IS ``Platform.RELAY``
    # (e.g. the interaction-passthrough path).
    # ``is True`` (not just truthiness): the marker is a real bool on a
    # SessionSource, and an explicit identity check refuses to authorize a
    # non-bool stand-in (e.g. a MagicMock attribute auto-vivifies truthy in
    # tests) — defensive against accidental fail-open.
    if allow_adapter_delegation and (
        source.delivered_via_upstream_relay is True
        or adapter_authorization_is_upstream(
            source.platform,
            source.profile,
        )
    ):
        return True

    user_id = source.user_id

    # Telegram (and similar) authorize entire group/forum/channel chats
    # by chat ID via TELEGRAM_GROUP_ALLOWED_CHATS / QQ_GROUP_ALLOWED_USERS.
    # That allowlist is chat-scoped, so it must work even when
    # source.user_id is None — Telegram emits anonymous-admin posts,
    # sender_chat traffic, and channel broadcasts with no `from_user`,
    # and an operator who explicitly listed the chat expects those to
    # be honored. Run this check before the no-user-id guard below so
    # documented behavior matches reality
    # (website/docs/reference/environment-variables.md,
    # website/docs/user-guide/messaging/telegram.md).
    if source.chat_type in {"group", "forum", "channel"} and source.chat_id:
        chat_allowlist_env = {
            Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
            Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
        }.get(source.platform, "")
        if chat_allowlist_env:
            raw_chat_allowlist = platform_gate_env(chat_allowlist_env)
            if raw_chat_allowlist:
                allowed_group_ids = {
                    cid.strip()
                    for cid in raw_chat_allowlist.split(",")
                    if cid.strip()
                }
                if "*" in allowed_group_ids or source.chat_id in allowed_group_ids:
                    return True

        # Fallback: also check adapter-level config (config.yaml) for
        # platforms.<platform>.extra.group_allowed_chats. The Telegram
        # observe-unmentioned mode strips user_id from triggered group
        # messages (_apply_telegram_group_observe_attribution), so the
        # env-var-only check above misses config.yaml-configured allowlists.
        adapter_group_chats = adapter_group_allowed_chats(source.platform, source.profile)
        if adapter_group_chats and ("*" in adapter_group_chats or source.chat_id in adapter_group_chats):
            return True

    # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist (#4466).
    # Checked before the no-user-id guard below: some platforms deliver
    # bot/automation traffic with no user_id at all -- e.g. Slack Workflow
    # Builder posts arrive as subtype=bot_message with user=None -- so
    # deferring past the guard would reject them outright (the same reason
    # the chat-scoped allowlist above runs early).
    platform_allow_bots_map = {
        Platform.DISCORD: "DISCORD_ALLOW_BOTS",
        Platform.FEISHU: "FEISHU_ALLOW_BOTS",
        Platform.TELEGRAM: "TELEGRAM_ALLOW_BOTS",
        Platform.SLACK: "SLACK_ALLOW_BOTS",
    }
    if getattr(source, "is_bot", False):
        allow_bots_var = platform_allow_bots_map.get(source.platform)
        if allow_bots_var and platform_gate_env(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
            return True

    if not user_id:
        return False

    platform_env_map = {
        Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
        Platform.DISCORD: "DISCORD_ALLOWED_USERS",
        Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
        Platform.WHATSAPP_CLOUD: "WHATSAPP_CLOUD_ALLOWED_USERS",
        Platform.SLACK: "SLACK_ALLOWED_USERS",
        Platform.SIGNAL: "SIGNAL_ALLOWED_USERS",
        Platform.EMAIL: "EMAIL_ALLOWED_USERS",
        Platform.SMS: "SMS_ALLOWED_USERS",
        Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
        Platform.MATRIX: "MATRIX_ALLOWED_USERS",
        Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
        Platform.FEISHU: "FEISHU_ALLOWED_USERS",
        Platform.WECOM: "WECOM_ALLOWED_USERS",
        Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOWED_USERS",
        Platform.WEIXIN: "WEIXIN_ALLOWED_USERS",
        Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOWED_USERS",
        Platform.QQBOT: "QQ_ALLOWED_USERS",
        Platform.YUANBAO: "YUANBAO_ALLOWED_USERS",
    }
    platform_group_user_env_map = {
        Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_USERS",
    }
    platform_group_chat_env_map = {
        Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
        Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS",
    }
    platform_allow_all_map = {
        Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
        Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
        Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
        Platform.WHATSAPP_CLOUD: "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
        Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
        Platform.EMAIL: "EMAIL_ALLOW_ALL_USERS",
        Platform.SMS: "SMS_ALLOW_ALL_USERS",
        Platform.MATTERMOST: "MATTERMOST_ALLOW_ALL_USERS",
        Platform.MATRIX: "MATRIX_ALLOW_ALL_USERS",
        Platform.DINGTALK: "DINGTALK_ALLOW_ALL_USERS",
        Platform.FEISHU: "FEISHU_ALLOW_ALL_USERS",
        Platform.WECOM: "WECOM_ALLOW_ALL_USERS",
        Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOW_ALL_USERS",
        Platform.WEIXIN: "WEIXIN_ALLOW_ALL_USERS",
        Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOW_ALL_USERS",
        Platform.QQBOT: "QQ_ALLOW_ALL_USERS",
        Platform.YUANBAO: "YUANBAO_ALLOW_ALL_USERS",
    }

    # Plugin platforms: check the registry for auth env var names
    if source.platform not in platform_env_map:
        try:
            from gateway.platform_registry import platform_registry
            entry = platform_registry.get(source.platform.value)
            if entry:
                if entry.allowed_users_env:
                    platform_env_map[source.platform] = entry.allowed_users_env
                if entry.allow_all_env:
                    platform_allow_all_map[source.platform] = entry.allow_all_env
        except Exception:
            pass

    # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
    platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
    if platform_allow_all_var and env_get(platform_allow_all_var, "").lower() in {"true", "1", "yes"}:
        return True

    # Adapter-verified role auth: the Discord adapter already confirmed the
    # user holds a role in DISCORD_ALLOWED_ROLES before dispatching the message.
    # Compare with ``is True`` so the real bool field authorizes while a
    # MagicMock source (test fixtures using ``object.__new__`` runners with
    # mock sources) does not auto-truthy through this gate (see pitfall #13).
    if allow_adapter_delegation and getattr(source, "role_authorized", False) is True:
        return True

    # Check pairing store. A pairing entry is a first-class authorization
    # grant, created only by a trusted operator approving a pairing code
    # (hermes gateway pairing approve / the authenticated dashboard) — an
    # inbound sender can never reach approve_code, so this is not an
    # attacker-controlled path. Honored as a UNION with the allowlist: a
    # paired user is authorized regardless of the allowlist, and when an
    # allowlist IS configured, operator approval also writes the user into
    # that allowlist (see PairingStore._approve_user), keeping a single
    # operator-visible source of truth. (#23778: the original bypass was the
    # inbound message/approval-button gate, not this grant; that gate is
    # fixed separately.)
    platform_name = source.platform.value if source.platform else ""
    if pairing_is_approved(platform_name, user_id):
        return True

    # Check platform-specific and global allowlists
    platform_allowlist = env_get(platform_env_map.get(source.platform, ""), "").strip()
    group_user_allowlist = ""
    group_chat_allowlist = ""
    if source.chat_type in {"group", "forum"}:
        group_user_allowlist = env_get(platform_group_user_env_map.get(source.platform, ""), "").strip()
        group_chat_allowlist = env_get(platform_group_chat_env_map.get(source.platform, ""), "").strip()
    global_allowlist = env_get("GATEWAY_ALLOWED_USERS", "").strip()

    if not platform_allowlist and not group_user_allowlist and not group_chat_allowlist and not global_allowlist:
        # No env allowlist configured. Adapters that own their own
        # config-driven access policy (dm_policy / group_policy /
        # allow_from / group_allow_from) gate access at intake, so for those
        # platforms we can honor the adapter's decision instead of the
        # env-only default-deny below -- but ONLY when that decision was an
        # actual allowlist restriction.
        #
        # The adapters default dm_policy / group_policy to "open", which
        # forwards EVERY sender. Reading "reached the gateway" as
        # authorization in that case would admit the whole external network
        # with no operator-configured allowlist -- the fail-open SECURITY.md
        # §2.6 forbids ("an allowlist is required for every enabled
        # network-exposed adapter ... code paths that fail open when no
        # allowlist is configured are code bugs"). "disabled" never
        # forwards, and "pairing" forwards unpaired DMs only so the gateway
        # can run its pairing handshake (the pairing-store check above
        # already denied this sender). So trust the adapter only when its
        # effective policy for THIS chat type is "allowlist"; for "open" /
        # "pairing" / anything else, fall through to default-deny, where
        # GATEWAY_ALLOW_ALL_USERS, the per-platform {PLATFORM}_ALLOW_ALL_USERS
        # flag (checked above), and the pairing flow remain the explicit
        # opt-ins to broader access. (#34515 follow-up: trusting "open" was a
        # fail-open.)
        if allow_adapter_delegation and adapter_enforces_own_access_policy(
            source.platform,
            source.profile,
        ):
            if source.chat_type in {"group", "forum", "channel"}:
                effective_policy = adapter_group_policy(
                    source.platform,
                    source.profile,
                )
                if adapter_group_has_sender_allowlist(
                    source.platform,
                    source.chat_id,
                    source.profile,
                ):
                    return True
            else:
                effective_policy = adapter_dm_policy(
                    source.platform,
                    source.profile,
                )
            if effective_policy == "allowlist":
                # Trust allowlist intake only when the live adapter still
                # allowlists this sender. Pairing revoke can clear
                # WHATSAPP_ALLOWED_USERS while a construction-time
                # ``_allow_from`` snapshot would otherwise keep authorizing
                # until restart; re-check when the adapter exposes a DM
                # allowlist helper. ``adapter_dm_is_allowed`` returns None when
                # it does not, which keeps the historical "reached the gateway
                # under allowlist policy" rubber-stamp for those adapters
                # (#34515).
                if source.chat_type not in {"group", "forum", "channel"}:
                    dm_allowed = adapter_dm_is_allowed(
                        source.platform,
                        source.profile,
                        user_id,
                    )
                    if dm_allowed is not None:
                        return bool(dm_allowed)
                return True
        # Some adapters (e.g. Telegram) gate access via config.extra.allow_from /
        # group_allow_from at intake but do not override enforces_own_access_policy.
        # Check their allowlist here so config.yaml-configured allow_from works
        # without requiring a separate {PLATFORM}_ALLOWED_USERS env var.
        adapter_allowed = adapter_allow_from(
            source.platform, source.profile, source.chat_type in {"group", "forum", "channel"}
        )
        if adapter_allowed and (user_id in adapter_allowed or "*" in adapter_allowed):
            return True
        # No allowlists configured -- check global allow-all flag
        return env_get("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}

    # Telegram can optionally authorize group traffic by chat ID.
    # Keep this separate from TELEGRAM_GROUP_ALLOWED_USERS, which gates
    # the sender user ID for group/forum messages.
    if group_chat_allowlist and source.chat_type in {"group", "forum"} and source.chat_id:
        allowed_group_ids = {
            chat_id.strip() for chat_id in group_chat_allowlist.split(",") if chat_id.strip()
        }
        if "*" in allowed_group_ids or source.chat_id in allowed_group_ids:
            return True

    # Backward-compat shim for #15027: prior to PR #17686,
    # TELEGRAM_GROUP_ALLOWED_USERS was (mis)used as a chat-ID allowlist.
    # Values starting with "-" are Telegram chat IDs, not user IDs, so if
    # users still have those in TELEGRAM_GROUP_ALLOWED_USERS we honor them
    # as chat IDs and warn once. The correct var is now
    # TELEGRAM_GROUP_ALLOWED_CHATS.
    if (
        source.platform == Platform.TELEGRAM
        and group_user_allowlist
        and source.chat_type in {"group", "forum"}
        and source.chat_id
    ):
        legacy_chat_ids = {
            v.strip()
            for v in group_user_allowlist.split(",")
            if v.strip().startswith("-")
        }
        if legacy_chat_ids:
            if on_legacy_group_users_warning is not None:
                on_legacy_group_users_warning(",".join(sorted(legacy_chat_ids)))
            if source.chat_id in legacy_chat_ids:
                return True

    # Check if user is in any allowlist. In group/forum chats,
    # TELEGRAM_GROUP_ALLOWED_USERS is the scoped allowlist and should not
    # imply DM access; TELEGRAM_ALLOWED_USERS remains the platform-wide
    # allowlist and still works everywhere for backward compatibility.
    allowed_ids = set()
    if platform_allowlist:
        allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
    if group_user_allowlist:
        allowed_ids.update(uid.strip() for uid in group_user_allowlist.split(",") if uid.strip())
    if global_allowlist:
        allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

    # Adapters that resolve username-shaped allowlist entries to numeric
    # IDs at connect time (Discord's ``_resolve_allowed_usernames``) keep
    # the authoritative resolved set in adapter memory and mirror it into
    # the process env. A per-turn .env hot-reload can restore the RAW
    # username strings from the .env file into the env, so from the second
    # agent turn onward ``platform_allowlist`` holds usernames while
    # ``source.user_id`` is numeric — the operator is admitted by the
    # adapter but dropped here as "Unauthorized user". Union in the
    # adapter's resolved IDs so runtime resolution survives env reloads.
    # This is a UNION of the resolution of entries already present in the
    # configured allowlist — never a widening: the empty-allowlist
    # fail-closed branch above has already returned, and adapters only
    # resolve entries the operator wrote. Guarded on ``platform_allowlist``
    # so group/global-only configurations never consult adapter memory,
    # and duck-typed + type-checked so a caller with no adapter, or a mock
    # one, cannot auto-truthy its way into an authorization.
    if platform_allowlist:
        resolved_ids = adapter_resolved_allowlist_user_ids(source.platform, source.profile)
        if isinstance(resolved_ids, (set, frozenset, list, tuple)):
            allowed_ids.update(
                str(entry).strip()
                for entry in resolved_ids
                if isinstance(entry, (str, int)) and str(entry).strip()
            )

    # "*" in any allowlist means allow everyone (consistent with
    # SIGNAL_GROUP_ALLOWED_USERS precedent)
    if "*" in allowed_ids:
        return True

    check_ids = {user_id}
    if "@" in user_id:
        check_ids.add(user_id.split("@")[0])

    # WhatsApp: resolve phone↔LID aliases from bridge session mapping files
    if source.platform == Platform.WHATSAPP:
        normalized_allowed_ids = set()
        for allowed_id in allowed_ids:
            normalized_allowed_ids.update(_expand_whatsapp_auth_aliases(allowed_id))
        if normalized_allowed_ids:
            allowed_ids = normalized_allowed_ids

        check_ids.update(_expand_whatsapp_auth_aliases(user_id))
        normalized_user_id = _normalize_whatsapp_identifier(user_id)
        if normalized_user_id:
            check_ids.add(normalized_user_id)

    # SimpleX: SIMPLEX_ALLOWED_USERS accepts either the numeric contactId
    # or the contact's display name. The adapter sets user_id=contactId for
    # stability across renames, but the SimpleX UI never surfaces the
    # numeric id — operators only see display names, so that's what they
    # naturally put in the env var. Match both so the allowlist works
    # regardless of which form was chosen.
    # Plugin platform: compare by value since Platform.SIMPLEX is not a
    # hardcoded enum member (it's a dynamic plugin platform).
    if (
        source.platform is not None
        and source.platform.value == "simplex"
        and source.user_name
    ):
        check_ids.add(source.user_name)

    # Buzz (Nostr-based): BUZZ_ALLOWED_USERS accepts npub or hex, but
    # inbound event pubkeys are always 64-char hex. Decode npub entries
    # to hex so an operator who listed only their npub authorizes the
    # same identity as the hex form (#78428). Hex entries pass through
    # unchanged, so existing hex-only allowlists keep working.
    if source.platform is not None and source.platform.value == "buzz":
        allowed_ids = _normalize_nostr_allow_entries(allowed_ids)
        if user_id.startswith("npub"):
            hex_user = _npub_to_hex(user_id)
            if hex_user:
                check_ids.add(hex_user)

    return bool(check_ids & allowed_ids)


# ---------------------------------------------------------------------------
# Nostr npub → hex normalization (Buzz and future Nostr-based platforms).
#
# ``BUZZ_ALLOWED_USERS`` accepts either a 64-char hex pubkey or an ``npub1…``
# bech32 string, but inbound event pubkeys are always hex.  Without decoding,
# the central allowlist comparison string-matches the raw npub against the
# hex pubkey and an operator who listed only their npub sees every message
# rejected ("Unauthorized user: <hex pubkey>", #78428).  Pure stdlib; mirrors
# the decoder in plugins/platforms/buzz/adapter.py.
# ---------------------------------------------------------------------------

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values):
    chk = 1
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str):
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data, frombits: int, tobits: int, pad: bool = True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def _npub_to_hex(npub: str) -> Optional[str]:
    """Decode an ``npub1…`` bech32 string to a 64-char hex pubkey, else None."""
    npub = npub.strip().lower()
    if not npub.startswith("npub1"):
        return None
    data_part = npub[len("npub1"):]
    try:
        data = [_BECH32_CHARSET.index(c) for c in data_part]
    except ValueError:
        return None
    if _bech32_polymod(_bech32_hrp_expand("npub") + data) != 1:
        return None
    decoded = _convertbits(data[:-6], 5, 8, pad=False)
    if decoded is None or len(decoded) != 32:
        return None
    return bytes(decoded).hex()


def _normalize_nostr_allow_entries(entries: set) -> set:
    """Expand npub entries in an allowlist set to their hex pubkey form.

    Hex entries pass through unchanged; each valid ``npub1…`` entry is decoded
    and its 64-char hex form added, so either form authorizes the same
    identity (#78428).  Invalid entries are kept as-is (they simply never
    match an inbound hex pubkey).
    """
    expanded = set(entries)
    for entry in entries:
        if entry.lower().startswith("npub1"):
            hex_key = _npub_to_hex(entry)
            if hex_key:
                expanded.add(hex_key)
    return expanded


class GatewayAuthorizationMixin:
    """User/chat authorization methods for ``GatewayRunner``."""

    def _authorization_adapter(
        self,
        platform: Optional[Platform],
        profile: Optional[str] = None,
    ):
        """Resolve the live adapter whose intake policy should gate authorization.

        In multiplex mode, secondary-profile adapters live in
        ``_profile_adapters[profile]`` while the default/active profile uses
        ``self.adapters``. ``SessionSource.profile`` selects which map to consult.
        When a stamped profile has its own adapter registry entry, the default
        profile's same-platform adapter must not be consulted as a fallback.

        Consult ``_profile_adapters`` *before* comparing against
        ``_active_profile_name()``. Multiplex turns wrap authz in
        ``_profile_runtime_scope``, which overrides ``HERMES_HOME`` so
        ``get_active_profile_name()`` returns the secondary profile for the
        duration of the turn. Treating that scoped name as "primary" would
        look up ``self.adapters`` (empty for secondary-only platforms like
        A2A) and default-deny an already-authenticated peer.
        """
        if not platform:
            return None
        profile_name = (profile or "").strip() or None
        if profile_name and profile_name != "default":
            profile_adapters = getattr(self, "_profile_adapters", None) or {}
            if profile_name in profile_adapters:
                return profile_adapters[profile_name].get(platform)
            # Adapter ownership is process-wide: only the profile the gateway
            # was LAUNCHED as owns ``self.adapters``. ``_active_profile_name()``
            # reads the per-turn HERMES_HOME override, so inside a secondary
            # profile's ``_profile_runtime_scope`` it reports that secondary
            # and would hand it the default bot. Compare against the identity
            # captured at construction instead.
            primary_profile = getattr(self, "_primary_profile_name", None)
            if not primary_profile:
                active_profile_fn = getattr(self, "_active_profile_name", None)
                if callable(active_profile_fn):
                    try:
                        primary_profile = active_profile_fn()
                    except Exception:
                        primary_profile = None
            if profile_name == primary_profile:
                adapters = getattr(self, "adapters", None) or {}
                return adapters.get(platform)
            # Fail closed: a stamped secondary profile with no registry entry
            # (e.g. its adapter failed to connect) must NOT fall back to the
            # default profile's adapter — that sends replies out the wrong bot.
            return None
        adapters = getattr(self, "adapters", None) or {}
        return adapters.get(platform)

    def _adapter_for_source(self, source: Optional[SessionSource]):
        """Resolve the live adapter for an inbound ``SessionSource``."""
        if source is None:
            return None
        transport_adapter = self._registered_transport_adapter(source)
        if transport_adapter is not None:
            return transport_adapter
        # Relay ingress deliberately keeps the underlying platform on the
        # source so session keys and display policy remain Slack/Discord/etc.
        # Delivery still has to use the one live RelayAdapter that owns the
        # authenticated connector socket. Looking up the underlying platform
        # here silently disables streaming, typing, and tool progress when a
        # managed gateway does not also run that platform's native adapter.
        if getattr(source, "delivered_via_upstream_relay", False) is True:
            # One process-level RelayAdapter owns the connector socket for all
            # multiplexed profiles. Secondary profiles intentionally do not
            # register their own relay adapters, so profile-aware lookup would
            # fail and suppress streamed delivery for those profiles.
            adapters = getattr(self, "adapters", None) or {}
            return adapters.get(Platform.RELAY)
        # ``getattr`` guards test fixtures that build a bare source via
        # SimpleNamespace and omit ``profile`` (see AGENTS.md pitfall #17).
        return self._authorization_adapter(
            getattr(source, "platform", None),
            getattr(source, "profile", None),
        )

    def _registered_transport_adapter(self, source: SessionSource):
        """Return the registered adapter that created *source*, if retained.

        ``source.profile`` is the runtime/session namespace. A chat-based
        profile route can therefore differ from the adapter profile when one
        shared credential serves several routed runtimes. ``build_source``
        keeps the receiving adapter as in-process provenance so replies and
        intake-policy checks stay on that transport without weakening the
        fail-closed fallback for restored or hand-built sources.
        """
        adapter_ref = getattr(source, "_transport_adapter_ref", None)
        adapter = adapter_ref() if callable(adapter_ref) else None
        platform = getattr(source, "platform", None)
        if adapter is None or platform is None:
            return None
        if adapter is (getattr(self, "adapters", None) or {}).get(platform):
            return adapter
        profile_maps = getattr(self, "_profile_adapters", None) or {}
        for profile_adapters in profile_maps.values():
            if adapter is profile_adapters.get(platform):
                return adapter
        return None

    def _adapter_profile_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the transport-owning profile for adapter policy lookups."""
        adapter = self._registered_transport_adapter(source)
        platform = getattr(source, "platform", None)
        if adapter is not None:
            if adapter is (getattr(self, "adapters", None) or {}).get(platform):
                return None
            for profile, profile_adapters in (
                getattr(self, "_profile_adapters", None) or {}
            ).items():
                if adapter is profile_adapters.get(platform):
                    return profile
        return getattr(source, "profile", None)

    def _adapter_authorization_is_upstream(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether the adapter for *platform* delegates authz to a trusted upstream.

        Mirrors ``BasePlatformAdapter.authorization_is_upstream``. The relay
        adapter sets this True: the Team Gateway connector authenticates the
        gateway's WS and resolves owner-only author bindings before delivering,
        so an inbound relay event is already authorized as this instance's bound
        user. Unlike ``_adapter_enforces_own_access_policy`` (a LOCAL config
        policy the gateway mirrors only when it's an allowlist), this is an
        UPSTREAM decision the gateway honors directly. Defaults to ``False`` when
        the adapter is unknown or doesn't expose the flag.
        """
        if not platform:
            return False
        adapter = self._authorization_adapter(platform, profile)
        if adapter is None:
            return False
        return bool(getattr(adapter, "authorization_is_upstream", False))

    def _adapter_enforces_own_access_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether the adapter for *platform* gates access at intake itself.

        Mirrors ``BasePlatformAdapter.enforces_own_access_policy``. Adapters
        such as WeCom, Weixin, Yuanbao, QQBot, and WhatsApp evaluate their
        documented ``dm_policy`` / ``group_policy`` / ``allow_from`` config before a
        message is dispatched to the gateway. The flag alone is NOT "already
        authorized": these adapters default to ``open``, which forwards every
        sender, so ``_is_user_authorized`` only trusts the adapter when its
        effective policy for the chat type is an actual ``allowlist`` restriction
        (see that method). Defaults to ``False`` when the adapter is unknown or
        doesn't expose the flag.
        """
        if not platform:
            return False
        # Some test helpers build a bare GatewayRunner via object.__new__ and
        # never set ``adapters``; treat a missing/empty map as "no adapter"
        # rather than raising (see pitfalls.md #17).
        adapter = self._authorization_adapter(platform, profile)
        if adapter is None:
            return False
        return bool(getattr(adapter, "enforces_own_access_policy", False))

    def _adapter_dm_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Best-effort read of an own-policy adapter's effective DM policy.

        Returns the lowercased ``dm_policy`` (``"open"`` / ``"allowlist"`` /
        ``"disabled"`` / ``"pairing"``) for *platform*, or ``""`` when unknown.
        Prefers the live adapter's resolved ``_dm_policy`` — which already folds
        in both ``config.extra`` and the ``<PLATFORM>_DM_POLICY`` env var (the
        env var is not always bridged back into ``config.extra``) — and falls
        back to ``config.extra`` for bare runners built without a live adapter.

        Used by ``_is_user_authorized`` to decide whether an own-policy adapter
        actually restricted DM senders to a configured allowlist (trustworthy)
        or merely forwarded everyone under ``dm_policy: open`` / for a pairing
        handshake (not authorization). "Reached the gateway" only carries an
        authorization signal in the ``allowlist`` case.
        """
        if not platform:
            return ""
        adapter = self._authorization_adapter(platform, profile)
        policy = getattr(adapter, "_dm_policy", None) if adapter is not None else None
        if policy is None:
            config = getattr(self, "config", None)
            platform_cfg = (
                config.platforms.get(platform)
                if config is not None and hasattr(config, "platforms")
                else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                policy = extra.get("dm_policy")
        return str(policy or "").strip().lower()

    def _adapter_group_policy(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Best-effort read of an own-policy adapter's effective group policy.

        Mirror of ``_adapter_dm_policy`` for group / forum / channel traffic:
        returns the lowercased ``group_policy`` (``"open"`` / ``"allowlist"`` /
        ``"disabled"``) for *platform*, or ``""`` when unknown. Prefers the live
        adapter's resolved ``_group_policy`` and falls back to ``config.extra``
        for bare runners built without a live adapter.

        Used by ``_is_user_authorized`` to decide whether an own-policy adapter
        restricted group senders to a configured allowlist (trustworthy) or
        forwarded the whole channel under ``group_policy: open`` (not
        authorization).
        """
        if not platform:
            return ""
        adapter = self._authorization_adapter(platform, profile)
        policy = getattr(adapter, "_group_policy", None) if adapter is not None else None
        if policy is None:
            config = getattr(self, "config", None)
            platform_cfg = (
                config.platforms.get(platform)
                if config is not None and hasattr(config, "platforms")
                else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                policy = extra.get("group_policy")
        return str(policy or "").strip().lower()

    def _adapter_group_has_sender_allowlist(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        *,
        profile: Optional[str] = None,
    ) -> bool:
        """Whether a per-group sender allowlist gated this group message.

        WeCom supports ``groups.<group_id>.allow_from`` on top of the top-level
        ``group_policy``. A group may be open at the chat level while still
        restricting which senders inside that group can invoke Hermes. If such a
        message reached the gateway, the adapter already checked that sender
        allowlist, so it is a trustworthy intake decision rather than the
        fail-open ``group_policy: open`` case.
        """
        if not platform or not chat_id:
            return False
        adapter = self._authorization_adapter(platform, profile)
        groups = getattr(adapter, "_groups", None) if adapter is not None else None
        if groups is None:
            config = getattr(self, "config", None)
            platform_cfg = (
                config.platforms.get(platform)
                if config is not None and hasattr(config, "platforms")
                else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                groups = extra.get("groups")
        if not isinstance(groups, dict):
            return False

        chat_id_str = str(chat_id)
        group_cfg = groups.get(chat_id_str)
        if not isinstance(group_cfg, dict):
            lowered = chat_id_str.lower()
            for key, value in groups.items():
                if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
                    group_cfg = value
                    break
        if not isinstance(group_cfg, dict):
            group_cfg = groups.get("*")
        if not isinstance(group_cfg, dict):
            return False

        sender_allow = group_cfg.get("allow_from") or group_cfg.get("allowFrom")
        if isinstance(sender_allow, str):
            return bool(sender_allow.strip())
        if isinstance(sender_allow, (list, tuple, set)):
            return any(str(item).strip() for item in sender_allow)
        return False

    def _pairing_store_for(self, source: "SessionSource"):
        """Pick the per-profile PairingStore for a source, falling back to global.

        In a multiplexing gateway, each profile owns its own pairing whitelist
        so isolation is preserved. When the source has no profile (single-
        profile gateway, or a path that hasn't stamped profile yet) or the
        profile isn't registered, fall back to ``self.pairing_store`` (the
        global default) so existing behavior is preserved.
        """
        per_profile = getattr(self, "pairing_stores", None) or {}
        profile = getattr(source, "profile", None)
        if profile and profile in per_profile:
            return per_profile[profile]
        return getattr(self, "pairing_store", None)

    def _is_user_authorized(
        self,
        source: SessionSource,
        *,
        allow_adapter_delegation: bool = True,
    ) -> bool:
        """
        Check if a user is authorized to use the bot.

        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny

        Thin wrapper around the module-level pure function :func:`is_authorized`
        — this method's only job is to bind that function's injected
        dependencies (the live adapter-policy lookups, this runner's
        per-profile pairing store via :meth:`_pairing_store_for`, and the
        one-time legacy-warning log) to ``self``. The actual decision logic
        lives in :func:`is_authorized` so it can be called without a
        ``GatewayRunner`` instance (e.g. by the Telegram Mini App dashboard
        auth tier check).
        """
        from gateway.run import logger

        # A routed/shared-adapter source's *transport* profile can differ from
        # source.profile (one shared credential serving several routed
        # runtimes -- see _adapter_profile_for_source's docstring). The
        # pre-extraction inline body always resolved this once up front and
        # used it for every adapter-policy lookup below; is_authorized() only
        # receives source.profile from its caller, so this wrapper must do
        # the same resolution and pass adapter_profile through explicitly
        # instead of letting the lambdas use whatever profile is_authorized()
        # happens to pass them (raw source.profile, which fails closed for a
        # routed adapter with no same-named entry in _profile_adapters).
        adapter_profile = self._adapter_profile_for_source(source)

        def _warn_legacy_group_users(chat_ids: str) -> None:
            if not getattr(self, "_warned_telegram_group_users_legacy", False):
                logger.warning(
                    "TELEGRAM_GROUP_ALLOWED_USERS contains chat-ID-shaped values "
                    "(%s). Treating them as chat IDs for backward compatibility. "
                    "Move chat IDs to TELEGRAM_GROUP_ALLOWED_CHATS — the _USERS var "
                    "is now for sender user IDs.",
                    chat_ids,
                )
                self._warned_telegram_group_users_legacy = True

        def _adapter_dm_is_allowed(
            platform: Optional[Platform], profile: Optional[str], uid: str
        ) -> Optional[bool]:
            """Re-check a DM sender against the live adapter's allowlist.

            Returns None when there is no live adapter or it exposes no
            ``_is_dm_allowed`` helper, so is_authorized keeps the historical
            allowlist-intake rubber-stamp for those adapters (#34515).
            """
            adapter = self._authorization_adapter(platform, profile=adapter_profile)
            dm_check = (
                getattr(adapter, "_is_dm_allowed", None) if adapter is not None else None
            )
            if not callable(dm_check):
                return None
            return bool(dm_check(uid))

        def _adapter_resolved_allowlist_user_ids(
            platform: Optional[Platform], profile: Optional[str]
        ):
            """Live adapter's resolved numeric allowlist, or None.

            Best-effort: an adapter mid-reconnect (or any resolver error)
            must not break authorization for senders the env allowlist
            already covers, so a raise here is swallowed rather than
            propagated (mirrors the pre-extraction inline body's own
            try/except around both the adapter lookup and the resolver call).
            """
            try:
                adapter = self._adapter_for_source(source)
            except Exception:
                return None
            resolver = getattr(adapter, "resolved_allowlist_user_ids", None)
            if not callable(resolver):
                return None
            try:
                return resolver()
            except Exception:
                return None

        return is_authorized(
            source,
            # Route through the per-profile PairingStore lookup (multiplex
            # gateways isolate each profile's whitelist) rather than the flat
            # ``self.pairing_store`` this wrapper used before -- upstream
            # added ``_pairing_store_for`` for that after this extraction was
            # originally written; binding it here keeps this wrapper current
            # with that behavior instead of silently reverting to the single
            # global store.
            pairing_is_approved=lambda platform_name, uid: (
                lambda store: store is not None and store.is_approved(platform_name, uid)
            )(self._pairing_store_for(source)),
            allow_adapter_delegation=allow_adapter_delegation,
            adapter_authorization_is_upstream=lambda platform, profile: (
                self._adapter_authorization_is_upstream(platform, profile=adapter_profile)
            ),
            adapter_enforces_own_access_policy=lambda platform, profile: (
                self._adapter_enforces_own_access_policy(platform, profile=adapter_profile)
            ),
            adapter_dm_policy=lambda platform, profile: (
                self._adapter_dm_policy(platform, profile=adapter_profile)
            ),
            adapter_group_policy=lambda platform, profile: (
                self._adapter_group_policy(platform, profile=adapter_profile)
            ),
            adapter_group_has_sender_allowlist=lambda platform, chat_id, profile: (
                self._adapter_group_has_sender_allowlist(platform, chat_id, profile=adapter_profile)
            ),
            # config.yaml-only allowlist fallbacks (platforms.<platform>.extra.
            # group_allowed_chats / allow_from / group_allow_from) that upstream
            # grew on the inline method after this extraction was originally
            # written -- ported here rather than left behind so the pure
            # function stays behavior-identical to the live gateway's checks.
            adapter_group_allowed_chats=lambda platform, profile: (
                lambda adapter: (
                    _coerce_allow_set(
                        (getattr(getattr(adapter, "config", None), "extra", None) or {}).get(
                            "group_allowed_chats"
                        )
                    )
                    if adapter is not None else set()
                )
            )(self._adapter_for_source(source)),
            adapter_allow_from=lambda platform, profile, is_group: (
                lambda adapter: (
                    (lambda allowed: (
                        {(adapter.normalize_user_id(entry) or entry) for entry in allowed}
                        if callable(getattr(adapter, "normalize_user_id", None))
                        else allowed
                    ))(
                        _coerce_allow_set(
                            (getattr(getattr(adapter, "config", None), "extra", None) or {}).get(
                                "group_allow_from" if is_group else "allow_from"
                            )
                            # Plugin platforms whose registry entry declares
                            # ``allowed_users_env`` (e.g. Buzz) carry their
                            # operator-configured allowlist in
                            # ``PlatformConfig.extra.allowed_users`` instead.
                            # Under multiplex the YAML→env bridge is
                            # first-writer-wins, so only the default profile's
                            # list ever reaches the env var read elsewhere in
                            # this function; fall back to the live
                            # (profile-routed) adapter's own config so a
                            # secondary profile's allowlist authorizes its
                            # users (#98738 / #82871). An absent/empty entry
                            # changes nothing here -- the default-deny path
                            # still applies. Upstream added both this fallback
                            # and the normalize_user_id step above to the
                            # inline method after this extraction was
                            # originally written.
                            or (
                                (getattr(getattr(adapter, "config", None), "extra", None) or {}).get(
                                    "allowed_users"
                                )
                                if _platform_declares_allowed_users_env(platform)
                                else None
                            )
                        )
                    )
                )
                if adapter is not None else set()
            )(self._adapter_for_source(source)),
            # Union the adapter's resolved numeric allowlist IDs (e.g.
            # Discord's ``resolved_allowlist_user_ids()``) into the
            # env-derived allowlist -- upstream added this to the inline
            # method after this extraction was originally written. Adapters
            # that resolve username-shaped allowlist entries to numeric IDs
            # at connect time keep the authoritative resolved set in adapter
            # memory and mirror it into the process env, but the gateway's
            # per-turn .env hot-reload restores the RAW username strings from
            # the .env file -- so from the second turn onward the env-derived
            # allowlist holds usernames while source.user_id is numeric and
            # the operator is wrongly dropped as "Unauthorized user". Duck-
            # typed + isinstance-guarded against mock adapters in the pure
            # function itself.
            adapter_resolved_allowlist_user_ids=_adapter_resolved_allowlist_user_ids,
            # Live-adapter DM allowlist re-check upstream added to the inline
            # method after this extraction was originally written (#34515):
            # without it, revoking the sole allowlist entry keeps authorizing
            # that sender until the gateway restarts. Routed through
            # ``_authorization_adapter`` (not ``_adapter_for_source``) to match
            # the upstream inline body exactly.
            adapter_dm_is_allowed=_adapter_dm_is_allowed,
            on_legacy_group_users_warning=_warn_legacy_group_users,
            # Preserve the profile-scoped secret_scope lookup upstream added to
            # this env-var read after this extraction was originally written
            # (_auth_env prefers the multiplex profile's secret_scope value over
            # a bare os.getenv) -- passing it here keeps live-gateway behavior
            # identical to the pre-extraction inline body instead of silently
            # falling back to is_authorized's plain-os.getenv default.
            env_get=_auth_env,
            # Upstream hardened the chat-allowlist and {PLATFORM}_ALLOW_BOTS
            # reads specifically (not every env_get read) to the stricter,
            # multiplex-authoritative helper after this extraction was
            # originally written -- passing it here keeps those two checks
            # identical to the pre-extraction inline body instead of falling
            # back to is_authorized's plain env_get default for them.
            platform_gate_env=_platform_gate_env,
        )

    def _get_unauthorized_dm_behavior(
        self,
        platform: Optional[Platform],
        *,
        profile: Optional[str] = None,
    ) -> str:
        """Return how unauthorized DMs should be handled for a platform.

        Resolution order:
        1. Explicit per-platform ``unauthorized_dm_behavior`` in config — always wins.
        2. Email defaults to ``"ignore"`` unless explicitly opted into
           pairing. Inboxes may contain arbitrary unread human messages, so
           replying with pairing codes is not a safe platform default.
        3. Explicit global ``unauthorized_dm_behavior`` in config — wins for
           chat-shaped platforms when no per-platform override is set.
        4. When an adapter-level DM policy opts into pairing or silent drop, honor it.
        5. When an allowlist (``PLATFORM_ALLOWED_USERS``,
           ``PLATFORM_GROUP_ALLOWED_USERS`` / ``PLATFORM_GROUP_ALLOWED_CHATS``,
           or ``GATEWAY_ALLOWED_USERS``) is configured, default to ``"ignore"`` —
           the allowlist signals that the owner has deliberately restricted
           access; spamming unknown contacts with pairing codes is both noisy
           and a potential info-leak. (#9337)
        6. No allowlist and no explicit config → ``"pair"`` (open-gateway default).
        """
        config = getattr(self, "config", None)

        # Check for an explicit per-platform override first.
        if config and hasattr(config, "get_unauthorized_dm_behavior") and platform:
            platform_cfg = config.platforms.get(platform) if hasattr(config, "platforms") else None
            if platform_cfg and "unauthorized_dm_behavior" in getattr(platform_cfg, "extra", {}):
                # Operator explicitly configured behavior for this platform — respect it.
                return config.get_unauthorized_dm_behavior(platform)

        # Email is inbox-shaped, not chat-shaped: an agent mailbox may contain
        # unrelated unread human email. Require an explicit per-platform
        # ``unauthorized_dm_behavior: pair`` opt-in before replying to unknown
        # senders with pairing codes. Keep this before the global fallback to
        # match GatewayConfig.get_unauthorized_dm_behavior().
        if platform == Platform.EMAIL:
            return "ignore"

        # Check for an explicit global config override.
        if config and hasattr(config, "unauthorized_dm_behavior"):
            if config.unauthorized_dm_behavior != "pair":  # non-default → explicit override
                return config.unauthorized_dm_behavior

        # Config-driven dm_policy (WeCom / Weixin / Yuanbao / QQBot). An
        # allowlist or disabled DM policy means the operator restricted access,
        # so unauthorized DMs should be dropped silently rather than answered
        # with a pairing code. An explicit pairing policy opts back into codes.
        # Prefer the profile-scoped live adapter's resolved policy in multiplex
        # mode; fall back to the default profile's config.extra.
        if platform:
            dm_policy = self._adapter_dm_policy(platform, profile=profile)
            if not dm_policy and config and hasattr(config, "platforms"):
                platform_cfg = config.platforms.get(platform)
                extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
                if isinstance(extra, dict):
                    dm_policy = str(extra.get("dm_policy") or "").strip().lower()
            if dm_policy == "pairing":
                return "pair"
            if dm_policy in {"allowlist", "disabled"}:
                return "ignore"

        # No explicit override.  Fall back to allowlist-aware default:
        # if any allowlist is configured for this platform, silently drop
        # unauthorized messages instead of sending pairing codes.
        if platform:
            platform_env_map = {
                Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
                Platform.DISCORD:  "DISCORD_ALLOWED_USERS",
                Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
                Platform.WHATSAPP_CLOUD: "WHATSAPP_CLOUD_ALLOWED_USERS",
                Platform.SLACK:    "SLACK_ALLOWED_USERS",
                Platform.SIGNAL:   "SIGNAL_ALLOWED_USERS",
                Platform.EMAIL:    "EMAIL_ALLOWED_USERS",
                Platform.SMS:      "SMS_ALLOWED_USERS",
                Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
                Platform.MATRIX:   "MATRIX_ALLOWED_USERS",
                Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
                Platform.FEISHU:   "FEISHU_ALLOWED_USERS",
                Platform.WECOM:    "WECOM_ALLOWED_USERS",
                Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOWED_USERS",
                Platform.WEIXIN:   "WEIXIN_ALLOWED_USERS",
                Platform.BLUEBUBBLES: "BLUEBUBBLES_ALLOWED_USERS",
                Platform.QQBOT:    "QQ_ALLOWED_USERS",
            }
            platform_group_env_map = {
                Platform.TELEGRAM: (
                    "TELEGRAM_GROUP_ALLOWED_USERS",
                    "TELEGRAM_GROUP_ALLOWED_CHATS",
                ),
                Platform.QQBOT: ("QQ_GROUP_ALLOWED_USERS",),
            }
            if _platform_gate_env(platform_env_map.get(platform, "")).strip():
                return "ignore"
            for env_key in platform_group_env_map.get(platform, ()):
                if _platform_gate_env(env_key).strip():
                    return "ignore"

        if _platform_gate_env("GATEWAY_ALLOWED_USERS").strip():
            return "ignore"

        return "pair"

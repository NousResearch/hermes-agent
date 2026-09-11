"""User-authorization mixin for ``GatewayRunner``: may this user/chat talk to the agent,
the per-adapter DM policy, the unauthorized-DM behavior, and the bot loop guard.

``gateway.run`` is never imported at module import time (cycle). The unauthorized-DM method still
logs through ``gateway.run``'s logger so its records keep that name.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Callable, Optional

from gateway.bot_loop_guard import BotLoopGuard
from gateway.config import Platform
from gateway.pairing import _PLATFORM_ALLOWLIST_ENV
from gateway.session import SessionSource
from gateway.whatsapp_identity import (
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)

_GROUP_CHAT_TYPES = frozenset({"group", "forum", "channel"})
_GROUP_FORUM_TYPES = frozenset({"group", "forum"})
_TRUTHY = frozenset({"true", "1", "yes"})
_BOT_LOOP_GUARD_INIT_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

# Platform -> ``<PLATFORM>_ALLOWED_USERS`` / ``<PLATFORM>_ALLOW_ALL_USERS``. Shared with the pairing
# store's allowlist mirror (single source of truth); plugin platforms are added per-call from the registry.
_ALLOWED_USERS_ENV = {Platform(k): v for k, v in _PLATFORM_ALLOWLIST_ENV.items()}
_ALLOW_ALL_ENV = {p: v.replace("_ALLOWED_USERS", "_ALLOW_ALL_USERS") for p, v in _ALLOWED_USERS_ENV.items()}
_GROUP_USER_ENV = {Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_USERS"}
_GROUP_CHAT_ENV = {Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS", Platform.QQBOT: "QQ_GROUP_ALLOWED_USERS"}
_ALLOW_BOTS_ENV = {
    # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist (#4466). Checked before the
    # no-user-id guard below: some platforms deliver bot/automation traffic with no user_id at all -- e.g.
    # Slack Workflow Builder posts arrive as subtype=bot_message with user=None -- so deferring past the
    # guard would reject them outright (the same reason the chat-scoped allowlist above runs early).
    Platform.DISCORD: "DISCORD_ALLOW_BOTS",
    Platform.FEISHU: "FEISHU_ALLOW_BOTS",
    Platform.TELEGRAM: "TELEGRAM_ALLOW_BOTS",
    Platform.SLACK: "SLACK_ALLOW_BOTS",
}


def _platform_gate_env(name: str, default: str = "") -> str:
    """Read an allow/deny gate env var with per-profile isolation.

    With a profile secret scope installed AND multiplexing active, a scoped miss returns ``default``
    instead of falling through to ``os.environ``, which may hold ANOTHER profile's first-writer
    bridged value (allowlist leak). Single-profile deployments behave exactly like ``os.getenv``.

    Under multiplex the process env may hold ANOTHER profile's first-writer-bridged value (the YAML→env
    bridges in the Discord/Telegram adapters' ``_apply_yaml_config`` are first-writer-wins), so falling
    through would leak profile A's allowlist into profile B (issue #72348).
    """
    if not name:
        return default
    with contextlib.suppress(Exception):
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        if scope is not None and is_multiplex_active():
            val = scope.get(name)
            return default if val is None else str(val).strip()
    return (os.getenv(name) or default).strip()


_auth_env = _platform_gate_env


def _env_truthy(name: str) -> bool:
    return _auth_env(name).lower() in _TRUTHY


def _registry_entry(platform):
    """Platform-registry entry for a (plugin) platform, or None."""
    if platform is None:
        return None
    with contextlib.suppress(Exception):
        from gateway.platform_registry import platform_registry

        return platform_registry.get(platform.value)
    return None


def _coerce_allow_set(raw) -> set[str]:
    """Parse an allowlist (YAML list or comma-separated scalar) into a set of strings."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _allows(allowed: set[str], candidate: Optional[str]) -> bool:
    return "*" in allowed or candidate in allowed


def _adapter_config_extra(adapter) -> dict:
    return getattr(getattr(adapter, "config", None), "extra", None) or {}


# Nostr npub -> hex (Buzz): ``BUZZ_ALLOWED_USERS`` accepts hex or ``npub1…`` but inbound pubkeys
# are always hex. Pure stdlib; mirrors plugins/platforms/buzz/adapter.py.
# Without decoding, the central allowlist comparison string-matches the raw npub against the hex pubkey and
# an operator who listed only their npub sees every message rejected ("Unauthorized user: <hex pubkey>",
# #78428).
_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_BECH32_GENERATOR = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)


def _bech32_polymod(values):
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i, gen in enumerate(_BECH32_GENERATOR):
            chk ^= gen if ((top >> i) & 1) else 0
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
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif not pad and (bits >= frombits or ((acc << (tobits - bits)) & maxv)):
        return None
    return ret


def _npub_to_hex(npub: str) -> Optional[str]:
    """Decode an ``npub1…`` bech32 string to a 64-char hex pubkey, else None."""
    npub = npub.strip().lower()
    if not npub.startswith("npub1"):
        return None
    try:
        data = [_BECH32_CHARSET.index(c) for c in npub[len("npub1"):]]
    except ValueError:
        return None
    if _bech32_polymod(_bech32_hrp_expand("npub") + data) != 1:
        return None
    decoded = _convertbits(data[:-6], 5, 8, pad=False)
    if decoded is None or len(decoded) != 32:
        return None
    return bytes(decoded).hex()


def _normalize_nostr_allow_entries(entries: set) -> set:
    """Add the hex form of every valid ``npub1…`` entry; invalid entries are kept as-is.

    Hex entries pass through unchanged; each valid ``npub1…`` entry is decoded and its 64-char hex form
    added, so either form authorizes the same identity (#78428). Invalid entries are kept as-is (they simply
    never match an inbound hex pubkey).
    """
    return set(entries) | {h for e in entries if e.lower().startswith("npub1") and (h := _npub_to_hex(e))}


def _principal_matches_allowlist(source, user_id: str, allowed_ids: set) -> bool:
    """Whether *user_id* (under any platform-specific alias) is in *allowed_ids*."""
    check_ids = {user_id}
    if "@" in user_id:
        check_ids.add(user_id.split("@")[0])

    # WhatsApp (Baileys + Cloud): phone<->LID / JID aliases match the same principal.
    if source.platform in {Platform.WHATSAPP, Platform.WHATSAPP_CLOUD}:
        allowed_ids = set().union(*(_expand_whatsapp_auth_aliases(a) for a in allowed_ids)) or allowed_ids
        check_ids.update(_expand_whatsapp_auth_aliases(user_id))
        normalized_user_id = _normalize_whatsapp_identifier(user_id)
        if normalized_user_id:
            check_ids.add(normalized_user_id)

    platform_value = source.platform.value if source.platform is not None else None
    # SimpleX: user_id is the numeric contactId but the UI only shows display names.
    if platform_value == "simplex" and source.user_name:
        check_ids.add(source.user_name)
    # Buzz: allowlist may hold npub or hex; inbound pubkeys are hex.
    if platform_value == "buzz":
        # Buzz (Nostr-based): BUZZ_ALLOWED_USERS accepts npub or hex, but inbound event pubkeys are always
        # 64-char hex. Decode npub entries to hex so an operator who listed only their npub authorizes the
        # same identity as the hex form (#78428). Hex entries pass through unchanged, so existing hex-only
        # allowlists keep working.
        allowed_ids = _normalize_nostr_allow_entries(allowed_ids)
        hex_user = _npub_to_hex(user_id) if user_id.startswith("npub") else None
        if hex_user:
            check_ids.add(hex_user)
    return bool(check_ids & allowed_ids)


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
        or adapter_authorization_is_upstream(source.platform, source.profile)
    ):
        return True

    user_id = source.user_id
    is_group = source.chat_type in _GROUP_CHAT_TYPES
    is_group_or_forum = source.chat_type in _GROUP_FORUM_TYPES

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
    if is_group and source.chat_id:
        chat_allowlist_env = _GROUP_CHAT_ENV.get(source.platform, "")
        if chat_allowlist_env:
            raw_chat_allowlist = platform_gate_env(chat_allowlist_env)
            if raw_chat_allowlist and _allows(_coerce_allow_set(raw_chat_allowlist), source.chat_id):
                return True

        # Fallback: also check adapter-level config (config.yaml) for
        # platforms.<platform>.extra.group_allowed_chats. The Telegram
        # observe-unmentioned mode strips user_id from triggered group
        # messages (_apply_telegram_group_observe_attribution), so the
        # env-var-only check above misses config.yaml-configured allowlists.
        adapter_group_chats = adapter_group_allowed_chats(source.platform, source.profile)
        if adapter_group_chats and _allows(adapter_group_chats, source.chat_id):
            return True

    # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist (#4466).
    # Checked before the no-user-id guard below: some platforms deliver
    # bot/automation traffic with no user_id at all -- e.g. Slack Workflow
    # Builder posts arrive as subtype=bot_message with user=None -- so
    # deferring past the guard would reject them outright (the same reason
    # the chat-scoped allowlist above runs early).
    if getattr(source, "is_bot", False):
        allow_bots_var = _ALLOW_BOTS_ENV.get(source.platform)
        if allow_bots_var and platform_gate_env(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
            return True

    if not user_id:
        return False

    platform_allow_env = _ALLOWED_USERS_ENV.get(source.platform, "")
    platform_allow_all_var = _ALLOW_ALL_ENV.get(source.platform, "")
    # Plugin platforms: check the registry for auth env var names.
    if source.platform not in _ALLOWED_USERS_ENV:
        entry = _registry_entry(source.platform)
        if entry:
            platform_allow_env = getattr(entry, "allowed_users_env", "") or platform_allow_env
            platform_allow_all_var = getattr(entry, "allow_all_env", "") or platform_allow_all_var

    # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
    if platform_allow_all_var and env_get(platform_allow_all_var, "").lower() in _TRUTHY:
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
    platform_allowlist = env_get(platform_allow_env, "").strip()
    group_user_allowlist = ""
    group_chat_allowlist = ""
    if is_group_or_forum:
        group_user_allowlist = env_get(_GROUP_USER_ENV.get(source.platform, ""), "").strip()
        group_chat_allowlist = env_get(_GROUP_CHAT_ENV.get(source.platform, ""), "").strip()
    global_allowlist = env_get("GATEWAY_ALLOWED_USERS", "").strip()

    if not (platform_allowlist or group_user_allowlist or group_chat_allowlist or global_allowlist):
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
        if allow_adapter_delegation and adapter_enforces_own_access_policy(source.platform, source.profile):
            if is_group:
                effective_policy = adapter_group_policy(source.platform, source.profile)
                if adapter_group_has_sender_allowlist(source.platform, source.chat_id, source.profile):
                    return True
            else:
                effective_policy = adapter_dm_policy(source.platform, source.profile)
            if effective_policy == "allowlist":
                # Trust allowlist intake only when the live adapter still
                # allowlists this sender. Pairing revoke can clear the env
                # allowlist while a construction-time snapshot on the adapter
                # would otherwise keep authorizing until restart; re-check
                # when the adapter exposes a DM allowlist helper.
                # ``adapter_dm_is_allowed`` returns None when it does not,
                # which keeps the historical "reached the gateway under
                # allowlist policy" rubber-stamp for those adapters (#34515).
                if not is_group:
                    dm_allowed = adapter_dm_is_allowed(source.platform, source.profile, user_id)
                    if dm_allowed is not None:
                        return bool(dm_allowed)
                return True
        # Some adapters (e.g. Telegram) gate access via config.extra.allow_from /
        # group_allow_from at intake but do not override enforces_own_access_policy.
        # Check their allowlist here so config.yaml-configured allow_from works
        # without requiring a separate {PLATFORM}_ALLOWED_USERS env var.
        adapter_allowed = adapter_allow_from(source.platform, source.profile, is_group)
        if adapter_allowed and _allows(adapter_allowed, user_id):
            return True
        # No allowlists configured -- check global allow-all flag
        return env_get("GATEWAY_ALLOW_ALL_USERS", "").lower() in _TRUTHY

    # Telegram can optionally authorize group traffic by chat ID.
    # Keep this separate from TELEGRAM_GROUP_ALLOWED_USERS, which gates
    # the sender user ID for group/forum messages.
    if is_group_or_forum and source.chat_id:
        if group_chat_allowlist and _allows(_coerce_allow_set(group_chat_allowlist), source.chat_id):
            return True

        # Backward-compat shim for #15027: prior to PR #17686,
        # TELEGRAM_GROUP_ALLOWED_USERS was (mis)used as a chat-ID allowlist.
        # Values starting with "-" are Telegram chat IDs, not user IDs, so if
        # users still have those in TELEGRAM_GROUP_ALLOWED_USERS we honor them
        # as chat IDs and warn once. The correct var is now
        # TELEGRAM_GROUP_ALLOWED_CHATS.
        if source.platform == Platform.TELEGRAM and group_user_allowlist:
            legacy_chat_ids = {
                v.strip() for v in group_user_allowlist.split(",") if v.strip().startswith("-")
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
    allowed_ids = (
        _coerce_allow_set(platform_allowlist)
        | _coerce_allow_set(group_user_allowlist)
        | _coerce_allow_set(global_allowlist)
    )

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
    # and duck-typed + isinstance-guarded so a caller with no adapter, or a
    # mock one, cannot auto-truthy its way into an authorization.
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
    return "*" in allowed_ids or _principal_matches_allowlist(source, user_id, allowed_ids)


class GatewayAuthorizationMixin:
    """User/chat authorization methods for ``GatewayRunner``."""

    # ``getattr(self, ...)`` throughout: test helpers build bare runners via ``object.__new__``
    # without ``adapters`` / ``config``.

    def _primary_adapters(self) -> dict:
        return getattr(self, "adapters", None) or {}

    def _profile_adapters_map(self) -> dict:
        return getattr(self, "_profile_adapters", None) or {}

    def _authorization_adapter(self, platform: Optional[Platform], profile: Optional[str] = None):
        """Live adapter whose intake policy gates authorization (``_adapters_for_profile`` for the
        profile rule). ``None`` when the profile has no adapter for *platform*.
        """
        if not platform:
            return None
        return self._adapters_for_profile(profile).get(platform)

    def _adapters_for_profile(self, profile: Optional[str]) -> dict:
        """The live adapter map *profile* may deliver through: ``_profile_adapters[p]`` for a
        secondary, ``self.adapters`` only for the primary/default. ``_profile_adapters`` is consulted
        BEFORE the active profile name: multiplex turns override ``HERMES_HOME`` so
        ``_active_profile_name()`` reports the secondary profile mid-turn, and treating it as primary
        would hand it the default bot. A named profile with no map gets ``{}`` — fail closed: a
        secondary whose adapter failed to connect must NOT fall back to the default profile's adapter
        (replies, tool sends, marker-file notices out the wrong bot)."""
        profile_name = (profile or "").strip() or None
        if not profile_name or profile_name == "default":
            return self._primary_adapters()
        profile_adapters = self._profile_adapters_map()
        if profile_name in profile_adapters:
            return profile_adapters[profile_name]
        # Identity captured at construction, not the per-turn HERMES_HOME-derived name.
        primary_profile = getattr(self, "_primary_profile_name", None)
        if not primary_profile:
            with contextlib.suppress(Exception):
                primary_profile = self._active_profile_name()
        return self._primary_adapters() if profile_name == primary_profile else {}

    def _adapter_for_source(self, source: Optional[SessionSource]):
        """Resolve the live adapter for an inbound ``SessionSource``."""
        if source is None:
            return None
        owner = self._transport_owner(source)
        if owner is not None:
            return owner[0]
        # Relay ingress keeps the underlying platform on the source, but delivery must use the one
        # process-level RelayAdapter owning the connector socket; a profile-aware lookup would
        # silently disable streaming/typing/tool progress.
        if getattr(source, "delivered_via_upstream_relay", False) is True:
            return self._primary_adapters().get(Platform.RELAY)
        # ``getattr``: test fixtures build bare SimpleNamespace sources without ``profile``.
        return self._authorization_adapter(getattr(source, "platform", None), getattr(source, "profile", None))

    def _owning_profile(self, adapter, platform):
        """Return (registered, profile) for a live adapter: profile is None for primary."""
        if adapter is self._primary_adapters().get(platform):
            return True, None
        for profile, profile_adapters in self._profile_adapters_map().items():
            if adapter is profile_adapters.get(platform):
                return True, profile
        return False, None

    def _transport_owner(self, source: SessionSource):
        """``(adapter, profile)`` of the registered adapter that created *source*, if retained; else None.

        ``source.profile`` may differ from the adapter profile when one shared credential serves
        several routed runtimes; ``build_source`` keeps the receiving adapter as provenance so replies
        stay on that transport. Restored/hand-built sources fall back (fail-closed) to profile lookup.
        """
        adapter_ref = getattr(source, "_transport_adapter_ref", None)
        adapter = adapter_ref() if callable(adapter_ref) else None
        platform = getattr(source, "platform", None)
        if adapter is None or platform is None:
            return None
        registered, profile = self._owning_profile(adapter, platform)
        return (adapter, profile) if registered else None

    def _adapter_profile_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the transport-owning profile for adapter policy lookups."""
        owner = self._transport_owner(source)
        return owner[1] if owner is not None else getattr(source, "profile", None)

    def _adapter_flag(self, platform, name: str, profile) -> bool:
        """Adapter-declared boolean, False when unknown. ``authorization_is_upstream`` (relay: a trusted
        authenticated upstream decides) is honored directly; ``enforces_own_access_policy`` (WeCom, Weixin,
        Yuanbao, QQBot, WhatsApp gate at intake) is NOT "already authorized" — those adapters default to
        ``open``, so ``_is_user_authorized`` only trusts them under an actual ``allowlist`` policy."""
        if not platform:
            return False
        adapter = self._authorization_adapter(platform, profile)
        return adapter is not None and bool(getattr(adapter, name, False))

    def _config_extra(self, platform) -> dict:
        """``config.platforms[platform].extra`` as a dict ({} when absent)."""
        platforms = getattr(getattr(self, "config", None), "platforms", None)
        extra = getattr(platforms.get(platform), "extra", None) if platforms is not None else None
        return extra if isinstance(extra, dict) else {}

    def _adapter_setting(self, platform, attr: str, extra_key: str, profile):
        """Live adapter's resolved ``attr`` (folds in the ``<PLATFORM>_*`` env override),
        else ``config.extra[extra_key]`` for bare runners with no adapter."""
        adapter = self._authorization_adapter(platform, profile)
        value = getattr(adapter, attr, None) if adapter is not None else None
        if value is None:
            value = self._config_extra(platform).get(extra_key)
        return value

    def _adapter_policy(self, platform, kind: str, profile) -> str:
        """Lowercased effective ``dm_policy`` (open/allowlist/disabled/pairing) or ``group_policy``
        (open/allowlist/disabled) for *kind* in {"dm", "group"}; ``""`` if unknown."""
        if not platform:
            return ""
        return str(self._adapter_setting(platform, f"_{kind}_policy", f"{kind}_policy", profile) or "").strip().lower()

    def _adapter_group_has_sender_allowlist(
        self, platform: Optional[Platform], chat_id: Optional[str], *, profile: Optional[str] = None
    ) -> bool:
        """Whether a per-group sender allowlist (WeCom ``groups.<id>.allow_from``) gated this message:
        a group may be open at the chat level while restricting senders, so reaching the gateway
        means the adapter already checked that list."""
        if not platform or not chat_id:
            return False
        groups = self._adapter_setting(platform, "_groups", "groups", profile)
        if not isinstance(groups, dict):
            return False
        chat_id_str = str(chat_id)
        group_cfg = groups.get(chat_id_str)
        if not isinstance(group_cfg, dict):
            lowered = chat_id_str.lower()
            group_cfg = next(
                (v for k, v in groups.items() if isinstance(k, str) and k.lower() == lowered and isinstance(v, dict)),
                groups.get("*"),
            )
        if not isinstance(group_cfg, dict):
            return False
        sender_allow = group_cfg.get("allow_from") or group_cfg.get("allowFrom")
        if isinstance(sender_allow, str):
            return bool(sender_allow.strip())
        return isinstance(sender_allow, (list, tuple, set)) and any(str(item).strip() for item in sender_allow)

    def _pairing_store_for(self, source: "SessionSource"):
        """Per-profile PairingStore for a source, else the global ``self.pairing_store``."""
        per_profile = getattr(self, "pairing_stores", None) or {}
        profile = getattr(source, "profile", None)
        return per_profile[profile] if profile and profile in per_profile else getattr(self, "pairing_store", None)

    def _adapter_extra_for_source(self, source) -> dict:
        return _adapter_config_extra(self._adapter_for_source(source))

    def _bot_loop_guard_instance(self) -> BotLoopGuard:
        guard = getattr(self, "_bot_loop_guard", None)
        if guard is None:
            with _BOT_LOOP_GUARD_INIT_LOCK:
                guard = getattr(self, "_bot_loop_guard", None)
                if guard is None:
                    guard = self._bot_loop_guard = BotLoopGuard()
        return guard

    def _bot_loop_guard_conversation(self, source: SessionSource) -> tuple:
        # One budget per conversation, not per sender pair: a per-pair key would hand N bots N budgets.
        platform = source.platform.value if source.platform else ""
        return (self._adapter_profile_for_source(source) or "", platform, str(source.chat_id or ""))

    def _admit_bot_message(self, source: SessionSource) -> bool:
        """Count one authorized bot-authored inbound message. False when it trips the budget or the chat is cooling down.
        The inbound handler calls this once per message; ``_is_user_authorized`` only peeks because it is asked several times."""
        if not getattr(source, "is_bot", False):
            return True
        allowed, state = self._bot_loop_guard_instance().admit(self._bot_loop_guard_conversation(source))
        if state == "tripped":
            logger.warning(
                "Bot loop guard is dropping bot messages in %s chat %s: bot %s sent one message too many "
                "for the window, cooling down (gateway.bot_loop_guard in config.yaml).",
                source.platform.value if source.platform else "", source.chat_id, source.user_id,
            )
        return allowed

    def _is_user_authorized(self, source: SessionSource, *, allow_adapter_delegation: bool = True) -> bool:
        """Whether a user may use the bot.

        Order: trusted-upstream delegation, chat-scoped group allowlists, ``{PLATFORM}_ALLOW_BOTS``,
        per-platform allow-all, adapter role auth, pairing store, env/config allowlists,
        ``GATEWAY_ALLOW_ALL_USERS``, default deny. A bot-authored message that any of these admits
        is still refused while its chat's loop guard is cooling down.
        """
        if not self._principal_authorized(source, allow_adapter_delegation=allow_adapter_delegation):
            return False
        if not getattr(source, "is_bot", False):
            return True
        # The guard judges the final verdict: a chat allowlist admits a bot before the ALLOW_BOTS block runs.
        return not self._bot_loop_guard_instance().blocked(self._bot_loop_guard_conversation(source))

    def _principal_authorized(self, source: SessionSource, *, allow_adapter_delegation: bool) -> bool:
        """The allowlist verdict alone, before the bot loop guard.

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
        # runtimes -- see _adapter_profile_for_source's docstring). is_authorized()
        # only receives source.profile from its caller, so this wrapper resolves
        # the transport-owning profile once up front and passes it through
        # explicitly to every adapter-policy lookup below instead of letting the
        # lambdas use whatever profile is_authorized() happens to pass them (raw
        # source.profile, which fails closed for a routed adapter with no
        # same-named entry in _profile_adapters).
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
            dm_check = getattr(adapter, "_is_dm_allowed", None) if adapter is not None else None
            if not callable(dm_check):
                return None
            return bool(dm_check(uid))

        def _adapter_group_allowed_chats(platform: Optional[Platform], profile: Optional[str]) -> set[str]:
            """config.yaml ``extra.group_allowed_chats`` fallback: the Telegram
            observe-unmentioned mode strips user_id from triggered group
            messages, so the env-var-only check misses config.yaml-configured
            allowlists."""
            with contextlib.suppress(Exception):
                return _coerce_allow_set(self._adapter_extra_for_source(source).get("group_allowed_chats"))
            return set()

        def _adapter_allow_from(platform: Optional[Platform], profile: Optional[str], is_group: bool) -> set[str]:
            """config.yaml-only allowlist fallback (``extra.allow_from`` /
            ``group_allow_from``) for adapters (e.g. Telegram) that gate access
            at intake without overriding ``enforces_own_access_policy``."""
            adapter = self._adapter_for_source(source)
            if adapter is None:
                return set()
            extra = _adapter_config_extra(adapter)
            adapter_allow = extra.get("group_allow_from" if is_group else "allow_from")
            if not adapter_allow:
                # Plugin platforms whose registry entry declares
                # ``allowed_users_env`` (e.g. Buzz) carry the same
                # operator-configured allowlist in
                # ``PlatformConfig.extra.allowed_users``. Under multiplex the
                # YAML→env bridge is first-writer-wins, so only the default
                # profile's list ever reaches the env var read elsewhere;
                # fall back to the live (profile-routed) adapter's own config
                # so a secondary profile's allowlist authorizes its users
                # (#98738 / #82871).
                entry = _registry_entry(platform)
                if entry and getattr(entry, "allowed_users_env", None):
                    adapter_allow = extra.get("allowed_users")
            allowed = _coerce_allow_set(adapter_allow)
            normalize = getattr(adapter, "normalize_user_id", None)
            if callable(normalize):
                # Ids and allowlist entries may use different spellings of the
                # same principal (e.g. Buzz hex pubkeys vs npubs).
                allowed = {normalize(entry) or entry for entry in allowed}
            return allowed

        def _adapter_resolved_allowlist_user_ids(platform: Optional[Platform], profile: Optional[str]):
            """Live adapter's resolved numeric allowlist, or None.

            Best-effort: an adapter mid-reconnect (or any resolver error)
            must not break authorization for senders the env allowlist
            already covers, so a raise here is swallowed rather than
            propagated.
            """
            adapter = None
            with contextlib.suppress(Exception):
                adapter = self._adapter_for_source(source)
            resolver = getattr(adapter, "resolved_allowlist_user_ids", None)
            if not callable(resolver):
                return None
            with contextlib.suppress(Exception):
                return resolver()
            return None

        return is_authorized(
            source,
            # Route through the per-profile PairingStore lookup (multiplex
            # gateways isolate each profile's whitelist) rather than a flat
            # ``self.pairing_store`` -- and only evaluated lazily, once
            # is_authorized() actually calls this callable, so a branch
            # resolved earlier (e.g. the chat-scoped group allowlist) never
            # touches ``self.pairing_store`` on a bare runner that never set it.
            pairing_is_approved=lambda platform_name, uid: (
                lambda store: store is not None and store.is_approved(platform_name, uid)
            )(self._pairing_store_for(source)),
            allow_adapter_delegation=allow_adapter_delegation,
            adapter_authorization_is_upstream=lambda platform, profile: (
                self._adapter_flag(platform, "authorization_is_upstream", adapter_profile)
            ),
            adapter_enforces_own_access_policy=lambda platform, profile: (
                self._adapter_flag(platform, "enforces_own_access_policy", adapter_profile)
            ),
            adapter_dm_policy=lambda platform, profile: self._adapter_policy(platform, "dm", adapter_profile),
            adapter_group_policy=lambda platform, profile: self._adapter_policy(platform, "group", adapter_profile),
            adapter_group_has_sender_allowlist=lambda platform, chat_id, profile: (
                self._adapter_group_has_sender_allowlist(platform, chat_id, profile=adapter_profile)
            ),
            adapter_group_allowed_chats=_adapter_group_allowed_chats,
            adapter_allow_from=_adapter_allow_from,
            adapter_dm_is_allowed=_adapter_dm_is_allowed,
            adapter_resolved_allowlist_user_ids=_adapter_resolved_allowlist_user_ids,
            on_legacy_group_users_warning=_warn_legacy_group_users,
            # Preserve the profile-scoped secret_scope lookup on these env
            # reads (``_auth_env`` / ``_platform_gate_env`` prefer the
            # multiplex profile's secret_scope value over a bare
            # ``os.getenv``) so live-gateway behavior stays identical to
            # this wrapper's pre-extraction inline body instead of silently
            # falling back to is_authorized's plain-``os.getenv`` default.
            env_get=_auth_env,
            platform_gate_env=_platform_gate_env,
        )

    def _get_unauthorized_dm_behavior(self, platform: Optional[Platform], *, profile: Optional[str] = None) -> str:
        """How unauthorized DMs are handled ("pair" / "ignore") for a platform.

        Order: explicit per-platform config; Email → "ignore" (inboxes hold arbitrary mail); explicit
        non-default global; adapter dm_policy (pairing → "pair", allowlist/disabled → "ignore"); any
        configured allowlist → "ignore" (spamming unknown contacts with codes is noisy and leaks); else "pair".

        1. 2. Email defaults to ``"ignore"`` unless explicitly opted into pairing. 3. Explicit global
        ``unauthorized_dm_behavior`` in config — wins for chat-shaped platforms when no per-platform
        override is set. 4. When an adapter-level DM policy opts into pairing or silent drop, honor it. 5.
        When an allowlist (``PLATFORM_ALLOWED_USERS``, ``PLATFORM_GROUP_ALLOWED_USERS`` /
        ``PLATFORM_GROUP_ALLOWED_CHATS``, or ``GATEWAY_ALLOWED_USERS``) is configured, default to
        ``"ignore"`` — the allowlist signals that the owner has deliberately restricted access; spamming
        unknown contacts with pairing codes is both noisy and a potential info-leak. (#9337) 6.
        """
        config = getattr(self, "config", None)
        if (
            config and hasattr(config, "get_unauthorized_dm_behavior") and platform
            and "unauthorized_dm_behavior" in self._config_extra(platform)
        ):
            return config.get_unauthorized_dm_behavior(platform)
        if platform == Platform.EMAIL:
            return "ignore"
        if config and hasattr(config, "unauthorized_dm_behavior") and config.unauthorized_dm_behavior != "pair":
            return config.unauthorized_dm_behavior

        allowlist_keys = ["GATEWAY_ALLOWED_USERS"]
        if platform:
            dm_policy = self._adapter_policy(platform, "dm", profile)
            if not dm_policy:
                dm_policy = str(self._config_extra(platform).get("dm_policy") or "").strip().lower()
            if dm_policy == "pairing":
                return "pair"
            if dm_policy in {"allowlist", "disabled"}:
                return "ignore"
            # Historical: Yuanbao is absent from this allowlist-aware default.
            env_key = "" if platform == Platform.YUANBAO else _ALLOWED_USERS_ENV.get(platform, "")
            allowlist_keys = [env_key, _GROUP_USER_ENV.get(platform), _GROUP_CHAT_ENV.get(platform), *allowlist_keys]
        if any(key and _platform_gate_env(key).strip() for key in allowlist_keys):
            return "ignore"
        return "pair"

"""User-authorization mixin for ``GatewayRunner``: may this user/chat talk to the agent,
the per-adapter DM policy, and the unauthorized-DM behavior.

``gateway.run`` is never imported at module import time (cycle); the one method that logs
imports its ``logger`` lazily so records keep the ``"gateway.run"`` name.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional

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


class GatewayAuthorizationMixin:
    """User/chat authorization methods for ``GatewayRunner``."""

    # ``getattr(self, ...)`` throughout: test helpers build bare runners via ``object.__new__``
    # without ``adapters`` / ``config``.

    def _primary_adapters(self) -> dict:
        return getattr(self, "adapters", None) or {}

    def _profile_adapters_map(self) -> dict:
        return getattr(self, "_profile_adapters", None) or {}

    def _authorization_adapter(self, platform: Optional[Platform], profile: Optional[str] = None):
        """Live adapter whose intake policy gates authorization.

        Secondary-profile adapters live in ``_profile_adapters[profile]``; the primary profile owns
        ``self.adapters``. ``_profile_adapters`` is consulted BEFORE the active profile name: multiplex
        turns override ``HERMES_HOME`` so ``_active_profile_name()`` reports the secondary profile
        mid-turn, and treating it as primary would hand it the default bot.
        """
        if not platform:
            return None
        profile_name = (profile or "").strip() or None
        if profile_name and profile_name != "default":
            profile_adapters = self._profile_adapters_map()
            if profile_name in profile_adapters:
                return profile_adapters[profile_name].get(platform)
            # Identity captured at construction, not the per-turn HERMES_HOME-derived name.
            primary_profile = getattr(self, "_primary_profile_name", None)
            if not primary_profile:
                with contextlib.suppress(Exception):
                    primary_profile = self._active_profile_name()
            if profile_name == primary_profile:
                return self._primary_adapters().get(platform)
            # Fail closed: a secondary profile whose adapter failed to connect must NOT
            # fall back to the default profile's adapter (replies out the wrong bot).
            return None
        return self._primary_adapters().get(platform)

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

    def _own_policy_authorizes(self, source, user_id, is_group, adapter_profile) -> Optional[bool]:
        """Own-policy adapter verdict when no env allowlist exists; None = no verdict.

        Trusted only when the effective policy for THIS chat type is ``allowlist``: ``open`` forwards
        EVERY sender (the fail-open SECURITY.md §2.6 forbids), ``disabled`` never forwards, ``pairing``
        forwards unpaired DMs for the handshake (already denied by the pairing-store check).
        Anything else → default-deny.
        """
        if is_group and self._adapter_group_has_sender_allowlist(source.platform, source.chat_id, profile=adapter_profile):
            return True
        if self._adapter_policy(source.platform, "group" if is_group else "dm", adapter_profile) != "allowlist":
            return None
        # Re-check DMs via the live adapter's ``_is_dm_allowed`` when present: pairing revoke can clear
        # WHATSAPP_ALLOWED_USERS while a construction-time snapshot would keep authorizing until
        # restart. Others keep the historical rubber-stamp.
        if not is_group:
            adapter = self._authorization_adapter(source.platform, profile=adapter_profile)
            dm_check = getattr(adapter, "_is_dm_allowed", None) if adapter is not None else None
            if callable(dm_check):
                return bool(dm_check(user_id))
        return True

    def _adapter_extra_allowlist_authorizes(self, source, user_id, is_group) -> bool:
        """Adapters (e.g. Telegram) that gate via config.extra.allow_from / group_allow_from
        without setting enforces_own_access_policy."""
        adapter = self._adapter_for_source(source)
        if adapter is None:
            return False
        extra = _adapter_config_extra(adapter)
        adapter_allow = extra.get("group_allow_from" if is_group else "allow_from")
        if not adapter_allow and is_group and source.platform == Platform.DISCORD:
            # Discord exposes ONE global ``allow_from`` sender list covering DMs
            # and guild channels. Under multiplex the YAML→env bridge
            # deliberately skips process-global env writes, so channel auth must
            # consult that adapter-scoped value too.
            adapter_allow = extra.get("allow_from")
        if not adapter_allow:
            # Plugin platforms (Buzz, DingTalk) spell their env allowlist as ``extra.allowed_users``;
            # under multiplex only the default profile's list reaches the env (first-writer-wins
            # bridge), so read the live adapter's.
            entry = _registry_entry(source.platform)
            if entry and entry.allowed_users_env:
                # Buzz) carry the same operator-configured allowlist in
                # ``PlatformConfig.extra.allowed_users``. An absent/empty entry changes nothing here — the
                # default-deny below still applies. See #82871, #98738.
                adapter_allow = extra.get("allowed_users")
        if not adapter_allow:
            return False
        allowed = _coerce_allow_set(adapter_allow)
        normalize = getattr(adapter, "normalize_user_id", None)
        if callable(normalize):
            # Ids and entries may spell the same principal differently (Buzz hex vs npub).
            allowed = {normalize(entry) or entry for entry in allowed}
        return _allows(allowed, user_id)

    def _adapter_resolved_allowlist_ids(self, source) -> set[str]:
        """IDs an adapter resolved from username-shaped allowlist entries at connect time (Discord).

        The per-turn .env hot-reload restores RAW usernames, so from the second turn on the env
        allowlist holds usernames while user_id is numeric. Never a widening: the empty-allowlist
        branch already returned and adapters only resolve operator-written entries. Only called with
        a non-empty platform allowlist so group/global-only configs never consult adapter memory;
        type-checked so mocks cannot auto-truthy in.
        """
        adapter = resolved_ids = None
        with contextlib.suppress(Exception):
            adapter = self._adapter_for_source(source)
        resolver = getattr(adapter, "resolved_allowlist_user_ids", None)
        if callable(resolver):
            with contextlib.suppress(Exception):
                resolved_ids = resolver()
        if not isinstance(resolved_ids, (set, frozenset, list, tuple)):
            return set()
        return {s for e in resolved_ids if isinstance(e, (str, int)) and (s := str(e).strip())}

    def _chat_scoped_grant(self, source, adapter_profile, is_group: bool, allow_adapter_delegation: bool) -> bool:
        """Grants that need no ``user_id`` (checked before the no-user-id guard)."""
        # Trusted-upstream delegation (relay): the connector authenticates this gateway's WS and
        # resolves owner bindings BEFORE delivering, so there is no local RELAY_ALLOWED_USERS. Not a
        # fail-open: fires only for events actually delivered over the relay WS
        # (``delivered_via_upstream_relay``) or whose adapter declares ``authorization_is_upstream``.
        # The delivery marker is PRIMARY because a relayed message carries the UNDERLYING platform,
        # not ``Platform.RELAY``. ``is True``: a MagicMock stand-in must not auto-truthy into authz.
        if allow_adapter_delegation and (
            source.delivered_via_upstream_relay is True
            or self._adapter_flag(source.platform, "authorization_is_upstream", adapter_profile)
        ):
            return True
        # Chat-scoped group allowlists must work with ``user_id is None`` (anonymous admins,
        # sender_chat posts, channel broadcasts).
        if is_group and source.chat_id:
            chat_allowlist_env = _GROUP_CHAT_ENV.get(source.platform, "")
            if chat_allowlist_env and _allows(_coerce_allow_set(_platform_gate_env(chat_allowlist_env)), source.chat_id):
                return True
            # config.yaml fallback (``extra.group_allowed_chats``): Telegram observe-unmentioned mode
            # strips user_id, so the env-only check above misses it.
            with contextlib.suppress(Exception):
                adapter_group_allowed = self._adapter_extra_for_source(source).get("group_allowed_chats")
                if adapter_group_allowed and _allows(_coerce_allow_set(adapter_group_allowed), source.chat_id):
                    return True
        # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist (Slack Workflow Builder
        # posts arrive with user=None).
        if getattr(source, "is_bot", False):
            allow_bots_var = _ALLOW_BOTS_ENV.get(source.platform)
            if allow_bots_var and _platform_gate_env(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
                return True
        return False

    def _legacy_telegram_chat_grant(self, source, group_user_allowlist: str) -> bool:
        """TELEGRAM_GROUP_ALLOWED_USERS was once (mis)used as a chat-ID allowlist; "-"-prefixed
        values are chat IDs, honor them and warn once."""
        from gateway.run import logger
        legacy_chat_ids = {v.strip() for v in group_user_allowlist.split(",") if v.strip().startswith("-")}
        if not legacy_chat_ids:
            return False
        if not getattr(self, "_warned_telegram_group_users_legacy", False):
            logger.warning(
                "TELEGRAM_GROUP_ALLOWED_USERS contains chat-ID-shaped values "
                "(%s). Treating them as chat IDs for backward compatibility. "
                "Move chat IDs to TELEGRAM_GROUP_ALLOWED_CHATS — the _USERS var "
                "is now for sender user IDs.",
                ",".join(sorted(legacy_chat_ids)),
            )
            self._warned_telegram_group_users_legacy = True
        return source.chat_id in legacy_chat_ids

    def _is_user_authorized(self, source: SessionSource, *, allow_adapter_delegation: bool = True) -> bool:
        """Whether a user may use the bot.

        Order: trusted-upstream delegation, chat-scoped group allowlists, ``{PLATFORM}_ALLOW_BOTS``,
        per-platform allow-all, adapter role auth, pairing store, env/config allowlists,
        ``GATEWAY_ALLOW_ALL_USERS``, default deny.
        """
        # HA events are system-generated (HASS_TOKEN); webhook events are HMAC-verified.
        if source.platform in {Platform.HOMEASSISTANT, Platform.WEBHOOK}:
            return True

        adapter_profile = self._adapter_profile_for_source(source)
        is_group = source.chat_type in _GROUP_CHAT_TYPES
        is_group_or_forum = source.chat_type in _GROUP_FORUM_TYPES
        if self._chat_scoped_grant(source, adapter_profile, is_group, allow_adapter_delegation):
            return True
        user_id = source.user_id
        if not user_id:
            return False

        platform_allow_env = _ALLOWED_USERS_ENV.get(source.platform, "")
        platform_allow_all_var = _ALLOW_ALL_ENV.get(source.platform, "")
        if source.platform not in _ALLOWED_USERS_ENV:
            entry = _registry_entry(source.platform)
            with contextlib.suppress(Exception):
                platform_allow_env = getattr(entry, "allowed_users_env", "") or platform_allow_env
                platform_allow_all_var = getattr(entry, "allow_all_env", "") or platform_allow_all_var
        if platform_allow_all_var and _env_truthy(platform_allow_all_var):
            return True
        # Adapter-verified role auth (Discord DISCORD_ALLOWED_ROLES). ``is True``: no MagicMock pass.
        if allow_adapter_delegation and getattr(source, "role_authorized", False) is True:
            return True
        # Pairing store: a first-class grant created only by an operator approving a code. Honored as
        # a UNION with the allowlist (approval also mirrors into it).
        pairing_store = self._pairing_store_for(source)
        if pairing_store is not None and pairing_store.is_approved(source.platform.value if source.platform else "", user_id):
            return True

        platform_allowlist = _auth_env(platform_allow_env)
        group_user_allowlist = _auth_env(_GROUP_USER_ENV.get(source.platform, "")) if is_group_or_forum else ""
        group_chat_allowlist = _auth_env(_GROUP_CHAT_ENV.get(source.platform, "")) if is_group_or_forum else ""
        global_allowlist = _auth_env("GATEWAY_ALLOWED_USERS")

        if not (platform_allowlist or group_user_allowlist or group_chat_allowlist or global_allowlist):
            # No env allowlist: own-policy adapters gate at intake (see _own_policy_authorizes).
            if allow_adapter_delegation and self._adapter_flag(source.platform, "enforces_own_access_policy", adapter_profile):
                verdict = self._own_policy_authorizes(source, user_id, is_group, adapter_profile)
                if verdict is not None:
                    return verdict
            if self._adapter_extra_allowlist_authorizes(source, user_id, is_group):
                return True
            return _env_truthy("GATEWAY_ALLOW_ALL_USERS")

        if is_group_or_forum and source.chat_id:
            # Telegram group traffic authorized by chat ID (TELEGRAM_GROUP_ALLOWED_USERS gates the sender).
            if group_chat_allowlist and _allows(_coerce_allow_set(group_chat_allowlist), source.chat_id):
                return True
            if (
                source.platform == Platform.TELEGRAM and group_user_allowlist
                and self._legacy_telegram_chat_grant(source, group_user_allowlist)
            ):
                return True

        # TELEGRAM_GROUP_ALLOWED_USERS is group-scoped (no DM access); TELEGRAM_ALLOWED_USERS is platform-wide.
        allowed_ids = (
            _coerce_allow_set(platform_allowlist)
            | _coerce_allow_set(group_user_allowlist)
            | _coerce_allow_set(global_allowlist)
        )
        if platform_allowlist:
            allowed_ids |= self._adapter_resolved_allowlist_ids(source)
        return "*" in allowed_ids or _principal_matches_allowlist(source, user_id, allowed_ids)

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

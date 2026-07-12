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
from typing import Optional

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.whatsapp_identity import (
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)


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
        """
        if not platform:
            return None
        profile_name = (profile or "").strip() or None
        if profile_name and profile_name != "default":
            profile_adapters = getattr(self, "_profile_adapters", None) or {}
            if profile_name in profile_adapters:
                return profile_adapters[profile_name].get(platform)
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
        # ``getattr`` guards test fixtures that build a bare source via
        # SimpleNamespace and omit ``profile`` (see AGENTS.md pitfall #17).
        return self._authorization_adapter(
            getattr(source, "platform", None),
            getattr(source, "profile", None),
        )

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

    def _is_user_authorized(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny
        """
        from gateway.run import logger
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
        if source.delivered_via_upstream_relay is True or self._adapter_authorization_is_upstream(
            source.platform,
            profile=source.profile,
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
                raw_chat_allowlist = os.getenv(chat_allowlist_env, "").strip()
                if raw_chat_allowlist:
                    allowed_group_ids = {
                        cid.strip()
                        for cid in raw_chat_allowlist.split(",")
                        if cid.strip()
                    }
                    if "*" in allowed_group_ids or source.chat_id in allowed_group_ids:
                        return True

        # Discord permits channel-only authorization only when no sender grant
        # is configured. The adapter performs the channel check at intake;
        # repeat it here because the gateway deliberately re-authorizes every
        # event before persistence/dispatch. Do not return early when a Discord
        # user/role or global user allowlist exists: those restrictions must be
        # evaluated below. Threads inherit an allowed parent channel.
        if source.platform == Platform.DISCORD and source.chat_type != "dm":
            sender_grant_configured = any(
                os.getenv(env_var, "").strip()
                for env_var in (
                    "DISCORD_ALLOWED_USERS",
                    "DISCORD_ALLOWED_ROLES",
                    "GATEWAY_ALLOWED_USERS",
                )
            )
            raw_allowed_channels = os.getenv("DISCORD_ALLOWED_CHANNELS", "").strip()
            if raw_allowed_channels and not sender_grant_configured:
                allowed_channel_ids = {
                    channel_id.strip()
                    for channel_id in raw_allowed_channels.split(",")
                    if channel_id.strip()
                }
                source_channel_ids = {
                    str(channel_id)
                    for channel_id in (source.chat_id, source.parent_chat_id)
                    if channel_id
                }
                if "*" in allowed_channel_ids or source_channel_ids & allowed_channel_ids:
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
            if allow_bots_var and os.getenv(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
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
        if platform_allow_all_var and os.getenv(platform_allow_all_var, "").lower() in {"true", "1", "yes"}:
            return True

        # Adapter-verified role auth: the Discord adapter already confirmed the
        # user holds a role in DISCORD_ALLOWED_ROLES before dispatching the message.
        # Compare with ``is True`` so the real bool field authorizes while a
        # MagicMock source (test fixtures using ``object.__new__`` runners with
        # mock sources) does not auto-truthy through this gate (see pitfall #13).
        if getattr(source, "role_authorized", False) is True:
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
        # inbound message/approval-button gate, not this gate; that gate is
        # fixed separately.)
        # In multiplex gateways, route to the per-profile PairingStore so each
        # profile's whitelist is isolated; falls back to the global store when
        # the source has no profile or the profile isn't registered.
        platform_name = source.platform.value if source.platform else ""
        pairing_store = self._pairing_store_for(source)
        if pairing_store is not None and pairing_store.is_approved(platform_name, user_id):
            return True

        # Check platform-specific and global allowlists
        platform_allowlist = os.getenv(platform_env_map.get(source.platform, ""), "").strip()
        group_user_allowlist = ""
        group_chat_allowlist = ""
        if source.chat_type in {"group", "forum"}:
            group_user_allowlist = os.getenv(platform_group_user_env_map.get(source.platform, ""), "").strip()
            group_chat_allowlist = os.getenv(platform_group_chat_env_map.get(source.platform, ""), "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()

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
            if self._adapter_enforces_own_access_policy(
                source.platform,
                profile=source.profile,
            ):
                if source.chat_type in {"group", "forum", "channel"}:
                    effective_policy = self._adapter_group_policy(
                        source.platform,
                        profile=source.profile,
                    )
                    if self._adapter_group_has_sender_allowlist(
                        source.platform,
                        source.chat_id,
                        profile=source.profile,
                    ):
                        return True
                else:
                    effective_policy = self._adapter_dm_policy(
                        source.platform,
                        profile=source.profile,
                    )
                if effective_policy == "allowlist":
                    return True
            # No allowlists configured -- check global allow-all flag
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}

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
                if not getattr(self, "_warned_telegram_group_users_legacy", False):
                    logger.warning(
                        "TELEGRAM_GROUP_ALLOWED_USERS contains chat-ID-shaped values "
                        "(%s). Treating them as chat IDs for backward compatibility. "
                        "Move chat IDs to TELEGRAM_GROUP_ALLOWED_CHATS — the _USERS var "
                        "is now for sender user IDs.",
                        ",".join(sorted(legacy_chat_ids)),
                    )
                    self._warned_telegram_group_users_legacy = True
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

        return bool(check_ids & allowed_ids)

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
            if os.getenv(platform_env_map.get(platform, ""), "").strip():
                return "ignore"
            for env_key in platform_group_env_map.get(platform, ()):
                if os.getenv(env_key, "").strip():
                    return "ignore"

        if os.getenv("GATEWAY_ALLOWED_USERS", "").strip():
            return "ignore"

        return "pair"

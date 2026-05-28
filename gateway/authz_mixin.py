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

    def _adapter_enforces_own_access_policy(self, platform: Optional[Platform]) -> bool:
        """Whether the adapter for *platform* gates access at intake itself.

        Mirrors ``BasePlatformAdapter.enforces_own_access_policy``. Adapters
        such as WeCom, Weixin, Yuanbao, QQBot, and WhatsApp evaluate their
        documented ``dm_policy`` / ``group_policy`` / ``allow_from`` config before a
        message is dispatched to the gateway, so a message that reaches
        ``_is_user_authorized`` has already been authorized by the adapter.
        Defaults to ``False`` when the adapter is unknown or doesn't expose
        the flag.
        """
        if not platform:
            return False
        # Some test helpers build a bare GatewayRunner via object.__new__ and
        # never set ``adapters``; treat a missing/empty map as "no adapter"
        # rather than raising (see pitfalls.md #17).
        adapters = getattr(self, "adapters", None)
        if not adapters:
            return False
        adapter = adapters.get(platform)
        if adapter is None:
            return False
        return bool(getattr(adapter, "enforces_own_access_policy", False))

    def _adapter_dm_policy(self, platform: Optional[Platform]) -> str:
        """Best-effort read of an own-policy adapter's effective DM policy.

        Returns the lowercased ``dm_policy`` (``"open"`` / ``"allowlist"`` /
        ``"disabled"`` / ``"pairing"``) for *platform*, or ``""`` when unknown.
        Prefers the live adapter's resolved ``_dm_policy`` — which already folds
        in both ``config.extra`` and the ``<PLATFORM>_DM_POLICY`` env var (the
        env var is not always bridged back into ``config.extra``) — and falls
        back to ``config.extra`` for bare runners built without a live adapter.

        Used by ``_is_user_authorized`` to carve ``dm_policy: pairing`` out of
        the adapter-trust shortcut: in pairing mode the adapter forwards the DM
        so the gateway can run its pairing handshake, so "reached the gateway"
        must not be read as "authorized".
        """
        if not platform:
            return ""
        adapters = getattr(self, "adapters", None) or {}
        adapter = adapters.get(platform)
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

        if not user_id:
            return False

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
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
        # Bots admitted by {PLATFORM}_ALLOW_BOTS bypass the human allowlist (#4466).
        platform_allow_bots_map = {
            Platform.DISCORD: "DISCORD_ALLOW_BOTS",
            Platform.FEISHU: "FEISHU_ALLOW_BOTS",
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
        if getattr(source, "role_authorized", False):
            return True

        if getattr(source, "is_bot", False):
            allow_bots_var = platform_allow_bots_map.get(source.platform)
            if allow_bots_var and os.getenv(allow_bots_var, "none").lower().strip() in {"mentions", "all"}:
                return True

        # Check pairing store (always checked, regardless of allowlists)
        platform_name = source.platform.value if source.platform else ""
        if self.pairing_store.is_approved(platform_name, user_id):
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
            # No env allowlists configured. Adapters that own their own
            # config-driven access policy (dm_policy / group_policy /
            # allow_from / group_allow_from) already gated this message at
            # intake — it would not have reached the gateway otherwise — so
            # honor that decision instead of falling through to the
            # env-only default-deny below, which would silently break
            # `dm_policy: open` and config-only allowlists. (#34515)
            if self._adapter_enforces_own_access_policy(source.platform):
                # Exception: `dm_policy: pairing` does NOT authorize at intake.
                # The adapter forwards the DM precisely so the gateway can run
                # its pairing handshake (issue a code, consult the pairing
                # store). The pairing-store approval check above already ran and
                # returned False for this sender, so blanket-trusting the
                # adapter here would silently turn pairing mode into open
                # access. Fall through to default-deny so the unpaired sender is
                # offered a pairing code instead. (Pairing is DM-only; group
                # traffic keeps the adapter-trust path.)
                if not (
                    source.chat_type == "dm"
                    and self._adapter_dm_policy(source.platform) == "pairing"
                ):
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

    def _get_unauthorized_dm_behavior(self, platform: Optional[Platform]) -> str:
        """Return how unauthorized DMs should be handled for a platform.

        Resolution order:
        1. Explicit per-platform ``unauthorized_dm_behavior`` in config — always wins.
        2. Explicit global ``unauthorized_dm_behavior`` in config — wins when no per-platform.
        3. When an allowlist (``PLATFORM_ALLOWED_USERS``,
           ``PLATFORM_GROUP_ALLOWED_USERS`` / ``PLATFORM_GROUP_ALLOWED_CHATS``,
           or ``GATEWAY_ALLOWED_USERS``) is configured, default to ``"ignore"`` —
           the allowlist signals that the owner has deliberately restricted
           access; spamming unknown contacts with pairing codes is both noisy
           and a potential info-leak. (#9337)
        4. No allowlist and no explicit config → ``"pair"`` (open-gateway default).
        """
        config = getattr(self, "config", None)

        # Check for an explicit per-platform override first.
        if config and hasattr(config, "get_unauthorized_dm_behavior") and platform:
            platform_cfg = config.platforms.get(platform) if hasattr(config, "platforms") else None
            if platform_cfg and "unauthorized_dm_behavior" in getattr(platform_cfg, "extra", {}):
                # Operator explicitly configured behavior for this platform — respect it.
                return config.get_unauthorized_dm_behavior(platform)

        # Check for an explicit global config override.
        if config and hasattr(config, "unauthorized_dm_behavior"):
            if config.unauthorized_dm_behavior != "pair":  # non-default → explicit override
                return config.unauthorized_dm_behavior

        # Config-driven dm_policy (WeCom / Weixin / Yuanbao / QQBot). An
        # allowlist or disabled DM policy means the operator restricted access,
        # so unauthorized DMs should be dropped silently rather than answered
        # with a pairing code. An explicit pairing policy opts back into codes.
        if platform and config and hasattr(config, "platforms"):
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

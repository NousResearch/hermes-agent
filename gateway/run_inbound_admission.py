"""Inbound identity, startup and authorisation gates for GatewayRunner."""

from __future__ import annotations

import dataclasses
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_inbound_unauthorized import (
    UnauthorizedOwnerNotifier,
    pairing_code_reply,
    pairing_profile_arg,
    pairing_rate_limited_reply,
    unauthorized_owner_hint,
)
from gateway.session import SessionSource

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.run")


class GatewayInboundAdmissionMixin:
    """Identity, startup, hook and authorisation checks before turn setup."""

    if TYPE_CHECKING:
        _defer_for_startup_restore: Callable[[MessageEvent], bool]
        _delivery_adapter_for: Callable[
            [Optional[SessionSource]], Optional[BasePlatformAdapter]
        ]

    async def _hm_pre_gateway_dispatch_hook(
        self, event: MessageEvent, source: SessionSource
    ) -> Optional[MessageEvent]:
        """Run the ``pre_gateway_dispatch`` plugin hook; None = drop, else the (maybe rewritten) event.
        Results: ``{"action": "skip"}`` → drop; ``{"action": "rewrite", "text"}`` → replace ``event.text``;
        ``allow``/None → normal dispatch. Runs BEFORE auth so plugins can handle unauthorized senders."""
        try:
            from hermes_cli.lifecycle import ainvoke_hook as _ainvoke_hook

            _hook_results = await _ainvoke_hook(
                "pre_gateway_dispatch",
                event=event,
                gateway=self,
                # getattr: bare-runner tests build GatewayRunner via object.__new__ without __init__.
                session_store=getattr(self, "session_store", None),
            )
        except Exception as _hook_exc:
            logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc)
            _hook_results = []

        for _result in _hook_results:
            if not isinstance(_result, dict):
                continue
            _action = _result.get("action")
            if _action == "skip":
                logger.info(
                    "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                    _result.get("reason"),
                    source.platform.value if source.platform else "unknown",
                    source.chat_id or "unknown",
                )
                return None
            if _action == "rewrite":
                _new_text = _result.get("text")
                if isinstance(_new_text, str):
                    event = dataclasses.replace(event, text=_new_text)
                break
            if _action == "allow":
                break
        return event

    async def _hm_offer_pairing_code(self, source: SessionSource) -> None:
        """DM an unauthorized sender a pairing code (rate-limited; groups never reach here)."""
        platform_name = source.platform.value if source.platform else "unknown"
        pairing_store = self._pairing_store_for(source)
        if pairing_store is None:
            logger.error(
                "Cannot offer pairing code on %s: no pairing store", platform_name
            )
            return
        # Rate-limit ALL pairing responses (code or rejection) so a burst of DMs doesn't spam.
        if pairing_store._is_rate_limited(platform_name, source.user_id):
            return
        code = pairing_store.generate_code(
            platform_name, source.user_id, source.user_name or ""
        )
        adapter = self._delivery_adapter_for(source)
        if code:
            reply = pairing_code_reply(
                platform_name, code, pairing_profile_arg(pairing_store)
            )
        else:
            reply = pairing_rate_limited_reply()
        if adapter:
            await adapter.send(source.chat_id, reply)
        if not code:
            # Record rate limit so subsequent messages are silently ignored
            pairing_store._record_rate_limit(platform_name, source.user_id)

    async def _hm_send_unauthorized_decline(self, source: SessionSource) -> None:
        """``decline`` behavior: one short refusal per sender per DECLINE_DEDUPE_SECONDS, then silence
        (#88028). The stamp is written BEFORE the send so a delivery
        hiccup cannot become a decline storm; without a store there is no dedupe state → stay silent."""
        from gateway.config import DEFAULT_UNAUTHORIZED_DM_DECLINE_MESSAGE

        platform_name = source.platform.value if source.platform else "unknown"
        pairing_store = self._pairing_store_for(source)
        if pairing_store is None or pairing_store.has_recent_decline(
            platform_name, source.user_id
        ):
            return
        pairing_store.record_decline(platform_name, source.user_id)
        adapter = self._delivery_adapter_for(source)
        if not adapter:
            return
        config = getattr(self, "config", None)
        text = str(getattr(config, "unauthorized_dm_decline_message", "") or "").strip()
        try:
            await adapter.send(
                source.chat_id, text or DEFAULT_UNAUTHORIZED_DM_DECLINE_MESSAGE
            )
        except Exception:
            logger.warning(
                "Failed to deliver unauthorized-DM decline on %s",
                platform_name,
                exc_info=True,
            )

    async def _hm_report_ignored_dm(self, source: SessionSource) -> None:
        """Unauthorized DM under behaviour ``ignore``: nothing goes to the sender. The owner gets the
        sender's ID and the allowlist fix in the WARNING log and, once per sender, in the home channel."""
        from hermes_constants import display_hermes_home

        platform_name = source.platform.value if source.platform else "unknown"
        hint = unauthorized_owner_hint(
            platform_name,
            source.user_id,
            source.user_name or "",
            hermes_home=display_hermes_home(),
        )
        logger.warning("Unauthorized user (ignored): %s", hint)
        notifier = getattr(self, "_unauthorized_owner_notifier", None)
        if notifier is None:
            notifier = self._unauthorized_owner_notifier = UnauthorizedOwnerNotifier()
        if (
            notifier.first_time(platform_name, source.user_id)
            and getattr(self, "config", None) is not None
        ):
            await notifier.notify(self, source, hint)

    async def _hm_admit_event(
        self, event: MessageEvent
    ) -> Optional[tuple[MessageEvent, SessionSource, bool]]:
        """Ingress gates for ``_handle_message``; None when dropped, else ``(event, source, is_internal)``
        (the ``pre_gateway_dispatch`` hook may have rewritten ``event``)."""
        from gateway.run import _is_slack_ignored_channel

        source = event.source
        # getattr(self, ...) throughout: bare test runners build GatewayRunner via object.__new__.
        _config = getattr(self, "config", None)

        # 🔴 Cross-session leak guard: this per-message task was create_task()'d with a copy of the
        # spawning context, which may carry ANOTHER message's HERMES_SESSION_* ContextVars; until
        # _set_session_env binds ours a subprocess would read the foreign identity. Reset to _UNSET.
        try:
            from gateway.session_context import reset_session_vars

            reset_session_vars()
        except Exception:
            logger.debug("reset_session_vars failed at handler entry", exc_info=True)

        # Identity FIRST. Most adapters canonicalize at their own ingress; internal/voice paths
        # construct SessionSource directly, so this is the shared fail-closed gate. Strict boolean
        # marker: require the literal True so duck-typed test/internal sources with dynamic
        # attributes are not mistaken for a rejection.
        if getattr(_config, "multiplex_profiles", False):
            self._canonicalize(source)
        if getattr(source, "profile_route_rejected", False) is True:
            logger.warning(
                "Dropping inbound message because its explicit profile route "
                "targets an unserved profile"
            )
            return None

        is_internal = bool(
            getattr(event, "internal", False)
        )  # e.g. background-process notifications

        # Ignored-channel guard runs FIRST — before startup-restore queueing, plugin hooks, auth,
        # and session setup — so an ignored channel can never reach pairing/auth/session state.
        _chat_id = getattr(source, "chat_id", None)
        if not is_internal and getattr(source, "platform", None) == Platform.SLACK:
            # The routed adapter's extra carries a secondary profile's own list; ``_config`` is the default's.
            _slack_adapter = None
            with suppress(Exception):
                _slack_adapter = self._intake_adapter_for(source)
        if (
            # See #51899.
            not is_internal
            and getattr(source, "platform", None) == Platform.SLACK
            and _is_slack_ignored_channel(_config, _chat_id, _slack_adapter)
        ):
            logger.info(
                "Dropping Slack message from configured ignored channel %s", _chat_id
            )
            return None

        if self._defer_for_startup_restore(event):
            return None

        if is_internal:
            return event, source, True

        # scale-to-zero: only real user-originated inbound stamps the last-inbound clock;
        # counting internal/system events would keep a genuinely idle gateway awake.
        self._scale_to_zero_note_real_inbound()
        event = await self._hm_pre_gateway_dispatch_hook(event, source)
        if event is None:
            return None
        source = event.source

        if not self._is_user_authorized_for_source(source):
            if source.user_id is None:
                # No user identity (Telegram service messages, channel forwards, anonymous admin
                # posts, sender_chat): can't be paired but may be authorized via a chat allowlist.
                logger.debug(
                    "Ignoring message with no user_id from %s", source.platform.value
                )
                return None
            # DMs get a pairing code or a one-time decline, groups are ignored. A bot cannot pair, and
            # answering one mid-cooldown is outbound traffic.
            pairable_dm = source.chat_type == "dm" and not getattr(
                source, "is_bot", False
            )
            behavior = (
                self._get_unauthorized_dm_behavior(
                    source.platform, profile=source.profile
                )
                if pairable_dm
                else None
            )
            if behavior == "pair":
                logger.warning(
                    "Unauthorized user: %s (%s) on %s",
                    source.user_id,
                    source.user_name,
                    source.platform.value,
                )
                await self._hm_offer_pairing_code(source)
            elif behavior == "decline":
                logger.warning(
                    "Unauthorized user: %s (%s) on %s",
                    source.user_id,
                    source.user_name,
                    source.platform.value,
                )
                await self._hm_send_unauthorized_decline(source)
            elif pairable_dm:
                await self._hm_report_ignored_dm(source)
            else:
                logger.warning(
                    "Unauthorized user: %s (%s) on %s",
                    source.user_id,
                    source.user_name,
                    source.platform.value,
                )
            return None
        # The busy path charged this event on arrival; a drained follow-up must not pay twice.
        if not getattr(
            event, "_bot_loop_admitted", False
        ) and not self._admit_bot_message_for_source(source):
            return None
        return event, source, False

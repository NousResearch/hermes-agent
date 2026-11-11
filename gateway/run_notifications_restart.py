"""Restart-completion notices for GatewayRunner: the requester's "restarted" reply and its replay.

Split out of ``gateway/run_notifications.py``; ``GatewayNotificationsMixin`` inherits it, so these
stay bound onto ``GatewayRunner`` through the MRO. ``gateway.run`` internals are imported lazily
inside method bodies (import cycle), so ``patch("gateway.run.X")`` keeps intercepting them.

A notice that could not be sent yet is kept, never consumed (#66589). Telegram refuses sends with
``send_path_degraded`` until its first getUpdates of the polling generation succeeds, and a platform
can still be down when the boot pass runs. The ``.restart_notify.json`` marker used to be unlinked
in ``finally`` either way, so the /restart confirmation was lost. Both markers (this one and the
planned-restart ``.restart_pending.json``) are now replayed when the platform reconnects and when an
adapter's send path heals in place (``_schedule_restart_notice_replay``). A kept requester marker
expires after ``_RESTART_NOTICE_TTL_SECONDS``, so a chat whose platform never comes back cannot leave
it behind for good (it also arms the redelivered-/restart guard of the next boot).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
from typing import Optional

from agent.i18n import t
from gateway.config import Platform
from gateway.run_shutdown import _send_error, _send_failed

logger = logging.getLogger("gateway.run")

# Long enough for a slow drain plus a platform's reconnect ladder; short enough that a requester
# whose platform never comes back is not answered hours later.
_RESTART_NOTICE_TTL_SECONDS = 30 * 60


def _notice_expired(path) -> bool:
    try:
        return time.time() - path.stat().st_mtime > _RESTART_NOTICE_TTL_SECONDS
    except OSError:
        return True


class GatewayRestartNoticeMixin:
    """``_send_restart_notification`` and the replay that retries it (and the planned-restart notice)."""

    _restart_notice_lock: Optional[asyncio.Lock] = None
    _restart_notice_context: Optional[contextvars.Context] = None

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back.

        The marker is consumed when the notice is delivered, suppressed, or provably undeliverable.
        It is kept while the requester's platform has no live transport or refuses the send because its
        send path is not ready (Telegram's ``send_path_degraded`` before its first getUpdates),
        so the reconnect / send-path-healed replay can deliver it. One lock serializes the boot pass
        against a replay, so the notice cannot go out twice."""
        from gateway.run import _hermes_home
        if self._restart_notice_lock is None:
            self._restart_notice_lock = asyncio.Lock()
            self._restart_notice_context = contextvars.copy_context()  # the boot pass's, for the replay
        async with self._restart_notice_lock:
            notify_path = _hermes_home / ".restart_notify.json"
            if not notify_path.exists():
                return None
            keep = False
            try:
                if _notice_expired(notify_path):
                    logger.warning("Restart notification dropped: still undelivered after %d minutes",
                                   _RESTART_NOTICE_TTL_SECONDS // 60)
                    return None
                target, keep = await self._deliver_restart_notification(
                    json.loads(notify_path.read_text(encoding="utf-8-sig")))
                return target
            except Exception:  # health: allow BLE001 -- moved boundary: a failed notice never breaks the boot pass
                logger.warning("Restart notification failed", exc_info=True)
                return None
            finally:
                if not keep:
                    notify_path.unlink(missing_ok=True)

    async def _deliver_restart_notification(self, data: dict) -> tuple[Optional[tuple[str, str, Optional[str]]], bool]:
        """One send of the requester notice: ``(delivered target or None, keep the marker)``."""
        from gateway.delivery import resolve_delivery_transport
        from gateway.run import _non_conversational_metadata
        platform_str = data.get("platform")
        chat_id = data.get("chat_id")
        thread_id = data.get("thread_id")
        if not platform_str or not chat_id:
            return None, False
        platform = Platform(platform_str)
        platform_cfg = self.config.platforms.get(platform)
        if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
            logger.info("Restart notification suppressed: %s has gateway_restart_notification=false", platform_str)
            return None, False
        # Relay-aware transport over the REQUESTER'S profile adapter map; ``self.adapters`` is the
        # default profile's, so a secondary's "restarted" notice would leave through the wrong bot.
        transport = resolve_delivery_transport(
            platform, self.config, self._adapters_for_profile(self._marker_profile(data)))
        if transport is None:
            logger.info("Restart notification pending: no live transport for %s yet", platform_str)
            return None, True
        metadata = self._pending_marker_metadata(platform, chat_id, data, transport.adapter)
        if data.get("delivered_via_upstream_relay") is True:
            metadata = dict(metadata or {})
            for field in ("user_id", "scope_id"):
                if data.get(field):
                    metadata[field] = str(data[field])
        result = await transport.send(
            platform, str(chat_id), t("gateway.startup.restarted"),
            metadata=_non_conversational_metadata(metadata, platform=platform),
        )
        # adapter.send() catches provider errors (e.g. "Chat not found") and returns
        # SendResult(success=False) rather than raising, so inspect the result before claiming success.
        if _send_failed(result):
            # Kept only for the refusal the delivery ledger also leaves to a reconnect (the same decider):
            # ``retryable`` alone also covers permanent refusals such as "chat not found".
            from gateway.delivery_ledger import is_reconnect_only
            keep = is_reconnect_only(_send_error(result))
            logger.warning(
                "Restart notification to %s:%s was not delivered: %s%s", platform_str, chat_id,
                _send_error(result), " (kept; retried when the platform's send path is ready)" if keep else "",
            )
            return None, keep
        logger.info("Sent restart notification to %s:%s", platform_str, chat_id)
        return (str(platform_str), str(chat_id), str(thread_id) if thread_id else None), False

    async def _replay_restart_notices(self, *, requester: bool) -> None:
        if requester:
            await self._send_restart_notification()
        await self._replay_pending_planned_restart_notification()

    def _schedule_restart_notice_replay(self) -> None:
        """Replay the owed restart notices (requester + planned-restart home channels) in the
        background: after a reconnect, and when an adapter's send path heals in place.

        Delivery must not hold up adapter recovery or other platforms' reconnects. The task runs in
        the boot pass's context (``_restart_notice_context``, captured by the boot pass's first send):
        both markers are host-wide (they span every served profile and pick each target's own bot), and
        the trigger may fire inside a secondary profile's adapter task, whose scope must not leak in."""
        from gateway.run import _planned_restart_notification_pending, _restart_notification_pending
        context = self._restart_notice_context
        # Before the boot pass the requester marker is its to send: it reads ``_booted_from_restart`` from it first.
        requester = context is not None and _restart_notification_pending()
        if not (requester or _planned_restart_notification_pending()):
            return
        task = self._retain_background_task(asyncio.get_running_loop().create_task(
            self._replay_restart_notices(requester=requester), context=context.copy() if context is not None else None))
        task.add_done_callback(self._late_failure_callback("restart notification replay failed"))

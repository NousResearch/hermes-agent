"""Nextcloud Notifications + Activity polling service.

Polls both the NC Notifications API (ETag-optimized) and the Activity API
to catch all Nextcloud events. Classifies events via configurable rules
and routes them to a target platform adapter for agent processing.

File handling (download, sync, local storage) is NOT part of this service.
It will be handled by a separate NextcloudFilesService (pr/nextcloud-files).
"""
import asyncio
import logging
import os
from typing import Optional

import httpx

from .base import BaseService, ServiceEvent
from .store import NotificationStore

logger = logging.getLogger(__name__)


def _extract_sender(raw: dict) -> str:
    """Extract the actor/sender from a notification or activity dict.

    Notifications API stores the actor in subjectRichParameters.actor.id.
    Activity API stores it in subject_rich[1].actor.id or subject_rich[1].user.id.
    The raw 'user' field is always the notification RECIPIENT (hermes), not the sender.
    Falls back to raw['user'] only if no actor can be found.
    """
    # Notifications API: subjectRichParameters is a dict
    srp = raw.get("subjectRichParameters")
    if isinstance(srp, dict):
        actor_id = srp.get("actor", {}).get("id", "")
        if actor_id:
            return actor_id

    # Activity API: subject_rich is a list [template, params_dict]
    sr = raw.get("subject_rich")
    if isinstance(sr, list) and len(sr) >= 2 and isinstance(sr[1], dict):
        actor_id = sr[1].get("actor", {}).get("id", "")
        if actor_id:
            return actor_id
        # Some activity types use 'user' instead of 'actor'
        user_id = sr[1].get("user", {}).get("id", "")
        if user_id:
            return user_id

    return raw.get("user", "")


def _classify_event(notification: dict, rules: list) -> str:
    """Match a notification against rules, return action. First match wins.

    Supports matching on app, object_type, and type fields.
    The Activity API uses different field values than the Notifications API:
      - Activity: object_type=files + type=shared (two fields)
      - Notifications: object_type=shared (one field)
    Use the 'type' field in rules for fine-grained Activity API filtering.
    """
    notif_app = notification.get("app", "")
    notif_object_type = notification.get("object_type", "")
    notif_type = notification.get("type", "")

    for rule in rules:
        rule_app = rule.get("app", "*")
        rule_object_type = rule.get("object_type")
        rule_type = rule.get("type")

        if rule_app != "*" and rule_app != notif_app:
            continue
        if rule_object_type and rule_object_type != notif_object_type:
            continue
        if rule_type and rule_type != notif_type:
            continue

        return rule.get("action", "silent")

    return "silent"


def _parse_notification(raw: dict, rules: list) -> ServiceEvent:
    """Parse a raw NC notification or activity dict into a ServiceEvent."""
    action = _classify_event(raw, rules)
    return ServiceEvent(
        service="nextcloud_notifications",
        notification_id=raw.get("notification_id", raw.get("activity_id", 0)),
        app=raw.get("app", ""),
        object_type=raw.get("object_type", raw.get("type", "")),
        object_id=str(raw.get("object_id", "")),
        subject=raw.get("subject", ""),
        message=raw.get("message", ""),
        link=raw.get("link", ""),
        sender=_extract_sender(raw),
        timestamp=raw.get("datetime", ""),
        action=action,
        raw=raw,
    )


class NotificationClient:
    """Lightweight HTTP client for NC Notifications + Activity API."""

    def __init__(self, base_url: str, username: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._http = httpx.AsyncClient(
            auth=(username, password),
            headers={
                "OCS-APIRequest": "true",
                "Accept": "application/json",
            },
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    async def fetch_notifications(self, etag: str = "") -> tuple:
        """Fetch notifications. Returns (list_of_notifs, new_etag)."""
        headers = {}
        if etag:
            headers["If-None-Match"] = etag

        url = f"{self._base_url}/ocs/v2.php/apps/notifications/api/v2/notifications"
        resp = await self._http.get(url, headers=headers)

        if resp.status_code == 304:
            return [], etag

        resp.raise_for_status()
        new_etag = resp.headers.get("ETag", etag)
        data = resp.json().get("ocs", {}).get("data", [])
        return data, new_etag

    async def fetch_activities(self, last_known_id: int = 0) -> tuple:
        """Fetch latest activities and return those newer than last_known_id.

        NC Activity API 'since' param paginates backwards (returns older items),
        so we always fetch the latest page and filter client-side.
        Returns (new_activities, highest_id).
        """
        url = f"{self._base_url}/ocs/v2.php/apps/activity/api/v2/activity"
        try:
            resp = await self._http.get(url, params={"limit": 50})
            if resp.status_code == 304:
                return [], last_known_id
            resp.raise_for_status()
            data = resp.json().get("ocs", {}).get("data", [])
            if not data:
                return [], last_known_id
            highest_id = max(a.get("activity_id", 0) for a in data)
            new_activities = [a for a in data if a.get("activity_id", 0) > last_known_id]
            return new_activities, highest_id
        except Exception as e:
            logger.warning("NC Activities: fetch failed: %s", e)
            return [], last_known_id

    async def delete_notification(self, notification_id: int):
        """Mark notification as read/processed."""
        url = f"{self._base_url}/ocs/v2.php/apps/notifications/api/v2/notifications/{notification_id}"
        await self._http.delete(url)

    async def close(self):
        await self._http.aclose()


class NextcloudNotificationService(BaseService):
    """Polls Nextcloud Notifications + Activity API, classifies events, routes to platform.

    This service handles event detection and routing only. File operations
    (download, upload, sync) are handled by a separate NextcloudFilesService.
    """

    name = "nextcloud_notifications"

    def __init__(self, config: dict, gateway_runner=None):
        super().__init__(config, gateway_runner)
        self._nc_url = config.get("nextcloud_url", "")
        self._username = config.get("username", "hermes")
        pw_env = config.get("app_password_env", "NEXTCLOUD_TALK_APP_PASSWORD")
        self._password = os.environ.get(pw_env, "").strip()
        self._poll_interval = config.get("poll_interval", 30)
        self._rules = config.get("rules", [])

        deliver = config.get("deliver", "")
        if ":" in deliver:
            self._deliver_platform, self._deliver_chat_id = deliver.split(":", 1)
        else:
            self._deliver_platform = deliver
            self._deliver_chat_id = ""

        store_path = config.get("store_path", "~/.hermes/notification_events.json")
        self._store = NotificationStore(path=store_path)

        self._client: Optional[NotificationClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._etag = ""
        self._last_seen_id = 0
        self._last_activity_id = 0

    def _on_poll_done(self, task: asyncio.Task):
        """Log if poll task dies unexpectedly."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.error("[NC Notifications] Poll task died: %s", exc, exc_info=exc)

    async def start(self) -> bool:
        if not self._nc_url or not self._password:
            logger.warning("NC Notifications: missing nextcloud_url or password")
            return False
        self._client = NotificationClient(self._nc_url, self._username, self._password)
        self._shutdown = False
        # Initialize activity ID to current latest to avoid processing old events
        if self._last_activity_id == 0:
            try:
                _, last_id = await asyncio.wait_for(
                    self._client.fetch_activities(0), timeout=10
                )
                self._last_activity_id = last_id
                logger.warning("[NC Notifications] Activity baseline: %d", last_id)
            except asyncio.TimeoutError:
                logger.warning("[NC Notifications] Activity baseline fetch timed out, starting from 0")
            except Exception as e:
                logger.warning("[NC Notifications] Activity baseline fetch failed: %s", e)
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._poll_task.add_done_callback(self._on_poll_done)
        logger.info("[NC Notifications] Started (poll_interval=%ds, stored_events=%d)",
                    self._poll_interval, self._store.count())
        return True

    async def stop(self):
        self._shutdown = True
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.close()
        logger.info("[NC Notifications] Stopped")

    async def _poll_loop(self):
        backoff = 5
        while not self._shutdown:
            try:
                # Poll Notifications API
                notifications, self._etag = await self._client.fetch_notifications(self._etag)
                backoff = 5

                for raw in notifications:
                    nid = raw.get("notification_id", 0)
                    if nid <= self._last_seen_id:
                        continue
                    self._last_seen_id = max(self._last_seen_id, nid)

                    event = _parse_notification(raw, self._rules)
                    await self.on_event(event)

                    try:
                        await self._client.delete_notification(nid)
                    except Exception as e:
                        logger.warning("NC Notifications: failed to delete %d: %s", nid, e)

                # Also poll Activity API for events that don't generate notifications
                activities, new_act_id = await self._client.fetch_activities(self._last_activity_id)
                if new_act_id > self._last_activity_id:
                    for act in activities:
                        aid = act.get("activity_id", 0)
                        if aid <= self._last_activity_id:
                            continue
                        event = _parse_notification(act, self._rules)
                        # Skip events caused by hermes itself to avoid loops
                        if event.sender == self._username:
                            continue
                        if event.action == "react":
                            await self.on_event(event)
                        else:
                            self._store.add(event)
                    self._last_activity_id = new_act_id

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("NC Notifications poll error: %s", e)
                backoff = min(backoff * 2, 60)

            await asyncio.sleep(self._poll_interval if backoff == 5 else backoff)

    async def on_event(self, event: ServiceEvent):
        if event.action == "ignore":
            return
        if event.action == "react":
            await self._deliver_react(event)
        else:
            self._store.add(event)

    async def _deliver_react(self, event: ServiceEvent):
        """Inject event as MessageEvent into target platform adapter."""
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent, MessageType

        runner = self.gateway_runner
        if not runner:
            logger.warning("NC Notifications: no gateway_runner, cannot deliver")
            return

        try:
            platform = Platform(self._deliver_platform)
        except ValueError:
            logger.error("NC Notifications: unknown platform '%s'", self._deliver_platform)
            return

        adapter = runner.adapters.get(platform)
        if not adapter:
            logger.warning("NC Notifications: adapter for %s not connected", platform.value)
            return

        # Build message text
        text = f"[NC Notification] {event.subject}"
        if event.message:
            text += f"\n{event.message}"
        if event.link:
            text += f"\nLink: {event.link}"

        # For calendar reminders: add tool guidance
        if event.app == "dav" and "calendar" in event.object_type.lower():
            text += "\n\n[System] This is a calendar reminder. Execute the task described above."
            text += " Use skill_view() to load relevant skills, then execute_code to run commands."
            text += " Do NOT use non-existent tools — check available tools first."

        # Use the event sender as user_id so the notification lands in their
        # existing chat session, preserving context for follow-up questions.
        user_id = event.sender or f"service:{self.name}"

        # Determine chat_type the same way the adapter does, so the session
        # key matches the user's existing chat session.
        chat_type = "dm"
        if hasattr(adapter, "_classify_chat"):
            chat_type = adapter._classify_chat(self._deliver_chat_id)

        source = adapter.build_source(
            chat_id=self._deliver_chat_id,
            user_id=user_id,
            user_name=f"NC/{event.app} ({event.sender})",
            chat_type=chat_type,
        )

        msg_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"ncnotif_{event.notification_id}",
        )

        await adapter.handle_message(msg_event)

        # Also store react events for history
        self._store.add(event)

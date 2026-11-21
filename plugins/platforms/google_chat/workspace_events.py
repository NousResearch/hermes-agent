"""Google Chat: Google Workspace Events subscriptions held by the impersonated Workspace user.

Google delivers a space message to a Chat app only when the app is @mentioned. A Workspace
Events subscription (``google.workspace.chat.message.v1.created``, resource included) created
by a Workspace user who is a member of the space sees every message there. Its notifications
go to the Pub/Sub topic the adapter already pulls from, so they arrive on the adapter's own
subscription; ``is_workspace_event`` tells them apart from Chat app events.

Subscriptions expire (Google caps their lifetime at hours, not days), so ``SpaceListener`` keeps
each one alive in the background: renew before expiry, reactivate when suspended, recreate
when deleted, adopt one that already exists. Lifecycle notifications wake it early.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
import re
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.platforms.google_chat")

MESSAGE_CREATED = "google.workspace.chat.message.v1.created"
_LIFECYCLE_PREFIX = "google.workspace.events.subscription."
_MESSAGE_EVENT_PREFIX = "google.workspace.chat.message.v1."


def is_workspace_event(attrs: dict[str, Any], envelope: Any) -> bool:
    """A Workspace Events delivery (CloudEvents attributes), not a Chat app event."""
    if not isinstance(envelope, dict):
        return False
    ce_type, ce_source = str(attrs.get("ce-type") or ""), str(attrs.get("ce-source") or "")
    return ce_source.startswith("//workspaceevents.googleapis.com/") or ce_type.startswith(
        (_LIFECYCLE_PREFIX, _MESSAGE_EVENT_PREFIX))


def is_lifecycle_event(attrs: dict[str, Any]) -> bool:
    """Expiration reminder, suspension, expiry: the subscription needs attention now."""
    return str(attrs.get("ce-type") or "").startswith(_LIFECYCLE_PREFIX)


def event_messages(attrs: dict[str, Any], envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """The Chat messages a created / batchCreated event carries (resource included)."""
    ce_type = str(attrs.get("ce-type") or "")
    if not (ce_type.startswith(_MESSAGE_EVENT_PREFIX) and "created" in ce_type.lower()):
        return []
    if isinstance(envelope.get("message"), dict):
        return [envelope["message"]]
    batch = envelope.get("messages") or []
    return [item["message"] for item in batch if isinstance(item, dict) and isinstance(item.get("message"), dict)]


def seconds_left(expire_time: Any) -> Optional[float]:
    """Seconds until an RFC 3339 ``expireTime``; None when absent or unparseable."""
    if not expire_time:
        return None
    text = re.sub(r"\.\d+", "", str(expire_time)).replace("Z", "+00:00")
    try:
        return (_dt.datetime.fromisoformat(text) - _dt.datetime.now(_dt.UTC)).total_seconds()
    except ValueError:
        return None


class SubscriptionKeeper:
    """Create / adopt / reactivate / renew one subscription per Chat space (blocking calls: run
    in a worker thread). ``subscriptions`` and ``http`` come from the delegated user."""

    RENEW_MARGIN = 3 * 3600.0
    _OP_POLLS = 10

    def __init__(self, events: Callable[[], Any], http: Callable[[], Any], topic: str) -> None:
        self._events, self._http, self.topic = events, http, topic
        self.names: dict[str, str] = {}  # space -> subscription name
        self.authority: Optional[str] = None  # users/{id} of the delegated user, learned from a subscription

    def _subs(self) -> Any:
        return self._events().subscriptions()

    def _op_result(self, op: Any) -> Optional[dict[str, Any]]:
        """The Subscription inside a long-running operation (normally already done)."""
        for _ in range(self._OP_POLLS):
            if not isinstance(op, dict):
                return None
            if op.get("error"):
                raise RuntimeError(f"Workspace Events operation failed: {op['error']}")
            if op.get("done"):
                resp = dict(op.get("response") or {})
                resp.pop("@type", None)
                return resp or None
            time.sleep(1.0)
            op = self._events().operations().get(name=op["name"]).execute(http=self._http())
        return None

    def _find(self, space: str) -> Optional[dict[str, Any]]:
        from . import adapter as gc
        if name := self.names.get(space):
            try:
                return self._subs().get(name=name).execute(http=self._http())
            except gc.HttpError as exc:
                if gc._http_status(exc) != 404:
                    raise
        flt = f'event_types:"{MESSAGE_CREATED}" AND target_resource="//chat.googleapis.com/{space}"'
        found = self._subs().list(filter=flt).execute(http=self._http()).get("subscriptions") or []
        return found[0] if found else None

    def _create(self, space: str) -> tuple[Optional[dict[str, Any]], str]:
        from . import adapter as gc
        body = {"targetResource": f"//chat.googleapis.com/{space}", "eventTypes": [MESSAGE_CREATED],
                "notificationEndpoint": {"pubsubTopic": self.topic}, "payloadOptions": {"includeResource": True}}
        try:
            return self._op_result(self._subs().create(body=body).execute(http=self._http())), "created"
        except gc.HttpError as exc:
            if gc._http_status(exc) != 409:  # ALREADY_EXISTS: one per user, target and event type
                raise
            return self._find(space), "adopted"

    def ensure(self, space: str) -> Optional[str]:
        """Make the subscription on ``space`` ACTIVE and not about to expire; the action taken."""
        sub, action = self._find(space), None
        if sub is not None and (sub.get("notificationEndpoint") or {}).get("pubsubTopic") != self.topic:
            self._subs().delete(name=sub["name"]).execute(http=self._http())  # the endpoint is immutable
            sub, action = None, "replaced (other topic)"
        if sub is not None and sub.get("state") == "SUSPENDED":
            reason = sub.get("suspensionReason")
            sub = self._op_result(self._subs().reactivate(name=sub["name"], body={}).execute(http=self._http())) or sub
            action = f"reactivated (was suspended: {reason})"
        if sub is None or sub.get("state") == "DELETED":
            sub, created = self._create(space)
            action = created if action is None else f"{action}, {created}"
        elif (left := seconds_left(sub.get("expireTime"))) is not None and left < self.RENEW_MARGIN:
            renewed = self._subs().patch(name=sub["name"], updateMask="ttl", body={"ttl": "0s"}).execute(http=self._http())
            sub, action = self._op_result(renewed) or sub, "renewed"
        if not sub or not sub.get("name"):
            raise RuntimeError(f"Workspace Events returned no subscription for {space}")
        self.names[space] = sub["name"]
        authority = str(sub.get("authority") or "")
        if authority.startswith("users/") and "@" not in authority:
            self.authority = authority
        if action:
            logger.info("[GoogleChat] Workspace Events subscription on %s %s; state=%s, expires %s",
                        space, action, sub.get("state"), sub.get("expireTime"))
        return action

    def drop(self, space: str) -> None:
        if name := self.names.pop(space, None):
            self._subs().delete(name=name).execute(http=self._http())


class SpaceListener:
    """Keeps the listen spaces' subscriptions alive and admits the messages they deliver."""

    CHECK_INTERVAL = 1800.0
    RETRY_INTERVAL = 300.0
    MEMBER_REFRESH = 600.0

    def __init__(self, user: Any, adapter: Any) -> None:
        self._user, self._adapter = user, adapter
        self.keeper: Optional[SubscriptionKeeper] = None
        self.members: dict[str, str] = {}  # users/{id} -> allowed email
        self.user_id: Optional[str] = None  # users/{id} of the Workspace user itself
        self._members_at = 0.0
        self._task: Optional[asyncio.Task] = None
        self._wake: Optional[asyncio.Event] = None

    # -- background upkeep -------------------------------------------------------
    def start(self) -> None:
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if (task := self._task) is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(task, timeout=5.0)
            self._task = None

    def wake(self) -> None:
        """Thread-safe: run an upkeep pass now (lifecycle event, refused send)."""
        loop, wake = self._adapter._loop, self._wake
        if wake is not None and self._adapter._loop_accepts_callbacks(loop):
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)

    def _topic(self) -> str:
        sub = self._adapter._subscriber.get_subscription(request={"subscription": self._adapter._subscription_path})
        return sub.topic

    def upkeep(self) -> None:
        """One pass (blocking): resolve members, ensure every listen space's subscription."""
        if self.keeper is None:
            self.keeper = SubscriptionKeeper(self._user.events, self._user.http, self._topic())
        if not self.members or time.monotonic() - self._members_at > self.MEMBER_REFRESH:
            self.refresh_members()
        for space in self._user.settings.listen_spaces:
            self.keeper.ensure(space)
        self.user_id = self.user_id or self.keeper.authority

    async def _run(self) -> None:
        last_err: Optional[str] = None
        while not self._adapter._shutting_down:
            try:
                await asyncio.to_thread(self.upkeep)
                if last_err is not None:
                    logger.info("[GoogleChat] Workspace Events subscriptions are ready")
                last_err, delay = None, self.CHECK_INTERVAL
            except Exception as exc:  # retried below; logged with its traceback
                msg = self._adapter_redact(str(exc))[:300]
                if msg != last_err:
                    logger.warning("[GoogleChat] Workspace Events subscription not ready (retrying every %ds): %s",
                                   int(self.RETRY_INTERVAL), msg, exc_info=True)
                last_err, delay = msg, self.RETRY_INTERVAL
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=delay)

    @staticmethod
    def _adapter_redact(text: str) -> str:
        from . import adapter as gc
        return gc._redact_sensitive(text)

    def refresh_members(self) -> None:
        """Map ``users/{id}`` -> email for the allowed users (Workspace Events payloads carry no
        email), and learn the Workspace user's own id. App-authenticated members.get by email."""
        from . import adapter as gc
        self._members_at = time.monotonic()
        settings = self._user.settings
        for space in settings.listen_spaces:
            for email in sorted(settings.allowed_users | {settings.email}):
                try:
                    found = self._adapter._chat_api.spaces().members().get(name=f"{space}/members/{email}").execute(
                        http=self._adapter._new_authed_http())
                except gc.HttpError as exc:
                    logger.debug("[GoogleChat] members.get %s in %s: %s", email, space, gc._redact_sensitive(str(exc)))
                    continue
                member_id = str(((found or {}).get("member") or {}).get("name") or "")
                if not member_id:
                    continue
                if email == settings.email:
                    self.user_id = member_id
                else:
                    self.members[member_id] = email

    # -- inbound (Pub/Sub thread) ------------------------------------------------
    def _sender_email(self, sender_id: str) -> Optional[str]:
        email = self.members.get(sender_id)
        if email is None and sender_id and time.monotonic() - self._members_at > self.MEMBER_REFRESH:
            self.refresh_members()
            email = self.members.get(sender_id)
        return email

    def admit(self, m: dict[str, Any]) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        """``(msg, envelope)`` for ``_dispatch_message`` when a listened message addresses the
        agent, else None. Skips: other spaces, bots, the Workspace user itself, slash commands and
        @mentions of a bot (the app's own delivery handles those), text that does not match
        ``listen_pattern``, senders outside GOOGLE_CHAT_ALLOWED_USERS, and duplicates."""
        name = str(m.get("name") or "")
        space_name = (m.get("space") or {}).get("name") or "/".join(name.split("/")[:2])
        settings = self._user.settings
        sender = m.get("sender") or {}
        sender_id = str(sender.get("name") or "")
        if not name or space_name not in settings.listen_spaces or sender.get("type") == "BOT":
            return None
        if (sender_id and sender_id == self.user_id) or m.get("slashCommand") or _mentions_a_bot(m):
            return None
        text = str(m.get("text") or "")
        if not settings.listen_pattern.search(text):
            return None
        email = self._sender_email(sender_id)
        if email is None or email not in settings.allowed_users:
            logger.info("[GoogleChat] listen_spaces: %s from %s is not from an allowed user; ignored", name, sender_id)
            return None
        if self._adapter._dedup.is_duplicate(name):
            return None
        space = {"spaceType": "SPACE", **(m.get("space") or {}), "name": space_name}
        msg = {**m, "space": space, "argumentText": text,
               "sender": {**sender, "email": email, "displayName": sender.get("displayName") or email}}
        logger.info("[GoogleChat] listen_spaces: admitted %s from %s (%d chars)", name, email, len(text))
        return msg, {"space": space, "message": msg}


def _mentions_a_bot(m: dict[str, Any]) -> bool:
    for ann in m.get("annotations") or []:
        user = ((ann or {}).get("userMention") or {}).get("user") or {}
        if (ann or {}).get("type") == "USER_MENTION" and user.get("type") == "BOT":
            return True
    return False

"""Google Chat: Google Workspace Events subscriptions held by the impersonated Workspace user.

Google delivers a space message to a Chat app only when the app is @mentioned. A Workspace
Events subscription (``google.workspace.chat.message.v1.created``, resource included) created
by a Workspace user who is a member of the space sees every message there. Its notifications
go to the Pub/Sub topic the adapter already pulls from, so they arrive on the adapter's own
subscription; ``is_workspace_event`` tells them apart from Chat app events.

Subscriptions expire (Google caps their lifetime at hours, not days), so ``SpaceListener`` keeps
each one alive in the background: renew before expiry, reactivate when suspended, recreate
when deleted, adopt one that already exists. Lifecycle notifications wake it early.

A subscription that expired anyway (the host was off or asleep for longer than its lifetime)
delivered nothing from its expiry on, and Google sends nothing for that time afterwards. The
keeper remembers each subscription's expiry across restarts, so it knows the gap; ``catch_up``
then reads what was said in the space meanwhile and hands it to the agent late, as one turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import logging
import os
import re
import time
from pathlib import Path
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


def _parse(stamp: Any) -> Optional[_dt.datetime]:
    """An RFC 3339 time (fractions dropped); None when absent or unparseable."""
    if not stamp:
        return None
    text = re.sub(r"\.\d+", "", str(stamp)).replace("Z", "+00:00")
    try:
        return _dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _stamp(when: Optional[_dt.datetime] = None) -> str:
    """RFC 3339 in UTC without fractions, the form ``spaces.messages.list`` filters take (default: now)."""
    return (when or _dt.datetime.now(_dt.UTC)).astimezone(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def seconds_left(expire_time: Any) -> Optional[float]:
    """Seconds until an RFC 3339 ``expireTime``; None when absent or unparseable."""
    when = _parse(expire_time)
    return None if when is None else (when - _dt.datetime.now(_dt.UTC)).total_seconds()


def _lapsed(expire_time: Any) -> Optional[str]:
    """``expire_time`` when it is in the past: from then on its subscription delivered nothing."""
    left = seconds_left(expire_time)
    return str(expire_time) if left is not None and left < 0 else None


def state_path(kind: str) -> Path:
    """Where a keeper remembers its subscriptions (``kind``: one file per keeper), under the
    profile's HERMES_HOME; resolved at call time so multiplexed profiles do not share it."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / f"google_chat_workspace_{kind}.json"


class SubscriptionKeeper:
    """Create / adopt / reactivate / renew one subscription per Chat space (blocking calls: run
    in a worker thread). ``subscriptions`` and ``http`` come from the delegated user.

    With ``state_path`` each subscription's name and expiry outlive a restart. That is what lets
    the keeper tell that one expired while the process was down, and since when: ``missed``
    then names the window nothing was delivered in, until ``caught_up`` settles it."""

    RENEW_MARGIN = 3 * 3600.0
    _OP_POLLS = 10

    def __init__(self, events: Callable[[], Any], http: Callable[[], Any], topic: str,
                 state_path: Optional[Path] = None) -> None:
        self._events, self._http, self.topic = events, http, topic
        self._state_path = state_path
        self.names: dict[str, str] = {}  # space -> subscription name
        self._expires: dict[str, str] = {}  # space -> expireTime as last seen
        self._missed: dict[str, dict[str, str]] = {}  # space -> {"since", "until"}: a catch-up still owed
        self.authority: Optional[str] = None  # users/{id} of the delegated user, learned from a subscription
        self._load()

    # -- what outlives a restart ---------------------------------------------------
    def _load(self) -> None:
        if self._state_path is None:
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return  # first run, or unreadable: nothing is known, so nothing is caught up
        subs = data.get("subscriptions") if isinstance(data, dict) else None
        for space, entry in (subs.items() if isinstance(subs, dict) else ()):
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                self.names[space] = entry["name"]
                if isinstance(entry.get("expireTime"), str):
                    self._expires[space] = entry["expireTime"]
        missed = data.get("missed") if isinstance(data, dict) else None
        for space, entry in (missed.items() if isinstance(missed, dict) else ()):
            if isinstance(entry, dict) and isinstance(entry.get("since"), str) and isinstance(entry.get("until"), str):
                self._missed[space] = {"since": entry["since"], "until": entry["until"]}

    def _save(self) -> None:
        if self._state_path is None:
            return
        data = {"subscriptions": {space: {"name": name, "expireTime": self._expires.get(space)}
                                  for space, name in self.names.items()},
                "missed": self._missed}
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError:
            logger.debug("[GoogleChat] could not persist the Workspace Events subscriptions", exc_info=True)

    def missed(self, space: str) -> Optional[tuple[str, str]]:
        """``(since, until)`` while a catch-up is owed on ``space``: its subscription delivered
        nothing between the two."""
        window = self._missed.get(space)
        return (window["since"], window["until"]) if window else None

    def caught_up(self, space: str) -> None:
        if self._missed.pop(space, None) is not None:
            self._save()

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
                # Google answers 403 ("Permission denied on subscription ...") for a subscription
                # that expired and was removed, not 404. Either way this name is gone: forget it
                # and look the space up. A real refusal fails at the list below with its own error.
                if gc._http_status(exc) not in (403, 404):
                    raise
                self.names.pop(space, None)
        flt =f'event_types:"{MESSAGE_CREATED}" AND target_resource="//chat.googleapis.com/{space}"'
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
        held = self.names.get(space)
        sub, action = self._find(space), None
        if sub is not None:
            lapsed = _lapsed(sub.get("expireTime"))  # still returned, but past its expiry
        else:  # the one held is gone: it delivered nothing since the expiry last seen
            lapsed = _lapsed(self._expires.get(space)) if held and space not in self.names else None
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
        if lapsed and space not in self._missed:
            self._missed[space] = {"since": lapsed, "until": _stamp()}
        expires = str(sub.get("expireTime") or "")
        if lapsed or self.names.get(space) != sub["name"] or self._expires.get(space) != expires:
            self.names[space], self._expires[space] = sub["name"], expires
            self._save()
        authority = str(sub.get("authority") or "")
        if authority.startswith("users/") and "@" not in authority:
            self.authority = authority
        if action:
            logger.info("[GoogleChat] Workspace Events subscription on %s %s; state=%s, expires %s",
                        space, action, sub.get("state"), sub.get("expireTime"))
        return action

    def drop(self, space: str) -> None:
        self._expires.pop(space, None)
        self._missed.pop(space, None)
        if name := self.names.pop(space, None):
            self._save()
            self._subs().delete(name=name).execute(http=self._http())


# -- what was said while a subscription was gone ----------------------------------

CATCH_UP_MAX = 20  # messages handed over per space (the newest)
CATCH_UP_MAX_AGE = 48 * 3600.0  # nothing older than this is read


def _local(stamp: Any) -> str:
    """``stamp`` in the host's time zone, for the note the model reads."""
    when = _parse(stamp)
    return str(stamp or "?") if when is None else when.astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()


def _list_messages(chat: Any, http: Callable[[], Any], space: str, since: str, until: str) -> list[dict[str, Any]]:
    """The messages created in ``space`` between the two times, oldest first (as the user)."""
    found: list[dict[str, Any]] = []
    token: Optional[str] = None
    for _ in range(10):
        kwargs: dict[str, Any] = {"parent": space, "pageSize": 100, "orderBy": "createTime ASC",
                                  "filter": f'createTime > "{since}" AND createTime < "{until}"'}
        if token:
            kwargs["pageToken"] = token
        resp = chat.spaces().messages().list(**kwargs).execute(http=http())
        found.extend(m for m in resp.get("messages") or [] if isinstance(m, dict))
        token = resp.get("nextPageToken")
        if not token:
            break
    return found


def late_turn(admitted: list[tuple[dict[str, Any], dict[str, Any]]], since: str, until: str,
              left_out: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """One ``(msg, envelope)`` for several admitted messages that arrive late. The newest carries
    the turn (the reply and the thinking reaction land on it); its text says the listener was
    down and from when to when, then gives each message with the time it was sent and its
    sender. One turn, because separate late messages would interrupt each other."""
    last, envelope = admitted[-1]
    lines = []
    for msg, _ in admitted:
        sender = msg.get("sender") or {}
        files = len(msg.get("attachment") or []) if msg is not last else 0
        lines.append(f"[{_local(msg.get('createTime'))}] {sender.get('displayName') or sender.get('email') or '?'}: "
                     f"{str(msg.get('text') or '').strip() or '(no text)'}"
                     + (f" (with {files} attachment(s), not handed over)" if files else ""))
    note = (f"[Delivered late. This conversation's listener was down from {_local(since)} to {_local(until)}, so "
            f"{len(admitted)} message(s) sent in that time never arrived"
            + (f" (the newest {len(admitted)} are below; {left_out} older ones are left out)" if left_out else "")
            + ". They are handed over now, each with the time it was sent. Before acting on a request, consider "
            "whether it still applies.]")
    text = "\n".join([note, *lines])
    late = {**last, "text": text, "argumentText": text}
    return late, {**envelope, "message": late}


def catch_up(part: Any, space: str) -> int:
    """Hand the agent what was said in ``space`` while its subscription was gone (blocking; run
    in a worker thread). ``part`` is the listener that owns the space: its ``keeper`` knows the
    window, its ``admit`` decides which messages count, exactly as for a live delivery. Returns
    how many messages were handed over. A failed read stays owed and is retried at the next pass."""
    from . import adapter as gc
    window = part.keeper.missed(space) if part.keeper is not None else None
    if window is None:
        return 0
    since, until = window
    oldest = _dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=CATCH_UP_MAX_AGE)
    start = _stamp(max(_parse(since) or oldest, oldest))
    try:
        messages = _list_messages(part._user.chat(), part._user.http, space, start, _stamp(_parse(until)))
    except Exception as exc:  # retried at the next upkeep pass; logged with its traceback
        msg = gc._redact_sensitive(str(exc))[:300]
        if part._catch_up_errors.get(space) != msg:
            part._catch_up_errors[space] = msg
            logger.warning("[GoogleChat] reading what was said in %s between %s and %s failed (retried at the next "
                           "pass): %s", space, start, until, msg, exc_info=True)
        return 0
    part._catch_up_errors.pop(space, None)
    admitted = []
    for m in messages:
        sender = m.get("sender") or {}
        if sender.get("type") == "BOT" or (part.user_id and sender.get("name") == part.user_id):
            continue  # the app's and the user's own messages, without a log line each
        if (taken := part.admit(m)) is not None:
            admitted.append(taken)
    part.keeper.caught_up(space)
    if not admitted:
        logger.info("[GoogleChat] nothing to answer in %s from between %s and %s (%d message(s) read)",
                    space, start, until, len(messages))
        return 0
    left_out = max(0, len(admitted) - CATCH_UP_MAX)
    admitted = admitted[-CATCH_UP_MAX:]
    logger.info("[GoogleChat] %d message(s) sent in %s between %s and %s are handed over late (%d older left out)",
                len(admitted), space, start, until, left_out)
    part._adapter._submit_on_loop(part._adapter._dispatch_message(*late_turn(admitted, start, until, left_out)))
    return len(admitted)


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
        self._catch_up_errors: dict[str, str] = {}
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
        """One pass (blocking): resolve members, ensure every listen space's subscription, and
        hand over what was said while one of them was gone."""
        if self.keeper is None:
            self.keeper = SubscriptionKeeper(self._user.events, self._user.http, self._topic(), state_path("spaces"))
        if not self.members or time.monotonic() - self._members_at > self.MEMBER_REFRESH:
            self.refresh_members()
        for space in self._user.settings.listen_spaces:
            self.keeper.ensure(space)
            self.user_id = self.user_id or self.keeper.authority
            catch_up(self, space)

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

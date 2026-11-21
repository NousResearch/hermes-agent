"""Google Chat: an optional second identity, a Workspace *user* the app's service account
impersonates through domain-wide delegation (DWD).

A Chat app cannot do two things with app authentication: react to a message
(``spaces.messages.reactions`` accepts user authentication only), and see a space message it
was not @mentioned in (Google delivers only those to an app). A real Workspace user can. With
``platforms.google_chat.workspace_user.email`` set, the adapter impersonates that one user for:

* ``thinking_reaction``: an emoji on the asker's message while the agent works, in place of
  the posted marker message, wherever the user is a member;
* ``listen_spaces`` + ``listen_pattern``: a Google Workspace Events subscription per listed
  space, so a message that matches the pattern reaches the agent without an @mention, from
  ``GOOGLE_CHAT_ALLOWED_USERS`` senders only (``workspace_events.py``);
* ``answer_dms``: person-to-person DMs that allowed people open with the user are answered by
  the user, as the user (``workspace_user_dms.py``).

The app still posts every reply. Nothing here runs unless configured, and the delegated token
carries only the scopes the enabled features need (``WorkspaceUserSettings.scopes``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("gateway.platforms.google_chat")

REACTIONS_SCOPE = "https://www.googleapis.com/auth/chat.messages.reactions"
MESSAGES_READONLY_SCOPE = "https://www.googleapis.com/auth/chat.messages.readonly"
# answer_dms: post, edit and delete as the user (also covers reading and reactions), find its DMs.
MESSAGES_SCOPE = "https://www.googleapis.com/auth/chat.messages"
SPACES_READONLY_SCOPE = "https://www.googleapis.com/auth/chat.spaces.readonly"
MEMBERSHIPS_READONLY_SCOPE = "https://www.googleapis.com/auth/chat.memberships.readonly"
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_SPACE_NAME_RE = re.compile(r"^spaces/[A-Za-z0-9_-]+$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _as_list(value: Any) -> list[str]:
    """A YAML list or a comma-separated string, stripped, blanks dropped."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def parse_allowed_users(csv: str) -> frozenset[str]:
    return frozenset(e.strip().lower() for e in (csv or "").split(",") if e.strip())


def _listen_settings(raw: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """``listen_spaces`` / ``listen_pattern`` when both are valid and an allowlist exists, else {}."""
    spaces = _as_list(raw.get("listen_spaces"))
    if not spaces:
        return {}
    bad = [s for s in spaces if not _SPACE_NAME_RE.match(s)]
    if bad:
        logger.warning("[GoogleChat] workspace_user.listen_spaces: ignoring %s (expected spaces/<id>)", ", ".join(bad))
        spaces = [s for s in spaces if s not in bad]
    pattern_text = str(raw.get("listen_pattern") or "")
    if not spaces or not pattern_text:
        logger.warning("[GoogleChat] workspace_user.listen_spaces needs listen_pattern, a regular expression the "
                       "message text must match (for example (?i)\\bhermes\\b; ^ admits every message); not listening")
        return {}
    try:
        pattern = re.compile(pattern_text)
    except re.error as exc:
        logger.warning("[GoogleChat] workspace_user.listen_pattern is not a valid regular expression (%s); "
                       "not listening", exc)
        return {}
    if not allowed:
        logger.warning("[GoogleChat] workspace_user.listen_spaces needs GOOGLE_CHAT_ALLOWED_USERS: messages the app "
                       "was not @mentioned in are admitted only from listed users (GOOGLE_CHAT_ALLOW_ALL_USERS is not "
                       "honoured here); not listening")
        return {}
    return {"listen_spaces": tuple(dict.fromkeys(spaces)), "listen_pattern": pattern}


def _answer_dms(raw: dict[str, Any], allowed: frozenset[str]) -> bool:
    value = raw.get("answer_dms")
    if not (value is True or str(value or "").strip().lower() in _TRUE_WORDS):
        return False
    if not allowed:
        logger.warning("[GoogleChat] workspace_user.answer_dms needs GOOGLE_CHAT_ALLOWED_USERS: only DMs with "
                       "listed users are answered (GOOGLE_CHAT_ALLOW_ALL_USERS is not honoured here); not answering")
        return False
    return True


@dataclass(frozen=True)
class WorkspaceUserSettings:
    """``platforms.google_chat.workspace_user`` from config.yaml, validated."""

    email: str
    allowed_users: frozenset[str]
    thinking_reaction: str = ""
    listen_spaces: tuple[str, ...] = ()
    listen_pattern: Optional[re.Pattern[str]] = None
    answer_dms: bool = False

    @property
    def listening(self) -> bool:
        return bool(self.listen_spaces) and self.listen_pattern is not None

    def scopes(self) -> list[str]:
        """The DWD scopes the enabled features need; the Workspace admin grants exactly these."""
        if self.answer_dms:  # chat.messages covers the reading and reacting of the other features
            return [MESSAGES_SCOPE, SPACES_READONLY_SCOPE, MEMBERSHIPS_READONLY_SCOPE]
        return [scope for scope, needed in ((REACTIONS_SCOPE, bool(self.thinking_reaction)),
                                            (MESSAGES_READONLY_SCOPE, self.listening)) if needed]

    @classmethod
    def from_extra(cls, extra: dict[str, Any], allowed_users_csv: str) -> Optional[WorkspaceUserSettings]:
        """Settings, or None when the feature is not configured (or configured unusably: warned)."""
        raw = extra.get("workspace_user")
        if not raw:
            return None
        if not isinstance(raw, dict):
            logger.warning("[GoogleChat] platforms.google_chat.workspace_user must be a mapping; ignored")
            return None
        email = str(raw.get("email") or "").strip().lower()
        if not _EMAIL_RE.match(email):
            logger.warning("[GoogleChat] workspace_user.email is missing or not an email address; the Workspace "
                           "user stays off")
            return None
        allowed = parse_allowed_users(allowed_users_csv)
        settings = cls(email=email, allowed_users=allowed,
                       thinking_reaction=str(raw.get("thinking_reaction") or "").strip(),
                       answer_dms=_answer_dms(raw, allowed), **_listen_settings(raw, allowed))
        if not settings.scopes():
            logger.warning("[GoogleChat] workspace_user.email is set but no feature is enabled "
                           "(thinking_reaction, listen_spaces, answer_dms); the Workspace user stays off")
            return None
        return settings


class WorkspaceUserUnavailable(RuntimeError):
    """The adapter's credentials cannot impersonate a user."""


class WorkspaceUser:
    """Delegated clients for the configured user and the features that use them.

    Thread-safety: the clients are built once under a lock; every request takes a fresh
    ``http()`` (httplib2 shares SSL state across threads, same rule as the app client)."""

    _REACT_BACKOFF_REFUSED = 3600.0  # 403/404: the user is not a member of that space (an app DM)
    _REACT_BACKOFF_ERROR = 300.0
    _REACT_WAIT = 5.0

    def __init__(self, adapter: Any, settings: WorkspaceUserSettings) -> None:
        self._adapter, self.settings = adapter, settings
        self._lock = threading.Lock()
        self._creds: Any = None
        self._clients: dict[str, Any] = {}
        self._react_skip_until: dict[str, float] = {}
        self._reaction_tasks: dict[str, asyncio.Task] = {}  # message name -> reaction create
        self._reactions: dict[str, str] = {}  # message name -> reaction name
        self._thinking_chat: dict[str, str] = {}  # chat_id -> message whose reaction replaces the marker
        self.listener: Any = None
        self.dms: Any = None  # workspace_user_dms.DirectMessages when answer_dms is on
        self.available = True  # False once start() found credentials that cannot impersonate

    # -- delegated credentials and clients -----------------------------------
    def credentials(self) -> Any:
        with self._lock:
            if self._creds is None:
                base = self._adapter._credentials
                if not hasattr(base, "with_subject"):
                    raise WorkspaceUserUnavailable(
                        "workspace_user needs a service-account key (GOOGLE_CHAT_SERVICE_ACCOUNT_JSON): Application "
                        "Default and workload-identity credentials cannot impersonate a user")
                self._creds = base.with_scopes(self.settings.scopes()).with_subject(self.settings.email)
            return self._creds

    def http(self) -> Any:
        from . import adapter as gc
        return gc.AuthorizedHttp(self.credentials(), http=gc.httplib2.Http(timeout=30))

    def _client(self, api: str, version: str) -> Any:
        from . import adapter as gc
        creds = self.credentials()
        with self._lock:
            if api not in self._clients:
                self._clients[api] = gc.build_service(api, version, credentials=creds, cache_discovery=False)
            return self._clients[api]

    def chat(self) -> Any:
        return self._client("chat", "v1")

    def events(self) -> Any:
        return self._client("workspaceevents", "v1")

    # -- lifecycle -------------------------------------------------------------
    async def start(self, *, pubsub: bool) -> None:
        """Called from ``connect()`` on the adapter loop, under its profile scope, before the
        Pub/Sub stream starts (the DMs are known before their first message can arrive)."""
        try:
            self.credentials()
        except WorkspaceUserUnavailable as exc:
            logger.warning("[GoogleChat] %s; the Workspace user stays off", exc)
            self.available = False
            return
        logger.info("[GoogleChat] Workspace user %s via domain-wide delegation: reaction=%s, listening in %d space(s), "
                    "answering DMs=%s", self.settings.email, self.settings.thinking_reaction or "off",
                    len(self.settings.listen_spaces), "on" if self.settings.answer_dms else "off")
        if not (self.settings.listening or self.settings.answer_dms):
            return
        if not pubsub:
            logger.warning("[GoogleChat] workspace_user.listen_spaces and answer_dms need Pub/Sub inbound (Workspace "
                           "Events are delivered to the app's Pub/Sub topic); off in HTTP callback mode")
            return
        if self.settings.listening:
            from .workspace_events import SpaceListener
            self.listener = SpaceListener(self, self._adapter)
            self.listener.start()
        if self.settings.answer_dms:
            from .workspace_user_dms import DirectMessages
            self.dms = DirectMessages(self, self._adapter)
            await self.dms.start()

    async def stop(self) -> None:
        for part in (self.listener, self.dms):
            if part is not None:
                await part.stop()
        tasks, self._reaction_tasks = list(self._reaction_tasks.values()), {}
        for task in tasks:
            task.cancel()

    def handle_event(self, attrs: dict[str, Any], envelope: Any) -> bool:
        """Pub/Sub thread: True when ``envelope`` was a Workspace Events delivery (consumed)."""
        from .workspace_events import event_messages, is_lifecycle_event, is_workspace_event
        if not is_workspace_event(attrs, envelope):
            return False
        if is_lifecycle_event(attrs):
            logger.info("[GoogleChat] Workspace Events lifecycle event: %s", str(attrs.get("ce-type")).rsplit(".", 1)[-1])
            for part in (self.listener, self.dms):
                if part is not None:
                    part.wake()
            return True
        for message in event_messages(attrs, envelope):
            part = self.dms if self.owns(message.get("name")) else self.listener
            admitted = part.admit(message) if part is not None else None
            if admitted:
                self._adapter._submit_on_loop(self._adapter._dispatch_message(*admitted))
        return True

    def owns(self, name: Any) -> bool:
        """True when ``name`` (space, message, attachment) is in a DM the user answers as itself."""
        return self.dms is not None and self.dms.owns(name)

    # -- thinking reaction -----------------------------------------------------
    def begin_thinking(self, chat_id: str, message_id: str) -> None:
        """Start the reaction on ``message_id``. Synchronous up to the task creation, so a
        ``send_typing`` that runs next finds the reservation and does not post the marker."""
        if not (self.available and self.settings.thinking_reaction and chat_id and message_id):
            return
        if message_id in self._reaction_tasks:
            return
        if time.monotonic() < self._react_skip_until.get(chat_id, 0.0):
            return
        self._thinking_chat[chat_id] = message_id
        self._reaction_tasks[message_id] = asyncio.create_task(self._create_reaction(chat_id, message_id))

    async def _create_reaction(self, chat_id: str, message_id: str) -> None:
        emoji = self.settings.thinking_reaction
        try:
            resp = await asyncio.to_thread(lambda: self.chat().spaces().messages().reactions().create(
                parent=message_id, body={"emoji": {"unicode": emoji}}).execute(http=self.http()))
        except Exception as exc:  # health: allow BLE001 -- any failure falls back to the marker message
            self._reaction_failed(chat_id, message_id, exc)
            return
        name = resp.get("name") if isinstance(resp, dict) else None
        if not name:
            self._reaction_failed(chat_id, message_id, RuntimeError("reactions.create returned no name"))
        elif message_id in self._reaction_tasks:
            self._reactions[message_id] = name
        else:  # the turn ended while the create was in flight
            await self._delete_reaction(name)

    def _reaction_failed(self, chat_id: str, message_id: str, exc: BaseException) -> None:
        from . import adapter as gc
        refused = gc._http_status(exc) in (403, 404)
        backoff = self._REACT_BACKOFF_REFUSED if refused else self._REACT_BACKOFF_ERROR
        self._react_skip_until[chat_id] = time.monotonic() + backoff
        if self._thinking_chat.get(chat_id) == message_id:
            self._thinking_chat.pop(chat_id, None)
        logger.info("[GoogleChat] thinking reaction as %s refused in %s (%s); using the marker message there for %ds%s",
                    self.settings.email, chat_id, gc._redact_sensitive(str(exc))[:200], int(backoff),
                    " (is the user a member of that space?)" if refused else "")
        logger.debug("[GoogleChat] thinking reaction failure detail", exc_info=exc)

    async def wait_thinking(self, chat_id: str) -> bool:
        """True when a reaction stands in for the marker in ``chat_id``; waits (bounded) for one
        still being created. False once it failed, so the caller posts the marker after all."""
        message_id = self._thinking_chat.get(chat_id)
        if not message_id:
            return False
        task = self._reaction_tasks.get(message_id)
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._REACT_WAIT)
            except TimeoutError:
                return True  # still in flight: check again on the next typing tick
        return self._thinking_chat.get(chat_id) == message_id

    async def end_thinking(self, chat_id: str, message_id: str) -> None:
        """Remove the reaction (a create still in flight removes its own on completion)."""
        if self._thinking_chat.get(chat_id) == message_id:
            self._thinking_chat.pop(chat_id, None)
        if self._reaction_tasks.pop(message_id, None) is None:
            return
        name = self._reactions.pop(message_id, None)
        if name:
            await self._delete_reaction(name)

    async def _delete_reaction(self, name: str) -> None:
        try:
            await asyncio.to_thread(lambda: self.chat().spaces().messages().reactions().delete(
                name=name).execute(http=self.http()))
        except Exception:  # a left-over reaction is cosmetic; the turn is over
            logger.debug("[GoogleChat] removing thinking reaction %s failed", name, exc_info=True)

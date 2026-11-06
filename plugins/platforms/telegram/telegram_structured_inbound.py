"""Inbound Telegram messages that carry structured data instead of text, rendered for the agent.

Location/venue pins and checklists (Bot API 9.1 ``Message.checklist``) have no ``text``. A checklist
goes through ``_handle_text_message`` as a text turn: ``_build_message_event`` renders it, so the
authorization, group gates, observe mode and text batching are the ones text gets, and an @mention
in its title or tasks counts like one in text (#78718). Its task updates (``checklist_tasks_done`` /
``checklist_tasks_added`` service messages, PTB ``ChecklistTasksDone`` / ``ChecklistTasksAdded``) are
observed, never an agent turn. python-telegram-bot 22.8 (the pin) parses all three fields and ships
their filters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gateway.platforms.event import MessageType

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_CHECKLIST_HEADER = "[Checklist]"
_UPDATE_HEADER = "[Checklist update]"


def with_checklists(text_filter: Any, filters: Any) -> Any:
    """``text_filter`` widened to new checklist messages. Edits stay out: an edited checklist is not
    a new request (task changes arrive as their own service messages). A PTB without checklist
    support keeps ``text_filter`` as is, like the FORUM_TOPIC_CREATED registration."""
    checklist = getattr(filters, "CHECKLIST", None)
    edited = getattr(getattr(filters, "UpdateType", None), "EDITED", None)
    if checklist is None or edited is None:
        return text_filter
    return text_filter | (checklist & ~edited)


def checklist_update_filter(filters: Any) -> Any:
    """Filter for the two checklist service messages, or None when PTB lacks them."""
    status = getattr(filters, "StatusUpdate", None)
    done = getattr(status, "CHECKLIST_TASKS_DONE", None)
    added = getattr(status, "CHECKLIST_TASKS_ADDED", None)
    return None if done is None or added is None else done | added


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _sequence(value: Any) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _checklist_of(message: Any) -> Any:
    """``message.checklist`` when it has the Bot API shape (title string, task sequence), else None.

    Every inbound message reaches ``_build_message_event``; the shape check keeps anything else
    (test doubles included) from rendering as a checklist."""
    checklist = getattr(message, "checklist", None)
    if isinstance(getattr(checklist, "title", None), str) and isinstance(getattr(checklist, "tasks", None), (list, tuple)):
        return checklist
    return None


def _is_done(task: Any) -> bool:
    """Bot API ``completion_date`` is 0 for an open task; PTB turns that into the epoch."""
    when = getattr(task, "completion_date", None)
    if hasattr(when, "timestamp") and when.timestamp() > 0:
        return True
    return getattr(task, "completed_by_user", None) is not None or getattr(task, "completed_by_chat", None) is not None


def _task_line(task: Any) -> str:
    done = _is_done(task)
    line = f"- [{'x' if done else ' '}] {_one_line(getattr(task, 'text', ''))}".rstrip()
    by = getattr(task, "completed_by_user", None) or getattr(task, "completed_by_chat", None)
    name = _one_line(getattr(by, "full_name", None) or getattr(by, "title", None) or getattr(by, "username", None))
    return f"{line} (done by {name})" if done and name else line


def _render(checklist: Any, tasks: Any) -> str:
    return "\n".join([f"{_CHECKLIST_HEADER} {_one_line(checklist.title)}".rstrip(), *map(_task_line, tasks)])


def checklist_text(message: Any) -> str:
    """A checklist message as the agent reads it: ``[Checklist] <title>``, then one ``- [ ]`` or
    ``- [x]`` line per task (who completed it, when known); "" for any other message."""
    checklist = _checklist_of(message)
    return _render(checklist, checklist.tasks) if checklist is not None else ""


def checklist_entity_sources(message: Any):
    """``(text, entities)`` for a checklist's title and each task, so an @mention or wake word there
    addresses the bot as it does in a text message; nothing for any other message."""
    checklist = _checklist_of(message)
    if checklist is None:
        return
    yield checklist.title, _sequence(getattr(checklist, "title_entities", None))
    for task in checklist.tasks:
        yield str(getattr(task, "text", None) or ""), _sequence(getattr(task, "text_entities", None))


def checklist_reply_text(message: Any) -> str:
    """Reply context for a reply to a checklist: only the task the reply targets
    (``reply_to_checklist_task_id``), else the whole checklist; "" when it replies to anything else."""
    checklist = _checklist_of(getattr(message, "reply_to_message", None))
    if checklist is None:
        return ""
    task_id = getattr(message, "reply_to_checklist_task_id", None)
    targeted = [task for task in checklist.tasks if task_id is not None and getattr(task, "id", None) == task_id]
    return _render(checklist, targeted or checklist.tasks)


def _task_refs(ids: Any, names: dict) -> str:
    return ", ".join(f'"{names[i]}"' if names.get(i) else f"task #{i}" for i in _sequence(ids))


def _change_lines(done: Any, added: Any, names: dict) -> list[str]:
    if done is not None:
        rows = (("Marked done", getattr(done, "marked_as_done_task_ids", None)),
                ("Marked not done", getattr(done, "marked_as_not_done_task_ids", None)))
        return [f"{label}: {refs}" for label, ids in rows if (refs := _task_refs(ids, names))]
    texts = [_one_line(getattr(task, "text", "")) for task in _sequence(getattr(added, "tasks", None))]
    return ["Added: " + ", ".join(f'"{text}"' for text in texts)] if texts else []


def checklist_update_text(message: Any) -> str:
    """``[Checklist update] <title>`` plus what changed, for a checklist service message; "" otherwise.

    Task texts come from the embedded ``checklist_message``, which the Bot API marks optional; when
    it is missing the task ids stand in."""
    done = getattr(message, "checklist_tasks_done", None)
    added = getattr(message, "checklist_tasks_added", None)
    change = done if done is not None else added
    if change is None:
        return ""
    checklist = _checklist_of(getattr(change, "checklist_message", None))
    names = {getattr(task, "id", None): _one_line(getattr(task, "text", "")) for task in checklist.tasks} if checklist else {}
    lines = _change_lines(done, added, names)
    if not lines:
        return ""
    title = _one_line(checklist.title) if checklist else ""
    return "\n".join([f"{_UPDATE_HEADER} {title}".rstrip(), *lines])


class TelegramStructuredInboundMixin:
    """Handlers for structured inbound messages on ``TelegramAdapter`` (relies on its intake helpers)."""

    async def _handle_location_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming location/venue pin messages."""
        msg = self._effective_update_message(update)
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        if not self._gate_or_observe(msg, update, MessageType.LOCATION):
            return
        venue = getattr(msg, "venue", None)
        location = getattr(venue, "location", None) if venue else getattr(msg, "location", None)
        if not location:
            return
        lat = getattr(location, "latitude", None)
        lon = getattr(location, "longitude", None)
        if lat is None or lon is None:
            return
        parts = ["[The user shared a location pin.]"]
        if venue:
            title = getattr(venue, "title", None)
            address = getattr(venue, "address", None)
            if title:
                parts.append(f"Venue: {title}")
            if address:
                parts.append(f"Address: {address}")
        parts += [
            f"latitude: {lat}", f"longitude: {lon}", f"Map: https://www.google.com/maps/search/?api=1&query={lat},{lon}",
            "Ask what they'd like to find nearby (restaurants, cafes, etc.) and any preferences."]
        event = self._build_message_event(msg, MessageType.LOCATION, update_id=update.update_id)
        event.text = "\n".join(parts)
        await self.handle_message(self._apply_telegram_group_observe_attribution(event))

    async def _handle_checklist_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Checklist service message (tasks marked done or not done, tasks added): never an agent turn.

        Ticking a task is not someone talking to the bot (the rule Slack applies to housekeeping
        subtypes, #110778). It passes the same authorization and group gates as text: where text
        would be dispatched (a DM, a free-response chat) it is logged; in a group that observes
        unmentioned messages it joins the observed transcript like other skipped chatter.
        """
        msg = self._effective_update_message(update)
        text = checklist_update_text(msg)
        if not text:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg, level=logging.INFO, what="checklist update from unauthorized user")
            return
        if self._should_process_message(msg):
            logger.info("[Telegram] Checklist update observed (no agent turn): chat=%s from=%s",
                        self._chat_id_str(msg), getattr(getattr(msg, "from_user", None), "id", None))
        elif self._should_observe_unmentioned_group_message(msg):
            event = self._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
            event.text = text
            self._observe_unmentioned_group_message(msg, MessageType.TEXT, update_id=update.update_id, event=event)

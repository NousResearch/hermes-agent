"""Declarative cards that general plugins may present on capable surfaces."""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

_CARD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _clean_text(value: str, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"plugin card {field} must not be empty")
    return text


def _action_id(label: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not value:
        value = f"action-{hashlib.sha256(label.encode('utf-8')).hexdigest()[:12]}"
    return value[:64]


@dataclass(frozen=True, slots=True)
class PluginCardAction:
    """A labelled action routed to one command registered by the card's plugin."""

    label: str
    command: str
    args: str = ""
    id: str | None = None

    def __post_init__(self) -> None:
        label = _clean_text(self.label, "action label")
        command = str(self.command).strip().lower().lstrip("/").replace(" ", "-")
        action_id = str(self.id).strip().lower() if self.id is not None else _action_id(label)
        if not _CARD_ID_RE.fullmatch(action_id):
            raise ValueError("plugin card action id must match [a-z0-9][a-z0-9_-]{0,63}")
        if not command:
            raise ValueError("plugin card action command must not be empty")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "args", str(self.args))
        object.__setattr__(self, "id", action_id)

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id or "", "label": self.label, "command": self.command, "args": self.args}


@dataclass(frozen=True, slots=True)
class PluginCard:
    """Small, host-rendered plugin card. Domain behavior belongs in action commands."""

    title: str
    body: str
    actions: tuple[PluginCardAction, ...] = ()
    id: str = "card"

    def __post_init__(self) -> None:
        card_id = str(self.id).strip().lower()
        if not _CARD_ID_RE.fullmatch(card_id):
            raise ValueError("plugin card id must match [a-z0-9][a-z0-9_-]{0,63}")
        title = _clean_text(self.title, "title")
        body = _clean_text(self.body, "body")
        actions = tuple(self.actions)
        if not all(isinstance(action, PluginCardAction) for action in actions):
            raise TypeError("plugin card actions must be PluginCardAction values")
        ids = [action.id for action in actions]
        if len(ids) != len(set(ids)):
            raise ValueError("plugin card action ids must be unique")
        object.__setattr__(self, "id", card_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "actions", actions)

    def text_fallback(self) -> str:
        return f"{self.title}\n\n{self.body}"


CardPublisher = Callable[[dict], object]
_publisher: ContextVar[CardPublisher | None] = ContextVar("plugin_card_publisher", default=None)


@contextmanager
def card_publisher_scope(publisher: CardPublisher) -> Iterator[None]:
    """Bind the current surface's nonblocking card sink for this execution."""
    token = _publisher.set(publisher)
    try:
        yield
    finally:
        _publisher.reset(token)


def attributed_card(plugin_id: str, plugin_name: str, card: PluginCard) -> dict:
    """Add host-owned plugin attribution to a card's declarative payload."""
    return {
        "id": card.id,
        "plugin_id": plugin_id,
        "plugin_name": plugin_name,
        "title": card.title,
        "body": card.body,
        "actions": [action.as_dict() for action in card.actions],
    }


def present_plugin_card(plugin_id: str, plugin_name: str, card: PluginCard) -> bool:
    """Publish to the current capable surface; return ``False`` when unsupported."""
    if not isinstance(card, PluginCard):
        raise TypeError("publish_card expects a PluginCard")
    publisher = _publisher.get()
    if publisher is None:
        return False
    return publisher(attributed_card(plugin_id, plugin_name, card)) is not False

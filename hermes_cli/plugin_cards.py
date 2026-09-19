"""Declarative cards that general plugins may present on capable surfaces."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

def _clean_text(value: str, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"plugin card {field} must not be empty")
    return text


@dataclass(frozen=True, slots=True)
class PluginCardAction:
    """A labelled action routed to one command registered by the card's plugin."""

    label: str
    command: str
    args: str = ""
    def __post_init__(self) -> None:
        label = _clean_text(self.label, "action label")
        command = str(self.command).strip().lower().lstrip("/").replace(" ", "-")
        if not command:
            raise ValueError("plugin card action command must not be empty")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "args", str(self.args))

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "command": self.command, "args": self.args}


@dataclass(frozen=True, slots=True)
class PluginCard:
    """Small, host-rendered plugin card. Domain behavior belongs in action commands."""

    title: str
    body: str
    actions: tuple[PluginCardAction, ...]

    def __post_init__(self) -> None:
        title = _clean_text(self.title, "title")
        body = _clean_text(self.body, "body")
        actions = tuple(self.actions)
        if not all(isinstance(action, PluginCardAction) for action in actions):
            raise TypeError("plugin card actions must be PluginCardAction values")
        if not actions:
            raise ValueError("plugin card must have at least one action")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "actions", actions)

    def text_fallback(self) -> str:
        return f"{self.title}\n\n{self.body}"


CardPublisher = Callable[[dict], object]
_publisher: ContextVar[CardPublisher | None] = ContextVar("plugin_card_publisher", default=None)


@contextmanager
def card_publisher_scope(publisher: CardPublisher) -> Iterator[None]:
    """Bind the current surface's direct notice sink for this execution."""
    token = _publisher.set(publisher)
    try:
        yield
    finally:
        _publisher.reset(token)


def attributed_card(plugin_id: str, plugin_name: str, card: PluginCard) -> dict:
    """Add host-owned plugin attribution to a card's declarative payload."""
    return {
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

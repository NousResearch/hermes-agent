"""Structured pending-turn input queue for Hermes (orchestrator Phase 2).

Hermes receives user input from several surfaces (Telegram, CLI, TUI, API) and,
when it is already busy, that input piles up in ad-hoc shapes: raw strings,
``(text, images)`` tuples, an identity-tagged "captured while busy" tuple, a
per-session single ``MessageEvent`` slot, etc.  That is enough for simple
queueing but it flattens *boundaries* and *provenance* -- a follow-up that is a
slash command, a media attachment, a correction, or a hard "new topic" wall
looks the same as a stray prose fragment once it has been merged.

This module introduces a small, explicit, *serializable* representation of one
pending input -- :class:`PendingTurnItem` -- plus a tiny ordered container --
:class:`PendingTurnQueue` -- and conversion helpers that round-trip the existing
legacy payloads losslessly.  The point is to preserve order, coalescing
boundaries, media metadata and source/session info so a later orchestrator can
decide whether a follow-up should *append to the current focus*, *steer the
active worker*, *start a new focused task*, or *ask for clarification* -- without
that decision being made (or precluded) here.

Scope discipline:

* This is **not** a task registry, a worker/background lane, a model-based
  follow-up classifier, or any "Ralph" focused-agent implementation.  Those are
  later phases.  This is only the *input substrate* they will build on.
* This module is a leaf: it imports nothing from ``cli.py`` or the ``gateway``
  package, holds no global mutable state, and every :class:`PendingTurnItem`
  field except ``raw`` is a plain JSON-serializable value -- so an item is safe
  to log, persist, or eventually move across a process boundary.  ``raw`` is the
  one local-process passthrough (the original legacy payload / ``MessageEvent``)
  and is intentionally dropped by :meth:`PendingTurnItem.to_dict`.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterable, Iterator

__all__ = [
    "SOURCE_TELEGRAM",
    "SOURCE_CLI",
    "SOURCE_TUI",
    "SOURCE_API",
    "SOURCE_UNKNOWN",
    "KIND_TEXT",
    "KIND_COMMAND",
    "KIND_MEDIA",
    "KIND_ATTACHMENT",
    "KIND_CONTROL",
    "BOUNDARY_COALESCE",
    "BOUNDARY_HARD",
    "BOUNDARY_CAPTION",
    "BOUNDARY_COMMAND",
    "INTEGRATED_BUSY_PAYLOAD",
    "PendingTurnItem",
    "PendingTurnQueue",
    "make_integrated_busy_payload",
    "is_integrated_busy_payload",
    "unwrap_integrated_busy_payload",
    "looks_like_slash_command",
    "legacy_cli_payload_is_coalescible_text",
    "from_legacy_cli_payload",
    "maybe_to_legacy_cli_payload",
    "from_gateway_event",
    "coalesced_text",
]

# --------------------------------------------------------------------------
# Vocabulary -- plain strings on purpose (boring, serializable, forward-compat).
# Unknown values are tolerated everywhere; these constants are the documented
# set, not an enum that rejects newcomers.
# --------------------------------------------------------------------------

# Where the input came from.
SOURCE_TELEGRAM = "telegram"
SOURCE_CLI = "cli"
SOURCE_TUI = "tui"
SOURCE_API = "api"
SOURCE_UNKNOWN = "unknown"

# What the input *is*.
KIND_TEXT = "text"            # a plain prose fragment
KIND_COMMAND = "command"      # a slash command / control message ("/busy", "/stop")
KIND_MEDIA = "media"          # inline media (photo, video, voice, audio, sticker)
KIND_ATTACHMENT = "attachment"  # a file-like attachment (document)
KIND_CONTROL = "control"      # non-text structured payload (location, opaque)

# How the item relates to its neighbours when coalescing adjacent fragments.
BOUNDARY_COALESCE = "coalesce"  # may merge with adjacent coalescible text
BOUNDARY_HARD = "hard"          # a wall: never merge across it
BOUNDARY_CAPTION = "caption"    # text that belongs to an adjacent media item
BOUNDARY_COMMAND = "command"    # a command: a wall, and never folded into text

_KNOWN_SOURCES = frozenset(
    {SOURCE_TELEGRAM, SOURCE_CLI, SOURCE_TUI, SOURCE_API, SOURCE_UNKNOWN}
)
_KNOWN_KINDS = frozenset(
    {KIND_TEXT, KIND_COMMAND, KIND_MEDIA, KIND_ATTACHMENT, KIND_CONTROL}
)
_KNOWN_BOUNDARIES = frozenset(
    {BOUNDARY_COALESCE, BOUNDARY_HARD, BOUNDARY_CAPTION, BOUNDARY_COMMAND}
)


# --------------------------------------------------------------------------
# Integrated-busy sentinel
#
# The CLI tags a plain-text fragment typed while Hermes is busy under
# ``/busy integrated`` as ``(INTEGRATED_BUSY_PAYLOAD, text)`` so the drain step
# can tell busy-time follow-ups apart from ordinary idle messages.  The sentinel
# lives here (next to the structured representation that understands it) and is
# re-exported by ``cli.py``.  It must NOT be a string: image payloads are also
# two-tuples ``(caption, images)`` and a caption such as ``"integrated_busy"``
# must stay a normal image payload.  It is identity-only and deliberately not
# serializable -- it never crosses a process boundary; a ``PendingTurnItem``
# carries the same information in the JSON-safe ``origin_busy`` flag instead.
# --------------------------------------------------------------------------
INTEGRATED_BUSY_PAYLOAD = object()


def make_integrated_busy_payload(text: str) -> tuple:
    """Wrap *text* as an integrated busy-time CLI fragment."""
    return (INTEGRATED_BUSY_PAYLOAD, text)


def is_integrated_busy_payload(item: Any) -> bool:
    """Return True if *item* is an integrated busy-time CLI fragment."""
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and item[0] is INTEGRATED_BUSY_PAYLOAD
    )


def unwrap_integrated_busy_payload(item: Any) -> Any:
    """Return the inner payload of an integrated fragment, else *item* unchanged."""
    return item[1] if is_integrated_busy_payload(item) else item


# --------------------------------------------------------------------------
# Slash-command detection
#
# Mirrors ``cli._looks_like_slash_command`` on purpose: keeping a faithful copy
# here means this module stays a leaf (no import of the multi-thousand-line
# ``cli`` module just to classify a string).  ``/busy`` is a command;
# ``/Users/foo/bar.md fix this`` is a pasted path that merely starts with ``/``.
# --------------------------------------------------------------------------
def looks_like_slash_command(text: Any) -> bool:
    if not isinstance(text, str) or not text.startswith("/"):
        return False
    parts = text.split()
    if not parts:
        return False
    first_word = parts[0]
    # After the leading "/", a command name contains no further "/"; a path does.
    return "/" not in first_word[1:]


def _new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# PendingTurnItem
# --------------------------------------------------------------------------
@dataclass
class PendingTurnItem:
    """One unit of pending user input, with its boundary and provenance.

    Construct with keyword arguments.  Every field except ``raw`` is a plain
    JSON-serializable value; ``raw`` holds the original payload (a legacy CLI
    ``str`` / tuple, a gateway ``MessageEvent``, ...) for lossless local
    round-tripping and is excluded from :meth:`to_dict`.
    """

    source: str = SOURCE_UNKNOWN
    kind: str = KIND_TEXT
    text: str | None = None
    media_refs: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    boundary: str = BOUNDARY_COALESCE
    session_key: str | None = None
    reply_to: str | None = None
    # Free-form hint about which task/focus this input is about (e.g. a task id
    # or a short label).  A later orchestrator may populate/consume it; nothing
    # in Phase 2 sets it automatically.
    task_hint: str | None = None
    # True when this fragment was captured while a foreground agent was busy
    # (e.g. CLI ``/busy integrated``).  Lets a later orchestrator tell
    # "follow-up to what I'm doing right now" apart from "a fresh idle message".
    origin_busy: bool = False
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=_new_id)
    raw: Any | None = None

    # -- introspection ----------------------------------------------------
    @property
    def is_text(self) -> bool:
        return self.kind == KIND_TEXT

    @property
    def is_command(self) -> bool:
        return self.kind == KIND_COMMAND

    @property
    def has_media(self) -> bool:
        return bool(self.media_refs)

    def is_coalescible_text(self) -> bool:
        """True when this item may be joined with adjacent coalescible text.

        A non-empty plain-text fragment whose boundary is ``coalesce``.
        Commands, media, attachments, captions and hard boundaries are not
        coalescible.
        """
        return (
            self.kind == KIND_TEXT
            and self.boundary == BOUNDARY_COALESCE
            and isinstance(self.text, str)
            and bool(self.text)
        )

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict.  Drops ``raw`` without touching it."""
        data: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "raw":
                continue
            value = getattr(self, f.name)
            # Copy mutable list fields so callers cannot mutate this item via the
            # serialized view.  Do not deep-copy arbitrary objects here: every
            # serialized field is intentionally plain/JSON-safe, while ``raw``
            # may be an uncopyable local gateway event and is skipped entirely.
            if isinstance(value, list):
                value = list(value)
            data[f.name] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingTurnItem":
        """Rebuild from :meth:`to_dict` output (unknown keys ignored)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def copy(self, **overrides: Any) -> "PendingTurnItem":
        return replace(self, **overrides)


# --------------------------------------------------------------------------
# PendingTurnQueue
# --------------------------------------------------------------------------
class PendingTurnQueue:
    """An ordered, in-memory queue of :class:`PendingTurnItem`.

    A thin wrapper over a ``deque`` -- intentionally *not* a ``queue.Queue``
    subclass: callers that need cross-thread safety should hold their own lock
    (the CLI uses this only transiently inside its single-threaded drain loop;
    the gateway, when it adopts this, runs on one asyncio loop).
    """

    def __init__(self, items: Iterable[PendingTurnItem] | None = None) -> None:
        self._items: deque[PendingTurnItem] = deque(items or ())

    # -- container protocol ----------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[PendingTurnItem]:
        # Non-consuming: iterate a snapshot so callers can drain afterwards.
        return iter(list(self._items))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PendingTurnQueue({list(self._items)!r})"

    # -- mutation ---------------------------------------------------------
    def append(self, item: PendingTurnItem) -> None:
        self._items.append(item)

    def appendleft(self, item: PendingTurnItem) -> None:
        self._items.appendleft(item)

    def extend(self, items: Iterable[PendingTurnItem]) -> None:
        self._items.extend(items)

    def clear(self) -> None:
        self._items.clear()

    # -- inspection -------------------------------------------------------
    def peek(self) -> PendingTurnItem | None:
        return self._items[0] if self._items else None

    def snapshot(self) -> list[PendingTurnItem]:
        return list(self._items)

    # -- consumption ------------------------------------------------------
    def pop(self) -> PendingTurnItem | None:
        """Remove and return the head item, or None when empty."""
        return self._items.popleft() if self._items else None

    def drain_coalescible_text_until_boundary(
        self, *, origin_busy: bool | None = None
    ) -> list[PendingTurnItem]:
        """Pop and return the leading run of coalescible-text items.

        Items are removed from the front of the queue as long as
        :meth:`PendingTurnItem.is_coalescible_text` holds and -- when
        *origin_busy* is not None -- their ``origin_busy`` flag equals
        *origin_busy*.  The first item that breaks the run (a command, media,
        attachment, a hard boundary, or a text fragment with a different
        ``origin_busy``) is left in place, so what remains in the queue starts at
        that boundary.  Returns the popped items in order (possibly empty).
        """
        run: list[PendingTurnItem] = []
        while self._items:
            head = self._items[0]
            if not head.is_coalescible_text():
                break
            if origin_busy is not None and head.origin_busy != origin_busy:
                break
            run.append(self._items.popleft())
        return run


# --------------------------------------------------------------------------
# Conversion helpers -- legacy CLI payloads
# --------------------------------------------------------------------------
def legacy_cli_payload_is_coalescible_text(payload: Any) -> bool:
    """True when a legacy CLI ``_pending_input`` payload is a coalescible text.

    Matches the predicate the CLI's plain ``queue``-mode coalescer uses for its
    *first* item: a ``str`` that does not look like a slash command.  (Empty
    strings qualify here -- mirroring the legacy guard -- but never start a merge
    run because :meth:`PendingTurnItem.is_coalescible_text` requires non-empty
    text, mirroring the legacy loop check.)
    """
    return isinstance(payload, str) and not looks_like_slash_command(payload)


def from_legacy_cli_payload(
    payload: Any,
    *,
    source: str = SOURCE_CLI,
    session_key: str | None = None,
) -> PendingTurnItem:
    """Lift a legacy CLI ``_pending_input`` payload into a :class:`PendingTurnItem`.

    Recognised shapes:

    * ``str`` -> a ``command`` item if it looks like a slash command, else a
      ``coalesce`` text item.
    * ``(INTEGRATED_BUSY_PAYLOAD, inner)`` -> the inner payload, lifted as above
      but with ``origin_busy=True`` (a non-``str`` inner becomes an opaque
      ``control`` item).
    * ``(caption, [paths...])`` -> a ``media`` item with a hard boundary
      (``caption`` rides in ``text``; the list contents become ``media_refs``).
    * anything else -> an opaque ``control`` item with a hard boundary.

    ``raw`` is always set to the original *payload* so
    :func:`maybe_to_legacy_cli_payload` can hand it back unchanged.
    """
    # Integrated busy-time tag: unwrap, then classify the inner value.
    if is_integrated_busy_payload(payload):
        inner = unwrap_integrated_busy_payload(payload)
        if isinstance(inner, str):
            if looks_like_slash_command(inner):
                return PendingTurnItem(
                    source=source,
                    kind=KIND_COMMAND,
                    text=inner,
                    boundary=BOUNDARY_COMMAND,
                    session_key=session_key,
                    origin_busy=True,
                    raw=payload,
                )
            return PendingTurnItem(
                source=source,
                kind=KIND_TEXT,
                text=inner,
                boundary=BOUNDARY_COALESCE,
                session_key=session_key,
                origin_busy=True,
                raw=payload,
            )
        # Pathological: integrated tag wrapping a non-string.  Preserve it.
        return PendingTurnItem(
            source=source,
            kind=KIND_CONTROL,
            text=None,
            boundary=BOUNDARY_HARD,
            session_key=session_key,
            origin_busy=True,
            raw=payload,
        )

    # Plain string: slash command vs prose.
    if isinstance(payload, str):
        if looks_like_slash_command(payload):
            return PendingTurnItem(
                source=source,
                kind=KIND_COMMAND,
                text=payload,
                boundary=BOUNDARY_COMMAND,
                session_key=session_key,
                raw=payload,
            )
        return PendingTurnItem(
            source=source,
            kind=KIND_TEXT,
            text=payload,
            boundary=BOUNDARY_COALESCE,
            session_key=session_key,
            raw=payload,
        )

    # ``(caption, [paths...])`` image/media tuple as produced by the CLI.
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[1], list)
    ):
        caption, refs = payload
        return PendingTurnItem(
            source=source,
            kind=KIND_MEDIA,
            text=caption if isinstance(caption, str) and caption else None,
            media_refs=[str(r) for r in refs],
            boundary=BOUNDARY_HARD,
            session_key=session_key,
            raw=payload,
        )

    # Anything else: keep it intact behind an opaque control item.
    return PendingTurnItem(
        source=source,
        kind=KIND_CONTROL,
        text=None,
        boundary=BOUNDARY_HARD,
        session_key=session_key,
        raw=payload,
    )


def maybe_to_legacy_cli_payload(item: PendingTurnItem) -> Any:
    """Best-effort inverse of :func:`from_legacy_cli_payload`.

    If *item* still carries its original ``raw`` payload, that is returned
    verbatim (the lossless common case).  Otherwise a payload is reconstructed
    from the structured fields: a ``media``/``attachment`` item with refs becomes
    ``(text or "", media_refs)``; anything with text becomes that text; an empty
    item falls back to ``""``.
    """
    if item.raw is not None:
        return item.raw
    if item.media_refs and item.kind in (KIND_MEDIA, KIND_ATTACHMENT):
        return (item.text or "", list(item.media_refs))
    if isinstance(item.text, str):
        return item.text
    return ""


# --------------------------------------------------------------------------
# Conversion helpers -- gateway MessageEvent
# --------------------------------------------------------------------------
# Map ``MessageType`` *values* (the enum's lowercase string values) to a
# (kind, boundary) pair.  Duck-typed on ``.value`` so this module need not import
# ``gateway.platforms.base``.
_GATEWAY_MESSAGE_TYPE_MAP: dict[str, tuple[str, str]] = {
    "text": (KIND_TEXT, BOUNDARY_COALESCE),
    "command": (KIND_COMMAND, BOUNDARY_COMMAND),
    "photo": (KIND_MEDIA, BOUNDARY_HARD),
    "video": (KIND_MEDIA, BOUNDARY_HARD),
    "audio": (KIND_MEDIA, BOUNDARY_HARD),
    "voice": (KIND_MEDIA, BOUNDARY_HARD),
    "sticker": (KIND_MEDIA, BOUNDARY_HARD),
    "document": (KIND_ATTACHMENT, BOUNDARY_HARD),
    "location": (KIND_CONTROL, BOUNDARY_HARD),
}


def _gateway_source_name(event: Any) -> str:
    platform = getattr(getattr(event, "source", None), "platform", None)
    if isinstance(platform, str) and platform:
        return platform.lower()
    name = getattr(platform, "value", None) or getattr(platform, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    return SOURCE_UNKNOWN


def _gateway_message_type_value(event: Any) -> str:
    message_type = getattr(event, "message_type", None)
    if isinstance(message_type, str):
        return message_type.lower()
    value = getattr(message_type, "value", None)
    if isinstance(value, str):
        return value.lower()
    name = getattr(message_type, "name", None)
    if isinstance(name, str):
        return name.lower()
    return ""


def _gateway_created_at(event: Any) -> float:
    ts = getattr(event, "timestamp", None)
    to_ts = getattr(ts, "timestamp", None)
    if callable(to_ts):
        try:
            return float(to_ts())
        except Exception:
            pass
    return time.time()


def from_gateway_event(event: Any, session_key: str | None = None) -> PendingTurnItem:
    """Lift a gateway ``MessageEvent`` into a :class:`PendingTurnItem`.

    Duck-typed (reads ``text``, ``message_type``, ``media_urls``,
    ``media_types``, ``reply_to_message_id``, ``source.platform``, ``timestamp``)
    so it does not depend on the ``gateway`` package -- which keeps this module a
    leaf and the conversion usable as a one-way bridge while the gateway still
    owns its ``Dict[str, MessageEvent]`` slot model.

    Slash-command text is classified as a ``command`` item even when the
    underlying ``message_type`` is ``TEXT`` (Telegram delivers ``/foo`` as TEXT),
    so commands keep a hard boundary and are never folded into a coalesced text
    run.  Media events keep ``boundary=hard`` and their caption (if any) in
    ``text`` -- the album/burst *merge* behaviour stays where it is today
    (``gateway.platforms.base.merge_pending_message_event``); this representation
    just refuses to dissolve media into a text run.  ``raw`` is the original
    event.
    """
    text = getattr(event, "text", None)
    text = text if isinstance(text, str) and text else None
    media_refs = list(getattr(event, "media_urls", None) or [])
    media_types = list(getattr(event, "media_types", None) or [])

    mtype = _gateway_message_type_value(event)
    kind, boundary = _GATEWAY_MESSAGE_TYPE_MAP.get(
        mtype,
        (KIND_TEXT, BOUNDARY_COALESCE),
    )

    # Unknown/future/plugin-specific message types may still carry media refs.
    # Preserve media as a hard boundary even when the message_type string is not
    # in our map, otherwise captions from new adapters could be folded into plain
    # text and lose attachment provenance.
    if kind == KIND_TEXT and media_refs:
        kind, boundary = KIND_MEDIA, BOUNDARY_HARD
    # Telegram (and friends) deliver "/cmd" as a TEXT message; treat it as a
    # command so it is never coalesced into prose.  Conversely, some adapters may
    # over-eagerly mark any slash-prefixed path as COMMAND; preserve path-like
    # text as coalescible prose rather than making it a hard command boundary.
    if kind == KIND_TEXT and looks_like_slash_command(text):
        kind, boundary = KIND_COMMAND, BOUNDARY_COMMAND
    elif kind == KIND_COMMAND and text and not looks_like_slash_command(text):
        kind, boundary = KIND_TEXT, BOUNDARY_COALESCE
    # A media/attachment event with no refs but with text is really just text.
    if kind in (KIND_MEDIA, KIND_ATTACHMENT) and not media_refs and text:
        kind, boundary = KIND_TEXT, BOUNDARY_COALESCE

    reply_to = getattr(event, "reply_to_message_id", None)
    return PendingTurnItem(
        source=_gateway_source_name(event),
        kind=kind,
        text=text,
        media_refs=media_refs,
        media_types=media_types,
        boundary=boundary,
        session_key=session_key,
        reply_to=str(reply_to) if reply_to is not None else None,
        created_at=_gateway_created_at(event),
        raw=event,
    )


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def coalesced_text(items: Iterable[PendingTurnItem], *, sep: str = "\n\n") -> str:
    """Join the ``text`` of *items* with *sep* (blanks/None skipped)."""
    return sep.join(it.text for it in items if isinstance(it.text, str) and it.text)

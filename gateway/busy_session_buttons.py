"""Per-message busy-session control buttons.

Callback-data length: Telegram limits ``InlineKeyboardButton.callback_data``
to 64 bytes.  ``bs:<primitive>:<session_key>`` exceeds that on
group/forum sessions whose keys embed long chat/thread/user ids
(e.g. ``agent:main:telegram:supergroup:-1001234567890:thread:42:user:9876543210``).
When the literal session_key would overflow, ``build_buttons`` switches
to a short hash handle (``bs:<primitive>:#<8-char-hash>``) and the
caller is expected to register the mapping via the returned
``BusySessionKeyboardSpec.handle_map`` so ``parse_callback_data`` can
later resolve it back.



Renders three platform-neutral buttons — /steer, /interrupt, /stop —
attached to the running tool bubble (or a standalone control bubble
fallback) while the agent is actively processing. Lets the user pick
an outcome on a specific follow-up message instead of changing the
global ``display.busy_input_mode``.

The buttons dispatch into upstream's existing primitives:

- ``steer``     → ``running_agent.steer(text)``    (mid-turn text injection)
- ``interrupt`` → ``running_agent.interrupt(text)`` (halt + replay text as next turn)
- ``stop``      → ``_interrupt_and_clear_session(...)`` (halt + idle, no replay)

Reactions on the user's follow-up messages give visible acknowledgement
that a tap was processed:

- ``steer``     → 👍
- ``interrupt`` → ⚡
- ``stop``      → 🙊

Telegram's reaction whitelist constrains the emoji choice; these three
glyphs are confirmed accepted on default-permission chats.

Callback data wire format::

    bs:<primitive>:<session_key>

Example: ``bs:steer:tg/123/456``. Each platform serializes the
:class:`BusySessionButton` spec into its native widget — Telegram
uses ``InlineKeyboardMarkup``, Discord uses ``discord.ui.View``,
Slack uses Block Kit ``actions`` blocks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CALLBACK_PREFIX = "bs"

# Telegram's hard cap on InlineKeyboardButton.callback_data (UTF-8 bytes).
# Discord and Slack are looser, but we standardize on the strictest so
# the same callback string round-trips on all three platforms.
CALLBACK_MAX_BYTES = 64

# Reserved prefix on hashed handles inside callback_data so the parser
# can distinguish "literal session_key" from "registered handle".
HANDLE_SIGIL = "#"
HANDLE_LEN = 12  # truncated SHA-256 hex; 12 chars = 48 bits, ~10^14 keys

PRIMITIVE_STEER = "steer"
PRIMITIVE_INTERRUPT = "interrupt"
PRIMITIVE_STOP = "stop"

BUSY_SESSION_PRIMITIVES: Tuple[str, ...] = (
    PRIMITIVE_STEER,
    PRIMITIVE_INTERRUPT,
    PRIMITIVE_STOP,
)

REACTION_STEER = "👍"
REACTION_INTERRUPT = "⚡"
REACTION_STOP = "🙊"

REACTION_BY_PRIMITIVE: dict[str, str] = {
    PRIMITIVE_STEER: REACTION_STEER,
    PRIMITIVE_INTERRUPT: REACTION_INTERRUPT,
    PRIMITIVE_STOP: REACTION_STOP,
}

# The label rendered on each button. Slash-prefixed so users mentally
# associate the button with the equivalent slash-command they already
# know. Keep these short — Telegram, Discord, and Slack all clip long
# labels (Telegram observed truncating mid-word at ~14 chars).
BUTTON_LABELS: dict[str, str] = {
    PRIMITIVE_STEER: "/steer",
    PRIMITIVE_INTERRUPT: "/interrupt",
    PRIMITIVE_STOP: "/stop",
}


@dataclass(frozen=True)
class BusySessionButton:
    """Platform-neutral spec for one button on the busy-session row."""

    primitive: str
    label: str
    callback_data: str


def _session_key_handle(session_key: str) -> str:
    """Stable short hash for a session_key, prefixed with ``HANDLE_SIGIL``."""
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:HANDLE_LEN]
    return f"{HANDLE_SIGIL}{digest}"


@dataclass
class BusySessionKeyboardSpec:
    """Built buttons + the (possibly empty) handle→session_key map.

    Callers attaching the keyboard MUST merge ``handle_map`` into their
    runner-side resolution table so a later tap can translate the
    hashed handle back to the real session_key.  Empty when no
    sessions exceeded the 64-byte callback cap.
    """

    buttons: List[BusySessionButton]
    handle_map: Dict[str, str] = field(default_factory=dict)


def build_buttons(session_key: str) -> List[BusySessionButton]:
    """Return the ordered list of buttons for a given session.

    Order matches reading direction: steer (least disruptive),
    interrupt (halt + redirect), stop (halt + idle).

    Backwards-compatible API: returns just the button list.  When
    callers also need the hashed-handle map (long session keys), they
    should call :func:`build_buttons_with_handles` instead.
    """
    return build_buttons_with_handles(session_key).buttons


def build_buttons_with_handles(session_key: str) -> BusySessionKeyboardSpec:
    """Build buttons and surface the handle map for long session keys."""
    handle_map: Dict[str, str] = {}
    payload = session_key
    # Worst-case fits-test: longest primitive is "interrupt" (9 bytes).
    longest = max(BUSY_SESSION_PRIMITIVES, key=len)
    candidate = f"{CALLBACK_PREFIX}:{longest}:{payload}"
    if len(candidate.encode("utf-8")) > CALLBACK_MAX_BYTES:
        payload = _session_key_handle(session_key)
        handle_map[payload] = session_key
    buttons = [
        BusySessionButton(
            primitive=p,
            label=BUTTON_LABELS[p],
            callback_data=f"{CALLBACK_PREFIX}:{p}:{payload}",
        )
        for p in BUSY_SESSION_PRIMITIVES
    ]
    return BusySessionKeyboardSpec(buttons=buttons, handle_map=handle_map)


def parse_callback_data(
    data: Optional[str],
    *,
    handle_resolver: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, str]]:
    """Parse ``bs:<primitive>:<session_key>`` (or hashed handle).

    When ``handle_resolver`` is provided and the third segment starts
    with :data:`HANDLE_SIGIL`, the function looks the handle up in the
    resolver and returns the original session_key.  An unresolved
    handle (e.g. gateway restart cleared the table) returns ``None`` —
    caller should treat the tap as stale.
    """
    if not data or not isinstance(data, str):
        return None
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None
    primitive, payload = parts[1], parts[2]
    if primitive not in BUSY_SESSION_PRIMITIVES:
        return None
    if not payload:
        return None
    if payload.startswith(HANDLE_SIGIL):
        if handle_resolver is None:
            return None
        resolved = handle_resolver.get(payload)
        if not resolved:
            return None
        return primitive, resolved
    return primitive, payload


def reaction_for(primitive: str) -> Optional[str]:
    """Return the acknowledgement reaction emoji for a primitive."""
    return REACTION_BY_PRIMITIVE.get(primitive)


def status_text(primitive: str, *, language_hint: Optional[str] = None) -> str:
    """Short toast/ack text shown after a successful button tap.

    ``language_hint`` is currently unused — reserved for future
    localization based on the matched halt-phrase language.
    """
    if primitive == PRIMITIVE_STEER:
        return "👍 Steered into current run."
    if primitive == PRIMITIVE_INTERRUPT:
        return "⚡ Interrupted — your message will start the next turn."
    if primitive == PRIMITIVE_STOP:
        return "🙊 Stopped."
    return ""


__all__ = [
    "BUSY_SESSION_PRIMITIVES",
    "BUTTON_LABELS",
    "BusySessionButton",
    "BusySessionKeyboardSpec",
    "CALLBACK_MAX_BYTES",
    "CALLBACK_PREFIX",
    "HANDLE_LEN",
    "HANDLE_SIGIL",
    "PRIMITIVE_INTERRUPT",
    "PRIMITIVE_STEER",
    "PRIMITIVE_STOP",
    "REACTION_BY_PRIMITIVE",
    "REACTION_INTERRUPT",
    "REACTION_STEER",
    "REACTION_STOP",
    "build_buttons",
    "build_buttons_with_handles",
    "parse_callback_data",
    "reaction_for",
    "status_text",
]

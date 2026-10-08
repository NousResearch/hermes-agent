"""Per-room opt-out of link preview cards via empty bundled previews (MSC4095).

A sender may attach its own previews to an event; an EMPTY list is an explicit "no previews for this
event" that clients honoring bundled previews (Sable, Beeper) render as no cards, without the
reader's global or the room's preview settings changing. ``com.beeper.linkpreviews`` is the
unstable name older clients still read.
"""

from __future__ import annotations

from typing import Any, Dict, Set

LINK_PREVIEW_OPT_OUT_KEYS = ("m.url_previews", "com.beeper.linkpreviews")
_TRUE_WORDS = frozenset({"true", "1", "yes", "on", "all"})
_FALSE_WORDS = frozenset({"", "false", "0", "no", "off", "none"})


def parse_link_preview_opt_out(raw: Any) -> tuple[bool, Set[str]]:
    """``disable_link_previews`` → ``(all_rooms, room_ids)``: true opts out every room, a room-ID
    list/CSV only those rooms, unset/false none."""
    if raw is None or isinstance(raw, bool):
        return bool(raw), set()
    if isinstance(raw, str) and raw.strip().lower() in _TRUE_WORDS:
        return True, set()
    if isinstance(raw, str) and raw.strip().lower() in _FALSE_WORDS:
        return False, set()
    items = raw if isinstance(raw, list) else str(raw).split(",")
    return False, {str(r).strip() for r in items if str(r).strip()}


def link_previews_disabled(opt_out: tuple[bool, Set[str]], room_id: Any) -> bool:
    all_rooms, rooms = opt_out
    return all_rooms or str(room_id) in rooms


def apply_link_preview_opt_out(msg_content: Dict[str, Any]) -> None:
    """Mark a text event as carrying no link previews."""
    for key in LINK_PREVIEW_OPT_OUT_KEYS:
        msg_content[key] = []

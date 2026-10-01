"""Who wrote the user side of a turn, carried from the dispatcher into the recipient's turn.

The dispatcher sets ``HERMES_TURN_AUTHOR`` on the recipient's one-shot subprocess only. A cached
gateway agent sees several authors over its lifetime, so the author is read per turn, never per agent.
"""

from __future__ import annotations

import json
import os
import socket
import unicodedata
from typing import Any, Dict, Mapping, MutableMapping, Optional

TURN_AUTHOR_ENV = "HERMES_TURN_AUTHOR"

_MAX_FIELD_LEN = 200


_DROPPED_CATEGORIES = frozenset({"Cc", "Cs", "Cn", "Co"})
_TRUTHY = frozenset({"true", "1", "yes"})


def _clean_text(value: Any) -> Optional[str]:
    """Strip whitespace, control and unassigned characters, then cap the length. None when nothing is left.
    Format characters and non-breaking spaces stay so emoji sequences and display names survive."""
    if not isinstance(value, str):
        return None
    text = "".join(ch for ch in value if unicodedata.category(ch) not in _DROPPED_CATEGORIES).strip()
    if not text:
        return None
    return text[:_MAX_FIELD_LEN]


def _bot_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return isinstance(value, (bool, int)) and bool(value)


def bot_author_id(profile: str, origin: Optional[str] = None) -> str:
    """``bot:<profile>`` for a bot on the recipient's own install, ``bot:<origin>/<profile>`` for one on another
    install. The origin is the Desktop's connection id on a relayed dm and the sender's hostname on a peer dm."""
    origin = (origin or "").strip()
    return f"bot:{origin}/{profile}" if origin else f"bot:{profile}"


def plugin_author_id(plugin_id: str) -> str:
    """``plugin:<id>`` for a turn a plugin queued through ``PluginContext.inject_message``.

    Deliberately not a ``bot:`` author: a peer DM relayed from another install and a local
    plugin are different senders and must not share one a2a session.
    """
    return f"plugin:{(plugin_id or '').strip()}"


def plugin_turn_author(plugin_id: str, name: str = "") -> Optional[Dict[str, Any]]:
    """The author of a plugin-injected turn.

    A plugin writes its own words — usually an instruction TO the agent — so the turn is never
    the human's. Without an author a memory provider that derives facts from user messages
    stores the plugin's text as durable facts about the user; with ``is_bot`` the turn is routed
    to the plugin's own a2a session and profile writes are refused for its duration. None when
    the caller named no plugin, which leaves the turn unattributed exactly as before.
    """
    clean_id = _clean_text(plugin_id)
    if not clean_id:
        return None
    return {"id": plugin_author_id(clean_id), "name": _clean_text(name) or clean_id, "is_bot": True}


def local_origin() -> str:
    """This machine's hostname as an author-id origin, cleaned like any author field. Empty when unknown."""
    try:
        host = socket.gethostname()
    except Exception:
        return ""
    # A slash would split the id into a different origin and profile at parse time.
    return (_clean_text(host) or "").replace("/", "")


def parse_turn_author(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize a dict or JSON string into ``{"id", "name", "is_bot"}``; None for anything else or without id and name.
    The id is whatever the transport knows the sender by: ``bot:<profile>`` on a bot-mode delivery inside one
    install, ``bot:<connection>/<profile>`` when the Desktop relayed it from another machine,
    ``bot:<hostname>/<profile>`` on a peer dm, the platform user id elsewhere.
    An ``origin`` field qualifies a bare ``bot:<profile>`` id the same way."""
    try:
        if isinstance(raw, (str, bytes)):
            raw = json.loads(raw)
        if not isinstance(raw, Mapping):
            return None
        author = {
            "id": _clean_text(raw.get("id")),
            "name": _clean_text(raw.get("name")),
            "is_bot": _bot_flag(raw.get("is_bot")),
        }
        if author["id"] is None and author["name"] is None:
            return None
        origin = _clean_text(raw.get("origin"))
        if origin and author["id"] and author["id"].startswith("bot:") and "/" not in author["id"]:
            author["id"] = _clean_text(bot_author_id(author["id"][len("bot:"):], origin))
        return author
    except Exception:
        return None


def plugin_author_for_event(event: Any) -> Optional[Dict[str, Any]]:
    """The author for a turn a plugin dispatched through the messaging gateway, else None.

    The dispatch stamps ``hermes_plugin_injection`` on the event; the restored source is the
    human's, so without this the plugin's instruction is indistinguishable from something the
    user typed. Keyed on the injection marker alone, so an event that merely names a plugin in
    its metadata is not re-attributed. A plugin that names its own ``author`` (a chat bridge
    relaying a real person) keeps that author instead of the plugin default.
    """
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, Mapping) or not metadata.get("hermes_plugin_injection"):
        return None
    return parse_turn_author(metadata.get("hermes_plugin_author")) or plugin_turn_author(
        str(metadata.get("hermes_plugin_id") or ""))


def turn_author_from_env(environ: Mapping[str, str] = os.environ) -> Optional[Dict[str, Any]]:
    """The author the dispatcher placed in ``HERMES_TURN_AUTHOR``, or None."""
    return parse_turn_author(environ.get(TURN_AUTHOR_ENV))


def take_turn_author_from_env(environ: MutableMapping[str, str] = os.environ) -> Optional[Dict[str, Any]]:
    """Read and remove ``HERMES_TURN_AUTHOR`` so subprocesses started during the turn do not inherit it."""
    return parse_turn_author(environ.pop(TURN_AUTHOR_ENV, None))


def turn_author_env(author: Dict[str, Any]) -> Dict[str, str]:
    """The environment entry a dispatcher merges into a child's env."""
    return {TURN_AUTHOR_ENV: json.dumps(author, separators=(",", ":"))}


def a2a_key(author: Optional[Dict[str, Any]]) -> Optional[str]:
    """``a2a:<bot id>``, the shared name for a bot author's turns. None for a human or an id-less bot."""
    if not isinstance(author, dict) or not author.get("is_bot") or not author.get("id"):
        return None
    return f"a2a:{author['id']}"

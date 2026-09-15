"""Shared helpers for canonicalising WhatsApp sender identity.

The bridge can surface one human as a LID (``999...@lid``) or a phone JID
(``1555...@s.whatsapp.net``) within one conversation. Authorisation (:mod:`gateway.run`) and
session keys (:mod:`gateway.session`) both resolve aliases here so they never drift apart;
plugins should use :func:`canonical_whatsapp_identifier` to line up with Hermes' session keys.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Set

from hermes_constants import get_hermes_dir

logger = logging.getLogger(__name__)

# WhatsApp JIDs are numeric (or plus-prefixed) with ``@``/``.``/``:`` separators.
# Explicit ASCII class so full-width digits / Unicode word chars can't sneak through.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9@.+\-]+$")

# "Just a phone number": optional ``+`` then digits and human separators.
# Anything carrying ``@`` is already a JID (``@g.us``, ``@lid``, ``status@broadcast``).
_BARE_PHONE_RE = re.compile(r"^\+?[\d\s().\-]+$")


def normalize_whatsapp_identifier(value: str) -> str:
    """Strip JID/LID/device/plus syntax down to the bare numeric identifier:
    ``"6012:47@s.whatsapp.net"``, ``"6012@lid"`` and ``"+6012"`` all become ``"6012"``."""
    return str(value or "").strip().replace("+", "", 1).split(":", 1)[0].split("@", 1)[0]


def to_whatsapp_jid(value: str) -> str:
    """Normalize an *outbound* target to a bridge-safe JID (inverse of normalize).  Baileys'
    ``jidDecode`` crashes on a bare phone, so bare phones become ``<digits>@s.whatsapp.net``;
    ``user:device@domain`` collapses to ``user@domain``; anything else is returned unchanged
    so the bridge can surface a real error.  ``""`` for empty input."""
    if not value:
        return ""
    normalized = str(value).strip()
    if ":" in normalized and "@" in normalized:
        prefix, _, domain = normalized.partition("@")
        normalized = f"{prefix.split(':', 1)[0]}@{domain}"
    if "@" in normalized:
        return normalized
    if _BARE_PHONE_RE.fullmatch(normalized):
        digits = re.sub(r"\D+", "", normalized)
        if digits:
            return f"{digits}@s.whatsapp.net"
    return normalized


def expand_whatsapp_aliases(identifier: str) -> Set[str]:
    """All identifiers transitively reachable via the bridge's ``lid-mapping-*.json`` files;
    always includes the normalized input itself (empty set if it normalizes to empty)."""
    normalized = normalize_whatsapp_identifier(identifier)
    if not normalized:
        return set()
    session_dir = get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")
    resolved: Set[str] = set()
    queue = [normalized]
    while queue:
        current = queue.pop(0)
        # _SAFE_IDENTIFIER_RE: defense-in-depth against path separators / traversal in the
        # ``lid-mapping-{current}`` filename (the fixed prefix already prevents escape).
        if not current or current in resolved or not _SAFE_IDENTIFIER_RE.match(current):
            continue
        resolved.add(current)
        for suffix in ("", "_reverse"):
            mapping_path = session_dir / f"lid-mapping-{current}{suffix}.json"
            if not mapping_path.exists():
                continue
            try:
                raw = json.loads(mapping_path.read_text(encoding="utf-8"))
                mapped = normalize_whatsapp_identifier(raw)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("whatsapp_identity: failed to read %s: %s", mapping_path, exc)
                continue
            if mapped and mapped not in resolved:
                queue.append(mapped)
    return resolved


def canonical_whatsapp_identifier(identifier: str) -> str:
    """Stable sender identity across phone-JID/LID variants (DM ``chat_id`` and group
    ``participant_id`` alike): the shortest alias from :func:`expand_whatsapp_aliases`, which
    degrades to the normalized input when no mapping files exist.  ``""`` for empty input."""
    normalized = normalize_whatsapp_identifier(identifier)
    if not normalized:
        return ""
    # expand_whatsapp_aliases includes ``normalized`` itself, so min() degrades to it
    # when no lid-mapping files are present.
    aliases = expand_whatsapp_aliases(normalized)
    return min(aliases, key=lambda c: (len(c), c))

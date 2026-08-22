"""Shared helpers for canonicalising WhatsApp sender identity.

WhatsApp's bridge can surface the same human under two different JID shapes
within a single conversation:

- LID form: ``999999999999999@lid``
- Phone form: ``15551234567@s.whatsapp.net``

Both the authorisation path (:mod:`gateway.run`) and the session-key path
(:mod:`gateway.session`) need to collapse these aliases to a single stable
identity. This module is the single source of truth for that resolution so
the two paths can never drift apart.

Public helpers:

- :func:`normalize_whatsapp_identifier` — strip JID/LID/device/plus syntax
  down to the bare numeric identifier.
- :func:`canonical_whatsapp_identifier` — walk the bridge's
  ``lid-mapping-*.json`` files and return a stable canonical identity
  across phone/LID variants.
- :func:`expand_whatsapp_aliases` — return the full alias set for an
  identifier. Used by authorisation code that needs to match any known
  form of a sender against an allow-list.

Plugins that need per-sender behaviour on WhatsApp (role-based routing,
per-contact authorisation, policy gating in a gateway hook) should use
``canonical_whatsapp_identifier`` so their bookkeeping lines up with
Hermes' own session keys.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Set

logger = logging.getLogger(__name__)

# WhatsApp JIDs are numeric (or plus-prefixed numeric) with optional
# ``@``, ``.`` and ``:`` separators. ``\w`` is pinned to ASCII so
# full-width digits / Unicode word chars can't sneak through.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9@.+\-]+$")

# Mapping files that have already been reported as unreadable. A corrupt or
# unreadable file stays that way, and expansion now runs on the inbound path
# for every routed message, so the warning fires once per file rather than per
# lookup. Bounded because the key is a filesystem path we constructed.
_WARNED_MAPPING_PATHS: Set[str] = set()

from hermes_constants import get_hermes_dir, get_process_hermes_home


def _session_dir():
    """Locate the bridge's session store, ignoring per-task profile overrides.

    The WhatsApp connection is a PROCESS-level asset: one bridge, one paired
    device, one ``whatsapp/session`` directory, written under the home the
    gateway was launched with. Profile homes never have one.

    ``get_hermes_dir`` defaults to :func:`get_hermes_home`, which follows the
    context-local override installed by ``_profile_runtime_scope`` for the
    duration of a routed turn. Resolving the session store through it made
    every ``lid-mapping-*.json`` lookup miss the moment a message was routed
    to a profile, so a group sender's LID no longer resolved to their phone
    number and every phone-keyed gate silently stopped matching them —
    ``whatsapp.allow_admin_from``, ``group_allow_admin_from``, ``allow_from``
    and the pairing store alike. Observed live 2026-08-14: with
    ``HERMES_HOME`` at the gateway home ``160868067209361@lid`` expanded to
    ``{160868067209361, 972547401660}``, but under the routed profile home it
    expanded to ``{160868067209361}`` alone, and the configured admin was
    rejected from their own group with "/new is admin-only here".

    :func:`get_process_hermes_home` is the right seam rather than
    ``get_default_hermes_root``: a gateway may legitimately be *launched* on a
    profile home, and then the bridge's session store is under THAT home, not
    the root above it.
    """
    return get_hermes_dir(
        "platforms/whatsapp/session", "whatsapp/session", home=get_process_hermes_home()
    )


def normalize_whatsapp_identifier(value: str) -> str:
    """Strip WhatsApp JID/LID syntax down to its stable numeric identifier.

    Accepts any of the identifier shapes the WhatsApp bridge may emit:
    ``"60123456789@s.whatsapp.net"``, ``"60123456789:47@s.whatsapp.net"``,
    ``"60123456789@lid"``, or a bare ``"+601****6789"`` / ``"60123456789"``.
    Returns just the numeric identifier (``"60123456789"``) suitable for
    equality comparisons.

    Useful for plugins that want to match sender IDs against
    user-supplied config (phone numbers in ``config.yaml``) without
    worrying about which variant the bridge happens to deliver.
    """
    return (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )


# A target that is "just a phone number" — optional leading ``+`` then digits
# and the usual human separators (spaces, dots, dashes, parens). Anything that
# already carries an ``@`` is a fully-qualified JID and must pass through
# untouched (group ``@g.us``, LID ``@lid``, ``status@broadcast`` etc.).
_BARE_PHONE_RE = re.compile(r"^\+?[\d\s().\-]+$")


def to_whatsapp_jid(value: str) -> str:
    """Normalize an *outbound* WhatsApp target to a bridge-safe JID.

    Baileys' ``jidDecode`` crashes on a bare phone number — it expects a
    fully-qualified JID such as ``50766715226@s.whatsapp.net``. This helper
    is the inverse of :func:`normalize_whatsapp_identifier`: instead of
    stripping a JID down to its numeric core for comparison, it *builds* the
    JID a send must use.

    Behaviour:

    - ``"+50766715226"`` / ``"50766715226"`` → ``"50766715226@s.whatsapp.net"``
    - ``"50766715226@s.whatsapp.net"`` → unchanged
    - ``"group-id@g.us"`` / ``"130631430344750@lid"`` → unchanged
    - ``"user:device@s.whatsapp.net"`` style colon-before-``@`` → ``@`` form
    - anything that isn't a recognizable bare phone → returned unchanged so
      the bridge can surface a meaningful error rather than us mangling it.

    Returns ``""`` for an empty/whitespace input.
    """
    if not value:
        return ""

    normalized = str(value).strip()
    # Drop a device suffix before the domain: ``user:device@domain`` is a
    # legacy Baileys shape whose ``:device`` part is not addressable — collapse
    # it to ``user@domain``. (Mirrors normalize_whatsapp_identifier, which
    # splits the bare id on ``:`` for the same reason.)
    if ":" in normalized and "@" in normalized:
        prefix, _, domain = normalized.partition("@")
        normalized = f"{prefix.split(':', 1)[0]}@{domain}"

    # Already a fully-qualified JID — leave it alone.
    if "@" in normalized:
        return normalized

    if _BARE_PHONE_RE.fullmatch(normalized):
        digits = re.sub(r"\D+", "", normalized)
        if digits:
            return f"{digits}@s.whatsapp.net"

    return normalized


def expand_whatsapp_aliases(identifier: str) -> Set[str]:
    """Resolve WhatsApp phone/LID aliases via bridge session mapping files.

    Returns the set of all identifiers transitively reachable through the
    bridge's ``$HERMES_HOME/whatsapp/session/lid-mapping-*.json`` files,
    starting from ``identifier``. The result always includes the
    normalized input itself, so callers can safely ``in`` check against
    the return value without a separate fallback branch.

    Returns an empty set if ``identifier`` normalizes to empty.
    """
    normalized = normalize_whatsapp_identifier(identifier)
    if not normalized:
        return set()

    session_dir = _session_dir()
    resolved: Set[str] = set()
    queue = [normalized]

    while queue:
        current = queue.pop(0)
        if not current or current in resolved:
            continue
        # Defense-in-depth: reject identifiers that could sneak path
        # separators / traversal segments into the ``lid-mapping-{current}``
        # filename below. The hardcoded ``lid-mapping-`` prefix already
        # prevents escape via pathlib's component split (an attacker can't
        # create ``lid-mapping-..`` as a real directory in session_dir), but
        # this keeps the identifier space to the characters WhatsApp JIDs
        # actually use and avoids depending on that filesystem-layout
        # invariant.
        if not _SAFE_IDENTIFIER_RE.match(current):
            continue

        resolved.add(current)
        for suffix in ("", "_reverse"):
            mapping_path = session_dir / f"lid-mapping-{current}{suffix}.json"
            if not mapping_path.exists():
                continue
            try:
                mapped = normalize_whatsapp_identifier(
                    json.loads(mapping_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError) as exc:
                # Degrading to the raw identifier is safe (a LID never matches
                # a phone-keyed allow-list), but it is invisible: the operator
                # sees "admin refused in their own group" or a DM landing on
                # the default profile, with nothing in the log connecting that
                # to an unreadable mapping file. Warn on the first failure per
                # file so the cause is one grep away.
                key = str(mapping_path)
                if key not in _WARNED_MAPPING_PATHS:
                    _WARNED_MAPPING_PATHS.add(key)
                    logger.warning(
                        "whatsapp_identity: cannot read alias mapping %s (%s) — "
                        "this sender's phone/LID forms will not resolve to each "
                        "other, so phone-keyed admin lists and profile routes "
                        "may not match them",
                        mapping_path,
                        exc,
                    )
                continue
            if mapped and mapped not in resolved:
                queue.append(mapped)

    return resolved


def canonical_whatsapp_identifier(identifier: str) -> str:
    """Return a stable WhatsApp sender identity across phone-JID/LID variants.

    WhatsApp may surface the same person under either a phone-format JID
    (``60123456789@s.whatsapp.net``) or a LID (``1234567890@lid``). This
    applies to a DM ``chat_id`` *and* to the ``participant_id`` of a
    member inside a group chat — both represent a user identity, and the
    bridge may flip between the two for the same human.

    This helper reads the bridge's ``whatsapp/session/lid-mapping-*.json``
    files, walks the mapping transitively, and picks the shortest
    (numeric-preferred) alias as the canonical identity.
    :func:`gateway.session.build_session_key` uses this for both WhatsApp
    DM chat_ids and WhatsApp group participant_ids, so callers get the
    same session-key identity Hermes itself uses.

    Plugins that need per-sender behaviour (role-based routing,
    authorisation, per-contact policy) should use this so their
    bookkeeping lines up with Hermes' session bookkeeping even when
    the bridge reshuffles aliases.

    Returns an empty string if ``identifier`` normalizes to empty. If no
    mapping files exist yet (fresh bridge install), returns the
    normalized input unchanged.
    """
    normalized = normalize_whatsapp_identifier(identifier)
    if not normalized:
        return ""

    # expand_whatsapp_aliases always includes `normalized` itself in the
    # returned set, so the min() below degrades gracefully to `normalized`
    # when no lid-mapping files are present.
    aliases = expand_whatsapp_aliases(normalized)
    return min(aliases, key=lambda candidate: (len(candidate), candidate))

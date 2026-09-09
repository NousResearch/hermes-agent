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


def canonical_phone_jid(jid: str) -> str:
    """One canonical form for phone-addressed WhatsApp identities: ``@s.whatsapp.net``.

    The three WhatsApp transports speak different wire dialects — the Baileys bridge
    uses ``@s.whatsapp.net`` natively, WAHA/NOWEB documents ``@c.us``, Cloud API uses
    bare ``wa_id`` digits — and allowlists, admin lists and session keys must compare
    equal for the same human.  Each adapter normalizes its inbound wire form to this
    canonical form at the boundary (``_map_payload`` / ``_build_message_event``) and
    renders it back to the engine's dialect on the way out via :func:`to_engine_chat_id`.
    ``@c.us`` and bare digits become ``@s.whatsapp.net``; ``@s.whatsapp.net``, ``@g.us``,
    ``@lid``, broadcasts and anything already suffixed pass through unchanged (LID
    resolution needs the adapter's alt/cache and happens before this call)."""
    value = str(jid or "").strip()
    if not value:
        return ""
    if "@" in value:
        local, _, domain = value.partition("@")
        if domain == "c.us":
            return f"{local}@s.whatsapp.net"
        return value
    if _BARE_PHONE_RE.fullmatch(value):
        digits = re.sub(r"\D+", "", value)
        if digits:
            return f"{digits}@s.whatsapp.net"
    return value


def to_engine_chat_id(chat_id: str, engine: str) -> str:
    """Render the canonical internal id into the dialect *engine* expects on its wire.

    ``"waha"`` wants ``@c.us`` (WAHA docs are explicit); ``"cloud"`` wants bare
    ``wa_id`` digits; ``"baileys"`` and unknown engines get the canonical
    ``@s.whatsapp.net`` unchanged — every dialect change is additive at the boundary,
    so a new transport only adds a branch here."""
    value = canonical_phone_jid(chat_id)
    engine = str(engine or "").strip().lower()
    if engine == "waha" and value.endswith("@s.whatsapp.net"):
        return value.split("@", 1)[0] + "@c.us"
    if engine == "cloud" and "@" in value:
        return value.split("@", 1)[0]
    return value


def canonicalize_id_list(entries, default_cc: str = "") -> list:
    """Canonicalize every entry of a config/env/API allowlist (allow_from,
    group_allow_from, allow_admin_from, user_allowed_commands payloads…) into the
    internal standard, accepting every shape an operator might write:

    - bare digits in any phone format (``0812…``, ``+62 812…``, ``(555) 123-4567``)
    - any JID dialect (``@c.us``, ``@s.whatsapp.net``, ``@g.us``, ``@lid``)
    - group JIDs, broadcasts, LIDs (unchanged — resolution is adapter-side)

    Group ids and non-phone entries pass through :func:`canonical_phone_jid` too, so
    one call covers both user and group lists; ``"*"`` and blank entries are preserved
    / dropped respectively.  *default_cc* (the bot's own country code) lets national
    trunk formats like ``0812…`` resolve; without it they are suffixed as-is and the
    adapter re-normalizes with its own country at init.  Returns a list (YAML
    round-trip friendly); input order is kept and duplicates collapse."""
    if entries is None:
        return []
    if isinstance(entries, str):
        items = [part for part in (p.strip() for p in entries.split(",")) if part]
    elif isinstance(entries, (list, tuple, set, frozenset)):
        items = [str(item).strip() for item in entries]
    else:
        items = [str(entries).strip()]
    out: list = []
    for item in items:
        if not item or item == "*":
            if item == "*" and "*" not in out:
                out.append(item)
            continue
        if re.fullmatch(r"\+?[\d\s().\-]{7,}", item) and "@" not in item:
            # Phone-shaped: normalize through the full path (trunk digits, dial-out
            # prefixes) when the home country is known; otherwise the plain canonical
            # form applies and the adapter re-normalizes with its own country.
            e164 = normalize_phone_e164(item, default_cc)
            canonical = f"{e164}@s.whatsapp.net" if e164 else canonical_phone_jid(item)
        else:
            canonical = canonical_phone_jid(item)
        if canonical and canonical not in out:
            out.append(canonical)
    return out


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


# ITU country calling codes (E.164). Longest-prefix matched, so ``1`` vs ``1809`` style
# overlaps resolve to the longer code. NANP sub-areas (``1-24x``/``1-6xx``/``1-8xx``) are
# covered by ``1`` itself; the list only needs the codes that start a number.
_ITU_COUNTRY_CODES: frozenset[str] = frozenset({
    "1", "7", "20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41", "43", "44",
    "45", "46", "47", "48", "49", "51", "52", "53", "54", "55", "56", "57", "58", "60", "61",
    "62", "63", "64", "65", "66", "81", "82", "84", "86", "90", "91", "92", "93", "94", "95",
    "98", "211", "212", "213", "216", "218", "220", "221", "222", "223", "224", "225", "226",
    "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239",
    "240", "241", "242", "243", "244", "245", "246", "248", "249", "250", "251", "252", "253",
    "254", "255", "256", "257", "258", "260", "261", "262", "263", "264", "265", "266", "267",
    "268", "269", "290", "291", "297", "298", "299", "350", "351", "352", "353", "354", "355",
    "356", "357", "358", "359", "370", "371", "372", "373", "374", "375", "376", "377", "378",
    "380", "381", "382", "383", "385", "386", "387", "389", "420", "421", "423", "500", "501",
    "502", "503", "504", "505", "506", "507", "508", "509", "590", "591", "592", "593", "594",
    "595", "596", "597", "598", "599", "670", "672", "673", "674", "675", "676", "677", "678",
    "679", "680", "681", "682", "683", "685", "686", "687", "688", "689", "690", "691", "692",
    "800", "808", "850", "852", "853", "855", "856", "870", "880", "881", "882", "883", "888",
    "960", "961", "962", "963", "964", "965", "966", "967", "968", "970", "971", "972", "973",
    "974", "975", "976", "977", "992", "993", "994", "995", "996", "998",
})

# E.164 caps at 15 digits; national numbers are never shorter than ~6 (7 with cc).
_MIN_E164_DIGITS = 7
_MAX_E164_DIGITS = 15
# Shortest national significant number accepted for "local number, prepend home cc".
_MIN_NSN_DIGITS = 7


def default_country_code(own_number: str) -> str:
    """Country calling code of *own_number* (the bot's own E.164 digits) via longest ITU
    prefix match; ``""`` when it cannot be determined.  This is how ``0812…`` on an
    Indonesian bot and ``(555) …`` on a US bot both resolve — the bot's own country is
    the only sane default for local-format input."""
    digits = re.sub(r"\D+", "", str(own_number or ""))
    for length in range(min(_MAX_E164_DIGITS, len(digits)), 0, -1):
        if digits[:length] in _ITU_COUNTRY_CODES and len(digits) - length >= _MIN_NSN_DIGITS:
            return digits[:length]
    return ""


def normalize_phone_e164(raw: str, default_cc: str = "") -> str:
    """Normalize a human-supplied phone number to bare E.164 digits (no ``+``).

    Accepts the formats people actually type: ``+62812…``, ``62 812-345…``, ``0812…``
    (national trunk — ``0`` replaced by *default_cc*), ``(555) 123-4567`` (no country —
    *default_cc* prepended), ``001…``/``011…`` (international dial-out prefixes), and
    numbers that already carry their country code.  ``default_cc`` comes from the bot's
    own number (:func:`default_country_code`) so the same input normalizes correctly on
    bots in different countries.  Returns ``""`` when *raw* is not a plausible phone
    number; already-international input never needs *default_cc*.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return ""
    if text.startswith("+"):
        # ``+`` contributes no digits and settles the question: already E.164.
        return digits if _MIN_E164_DIGITS <= len(digits) <= _MAX_E164_DIGITS else ""
    # International dial-out prefixes, longest first so ``0011`` (AU) wins over ``00``;
    # ``011`` is the NANP prefix. The remainder is already E.164 — return it as-is.
    for prefix in ("0011", "011", "00"):
        if digits.startswith(prefix):
            digits = digits[len(prefix):]
            return digits if _MIN_E164_DIGITS <= len(digits) <= _MAX_E164_DIGITS else ""
    if digits.startswith("0"):
        # National trunk digit: only meaningful with a known home country.
        if not default_cc:
            return ""
        digits = default_cc + digits[1:]
    elif not default_cc:
        # No home context and no ``+``/dial-out prefix: the digits could be anything,
        # and a wrong guess silently stores a dead allowlist entry.  Reject.
        return ""
    elif not _looks_international(digits, default_cc):
        # Local interpretation wins for ambiguous input (``5551234567`` on a US bot,
        # ``8123456789`` on an Indonesian bot): prepend the home country code.  A
        # *foreign* number must be typed with ``+`` — guessing a foreign code from a
        # bare digit prefix (``55`` = Brazil? or a local ``555…``?) silently stores a
        # dead allowlist entry, which is worse than requiring the ``+``.
        if len(digits) < _MIN_NSN_DIGITS:
            return ""
        digits = default_cc + digits
    if not (_MIN_E164_DIGITS <= len(digits) <= _MAX_E164_DIGITS):
        return ""
    return digits


# NANP (country code ``1``) national numbers are EXACTLY 10 digits — the only common
# case where local dialing ("5551234567") collides head-on with the country code.
_NANP_NSN_DIGITS = 10


def _nsn_plausible(digits: str, default_cc: str) -> bool:
    """Whether *digits* minus *default_cc* looks like a real national number for the
    home country.  Only the NANP's fixed-10 rule is worth encoding; elsewhere any
    remainder of at least ``_MIN_NSN_DIGITS`` is plausible."""
    rest = digits[len(default_cc):]
    if default_cc == "1":
        return len(rest) == _NANP_NSN_DIGITS
    return len(rest) >= _MIN_NSN_DIGITS


def _looks_international(digits: str, default_cc: str) -> bool:
    """Whether *digits* already carries the HOME country code rather than being a local
    number.  Starts-with-home-cc is only accepted when what follows is a plausible
    national number, so a NANP local number (``5551234567``) is not mistaken for
    ``1``-international.  Foreign codes are deliberately NOT detected here: ``55…``
    could be Brazil or a local ``555…`` — the local reading wins and foreign numbers
    are typed with ``+``."""
    if not digits.startswith(default_cc):
        return False
    return _nsn_plausible(digits, default_cc)

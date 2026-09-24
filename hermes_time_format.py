"""Locale-safe datetime formatting helpers for Hermes.

Split out of :mod:`hermes_time` so a long-running process that crossed an on-disk upgrade can
import this module fresh and still get ``safe_strftime`` even though ``hermes_time`` is already
cached in ``sys.modules`` from the revision that predated it. ``hermes_time`` re-exports these
names for compatibility.

Keep this module a stdlib-only leaf: importing ``hermes_time`` (or anything that imports it)
here would recreate the stale-cached-module hazard this split exists to remove — a module
imported for the first time *after* an upgrade must never need new names from a module the
process already has cached.
"""

from __future__ import annotations

import locale
import re
from datetime import datetime
from typing import Optional

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
# ASCII plus surrogateescape'd bytes only: the shape of native text decoded with the wrong codec.
_ESCAPED_BYTES_RE = re.compile(r"[\x00-\x7f\udc80-\udcff]*")


def _repair_surrogates(text: str, encoding: Optional[str] = None) -> str:
    """Make locale text JSON/UTF-8 safe; a no-op (same object) on valid text.

    Windows hands back zone names in the ANSI code page (``heure d'\\xe9t\\xe9``) but a UTF-8
    ``LC_CTYPE`` (UTF-8 mode, or a library flipping the process locale mid-run) decodes those
    bytes with ``surrogateescape`` into lone surrogates. Recover the bytes and decode them with
    the ANSI code page; when that is not possible, replace each surrogate with U+FFFD."""
    if text.isascii() or not _SURROGATE_RE.search(text):
        return text
    if _ESCAPED_BYTES_RE.fullmatch(text):
        try:
            return text.encode("ascii", "surrogateescape").decode(encoding or locale.getencoding())
        except (LookupError, UnicodeError):
            pass
    return _SURROGATE_RE.sub("\ufffd", text)


def safe_strftime(value: datetime, fmt: str) -> str:
    """``value.strftime(fmt)`` that never raises or returns lone surrogates over locale text.

    ``datetime.strftime`` splices ``tzname()`` into the format as UTF-8, so a zone name carrying
    surrogates (see ``_repair_surrogates``) raises ``UnicodeEncodeError`` before any string exists
    (#102910). On that failure ``%Z`` is rendered from the repaired name instead. Output for valid
    locale text is byte-identical to ``strftime`` (the system prompt must stay cache-stable)."""
    try:
        return _repair_surrogates(value.strftime(fmt))
    except UnicodeEncodeError:
        zone = _repair_surrogates(value.tzname() or "").replace("%", "%%")
        return _repair_surrogates(value.strftime(
            re.sub(r"%([%Z])", lambda m: zone if m.group(1) == "Z" else "%%", fmt)))
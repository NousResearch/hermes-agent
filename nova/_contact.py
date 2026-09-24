"""Checks for contact details entered through the Control Centre's settings forms.

Applied where a value is *entered* (``nova.branding``), not where a bundle is *loaded*: a
bundle already on disk with a value these reject must keep loading after an upgrade —
refusing it at load would take the control plane down to report a typo.

Deliberately narrow: each accepts what a real value looks like and refuses what would be
wrong everywhere it is used, without trying to be a full RFC parser.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from nova.errors import SpecError

#: Something@something.tld, no spaces. Deliverability is the mail server's question.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: A support link is shown to people and will be rendered as a link. ``javascript:`` or
#: ``data:`` there would run in the viewer's browser on a click, so only these schemes.
_LINK_SCHEMES = ("https://", "http://", "mailto:")


def check_email(value: str, *, field: str, source: Optional[Path]) -> str:
    if value and not _EMAIL.match(value):
        raise SpecError(f"{value!r} is not an email address", field=field, source=source)
    return value


def check_link(value: str, *, field: str, source: Optional[Path]) -> str:
    if value and not value.lower().startswith(_LINK_SCHEMES):
        raise SpecError(
            f"{value!r} must start with https://, http:// or mailto: — it is shown to people "
            "as a link, and any other kind of link can run code when clicked",
            field=field, source=source,
        )
    return value


def check_timezone(value: str, *, field: str, source: Optional[Path]) -> str:
    if not value:
        return value
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise SpecError(
            f"{value!r} is not a time zone; use an IANA name such as Europe/London or "
            "America/New_York",
            field=field, source=source,
        ) from None
    return value

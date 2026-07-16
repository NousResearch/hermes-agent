"""Reviewed normalization allowlist for refactor-equivalence goldens."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

ALLOWLIST: dict[str, str] = {}
_FORBIDDEN = re.compile(r"(time|date|_at$|_ts$|id$)", re.IGNORECASE)


class AllowlistError(AssertionError):
    """Raised when a normalization allowlist entry is not reviewable."""


def lint_allowlist(allowlist: Mapping[str, str] | Iterable[str] | None = None) -> None:
    entries = ALLOWLIST if allowlist is None else allowlist
    names = entries.keys() if isinstance(entries, Mapping) else entries
    bad = [name for name in names if _FORBIDDEN.search(str(name))]
    if bad:
        raise AllowlistError(
            "normalization allowlist may not hide clock/sequence fields: "
            + ", ".join(sorted(map(str, bad)))
        )


def normalize(value: Any, allowlist: Mapping[str, str] | Iterable[str] | None = None) -> Any:
    lint_allowlist(allowlist)
    return value


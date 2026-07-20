"""Deterministic identity normalization + resolution v1 (#12323 Phase 1)."""

from __future__ import annotations

import re
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.people.store import PeopleMessageStore


_PHONE_RE = re.compile(r"[^\d+]")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_identity(kind: str, value: str) -> str:
    """Normalize identity values for durable unique keys."""
    kind_l = (kind or "").strip().lower()
    raw = (value or "").strip()
    if kind_l in ("phone", "imessage") and raw:
        # keep leading + if present; strip formatting
        digits = _PHONE_RE.sub("", raw)
        if digits.startswith("00"):
            digits = "+" + digits[2:]
        if raw.startswith("+") and not digits.startswith("+"):
            digits = "+" + digits.lstrip("+")
        # US-ish 10-digit → +1
        if digits.isdigit() and len(digits) == 10:
            digits = "+1" + digits
        elif digits.isdigit() and len(digits) == 11 and digits.startswith("1"):
            digits = "+" + digits
        return digits
    if kind_l == "email":
        return raw.lower()
    if kind_l == "handle":
        return raw.lstrip("@").lower()
    return raw.lower()


def infer_kind(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return "handle"
    if _EMAIL_RE.match(v):
        return "email"
    digits = _PHONE_RE.sub("", v)
    if digits.isdigit() and len(digits) >= 10:
        return "phone"
    if v.startswith("+") and any(c.isdigit() for c in v):
        return "phone"
    return "handle"


def slugify_name(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return base or f"person-{uuid.uuid4().hex[:10]}"


class IdentityResolver:
    """Resolve handles → person_id with override > existing link > create."""

    def __init__(self, store: "PeopleMessageStore") -> None:
        self.store = store

    def resolve(
        self,
        value: str,
        *,
        kind: Optional[str] = None,
        display_name: Optional[str] = None,
        source: Optional[str] = None,
    ) -> str:
        k = kind or infer_kind(value)
        # 1) manual override
        ov = self.store.get_override_person(k, value)
        if ov:
            return ov
        # 2) existing identity
        existing = self.store.find_person_by_identity(k, value)
        if existing:
            return existing
        # 3) create person + link
        person_id = f"person-{uuid.uuid4().hex[:12]}"
        slug = slugify_name(display_name or value)
        self.store.upsert_person(
            person_id,
            display_name=display_name or value,
            slug=slug,
        )
        self.store.link_identity(
            person_id, k, value, source=source or "identity_resolver"
        )
        return person_id

"""The in-app offer the Nous Portal wants this profile to see, as a memo.

The Portal serves an offer on ``/api/oauth/account`` (``in_app_offer``; see the
account-service's ``src/server/in-app-offers.ts`` for the campaign model).
This module remembers the latest answer per profile so ``free_tier.status``
can report it without a network read, the same shape as the pending browser
challenge (``hermes_cli/anon_challenge.py``): the account fetch is the writer,
the status RPC is the reader.

There is deliberately no dismissal state here. Dismissal is a presentation
fact and lives on the client (keyed by offer id), so the backend never has to
be told what the user has already seen; the memo only mirrors what the Portal
currently serves, and an offer that stops being served disappears on the next
account fetch.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from hermes_cli.anon_auth import _mint_memo_key

_lock = threading.Lock()
# Profile memo key -> the offer dict as served by the Portal (already
# shape-checked by record_from_account_payload), or None once the Portal
# stops serving one.
_offers: Dict[str, Dict[str, str]] = {}

_REQUIRED_STRINGS = ("id", "title", "body", "cta_label", "cta_url")


def record_from_account_payload(payload: Dict[str, Any]) -> None:
    """Remember the offer this account fetch carried (or the absence of one).

    Called from the account-payload normalizer, so every path that refreshes
    Nous account info (sign-in, JWT refresh, the credits cold start) updates
    the memo. A payload whose ``in_app_offer`` is missing or malformed clears
    the memo: a malformed value is the Portal's contract breaking, and a
    missing key is a campaign that ended.
    """
    offer = payload.get("in_app_offer") if isinstance(payload, dict) else None
    checked = _validated(offer)
    key = _mint_memo_key()
    with _lock:
        if checked is None:
            _offers.pop(key, None)
        else:
            _offers[key] = checked


def pending_offer() -> Optional[Dict[str, str]]:
    """The offer the Portal currently serves this profile, or None."""
    with _lock:
        return dict(_offers[_key]) if (_key := _mint_memo_key()) in _offers else None


def _validated(value: Any) -> Optional[Dict[str, str]]:
    """The five wire fields as non-empty strings, or None.

    Shape-checks rather than trusting the JSON: the payload crosses a network
    boundary, and a half-formed offer must never reach a dialog.
    """
    if not isinstance(value, dict):
        return None
    checked: Dict[str, str] = {}
    for field_name in _REQUIRED_STRINGS:
        text = value.get(field_name)
        if not isinstance(text, str) or not text.strip():
            return None
        checked[field_name] = text
    return checked

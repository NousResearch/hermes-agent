"""The in-app offer the Nous Portal wants this profile to see, as a memo.

The Portal serves an offer on ``/api/oauth/account`` (``in_app_offer``; see the
account-service's ``src/server/in-app-offers.ts`` for the campaign model).
This module remembers the latest answer per profile so ``free_tier.status``
can report it without a network read, the same shape as the pending browser
challenge (``hermes_cli/anon_challenge.py``): an account fetch is the writer,
the status RPC is the reader.

Two writers keep the memo current. Every account fetch any other feature makes
records into it (``nous_account._info_from_account_payload``), and the desktop
calls ``in_app_offer.refresh`` on a slow cadence (``refresh_from_portal``) so
the answer exists even for a user whose session never fetches the account: a
valid JWT serves most account reads locally, and the session-start credits
seed runs only for a Nous session.

Dismissal is deliberately not stored here. It is a presentation fact and lives
on the client (keyed by offer id), so the backend never has to be told what the
user has already seen. The memo only mirrors what the Portal currently serves.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from hermes_cli.anon_auth import _mint_memo_key

logger = logging.getLogger(__name__)
_lock = threading.Lock()
# Profile memo key -> the offer dict as served by the Portal (already
# shape-checked by record_from_account_payload).
_offers: Dict[str, Dict[str, str]] = {}

_REQUIRED_STRINGS = ("id", "title", "body", "cta_label", "cta_url")


def record_from_account_payload(payload: Dict[str, Any]) -> None:
    """Remember the offer this account fetch carried (or the absence of one).

    A payload whose ``in_app_offer`` is missing or malformed clears the memo:
    a malformed value is the Portal's contract breaking, and a missing key is a
    campaign that ended or an account that no longer qualifies.
    """
    offer = payload.get("in_app_offer") if isinstance(payload, dict) else None
    checked = _validated(offer)
    key = _mint_memo_key()
    with _lock:
        if checked is None:
            _offers.pop(key, None)
        else:
            _offers[key] = checked


def clear() -> None:
    """Forget this profile's offer (the account it was evaluated for is gone)."""
    with _lock:
        _offers.pop(_mint_memo_key(), None)


def pending_offer() -> Optional[Dict[str, str]]:
    """The offer the Portal currently serves this profile, or None.

    Signed out means None, whatever the memo holds: the offer was evaluated
    for an account that is gone, so it must not outlive the login. Checking
    the login here (a local auth-store read, no network) covers every way out
    of an account (``hermes logout``, the desktop's disconnect, a quarantined
    token) without a hook in each.
    """
    key = _mint_memo_key()
    with _lock:
        offer = _offers.get(key)
    if offer is None:
        return None
    if not _signed_in():
        clear()
        return None
    return dict(offer)


def _signed_in() -> bool:
    """The profile holds a Nous identity the Portal could have evaluated: an
    account login (an access token), or a guest carrying the welcome trial.
    A guest's access token is minted on demand from its anon token, so its
    auth state can hold no ``access_token`` while still being that identity.
    Signed out, or a guest with no trial (connectors only), is neither."""
    from hermes_cli.anon_auth import guest_without_free_tier, held_connectors_token, is_guest_state
    from hermes_cli.auth import get_provider_auth_state

    state = get_provider_auth_state("nous") or {}
    if guest_without_free_tier(state):
        return False
    return is_guest_state(state) or bool(held_connectors_token(state))


def refresh_from_portal() -> Optional[Dict[str, str]]:
    """Fetch the account once with the stored token and return the offer.

    Uses ``stored_token_account_info``: it never refreshes the token, so this
    is safe to call while a session runs on any provider. Signed out (or a
    guest without inference, or a token about to expire) clears the memo
    instead of fetching. A failed fetch keeps the last answer: a transient
    portal error must not withdraw an offer the user is about to see.
    """
    from hermes_cli.nous_account import stored_token_account_info

    try:
        info = stored_token_account_info()
    except (OSError, ValueError) as exc:
        # urllib's network errors are OSErrors (URLError, timeouts); a bad JSON
        # body is a ValueError. Anything else is a bug and should surface.
        logger.info("in-app offer refresh failed, keeping the last answer: %s", exc)
        return pending_offer()
    if info is None and not _signed_in():
        clear()
        return None
    # A successful fetch recorded into the memo on its way through the
    # normalizer. None while still signed in means the stored token is about
    # to expire and was not refreshed (by design: this read must never write
    # the auth store), so there is no new answer: keep the last one. An idle
    # user's short-lived token is exactly this case, and clearing here would
    # withdraw the offer from the users it is meant for.
    return pending_offer()


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

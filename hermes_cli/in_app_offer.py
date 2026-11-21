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

Dismissals are kept per profile in ``in_app_offer_dismissed.json`` under the
profile home, and ``pending_offer`` never returns a dismissed offer. Desktop, the
Webapp and the dashboard keep separate browser storage, so a client-only
dismissal would show the same person the same offer once per surface; one file
per profile shows it once. Clients still keep their own record too, which
covers an older backend without ``dismiss``.

Each memo entry remembers the account it was evaluated for and when. A
different account on the profile (a logout and another login, a guest that
claimed an account, a switched portal) never sees it, and an entry older than
``MEMO_TTL_SECONDS`` is dropped: an offer that could not be re-checked for a day
(an idle user's expired token) may have ended.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Optional

from hermes_cli.anon_auth import _mint_memo_key

logger = logging.getLogger(__name__)
_lock = threading.Lock()
# Profile memo key -> (account identity, recorded at, the offer as served by the
# Portal, already shape-checked by record_from_account_payload).
_offers: dict[str, tuple[str, float, dict[str, str]]] = {}

#: An offer the Portal has not re-confirmed for this long is dropped.
MEMO_TTL_SECONDS = 24 * 60 * 60

_DISMISSED_FILE = "in_app_offer_dismissed.json"
#: Far more than campaigns ever run; keeps the file from growing forever.
_MAX_DISMISSED = 50


def _dismissed_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _DISMISSED_FILE


def dismissed_ids() -> list[str]:
    """Offer ids this profile closed, oldest first. A missing or unreadable file reads as none."""
    from utils.files import read_json_or_empty

    ids = read_json_or_empty(_dismissed_path()).get("ids")
    return [i for i in ids if isinstance(i, str) and i] if isinstance(ids, list) else []


def dismiss(offer_id: str) -> bool:
    """Close this offer for the profile for good, on every surface. Returns False for a blank id."""
    from utils.files import atomic_json_write

    if not isinstance(offer_id, str) or not offer_id.strip():
        return False
    with _lock:
        ids = [i for i in dismissed_ids() if i != offer_id] + [offer_id]
        atomic_json_write(_dismissed_path(), {"ids": ids[-_MAX_DISMISSED:]}, indent=None)
    return True

_REQUIRED_STRINGS = ("id", "title", "body", "cta_label", "cta_url")


def record_from_account_payload(payload: dict[str, Any]) -> None:
    """Remember the offer this account fetch carried (or the absence of one).

    A payload whose ``in_app_offer`` is missing or malformed clears the memo:
    a malformed value is the Portal's contract breaking, and a missing key is a
    campaign that ended or an account that no longer qualifies.
    """
    offer = payload.get("in_app_offer") if isinstance(payload, dict) else None
    checked = _validated(offer)
    key = _mint_memo_key()
    identity = _account_identity()
    with _lock:
        if checked is None or identity is None:
            _offers.pop(key, None)
        else:
            _offers[key] = (identity, time.time(), checked)


def clear() -> None:
    """Forget this profile's offer (the account it was evaluated for is gone)."""
    with _lock:
        _offers.pop(_mint_memo_key(), None)


def pending_offer() -> Optional[dict[str, str]]:
    """The offer the Portal currently serves this profile, or None.

    Signed out means None, whatever the memo holds: the offer was evaluated
    for an account that is gone, so it must not outlive the login. Checking
    the login here (a local auth-store read, no network) covers every way out
    of an account (``hermes logout``, the desktop's disconnect, a quarantined
    token) without a hook in each.
    """
    key = _mint_memo_key()
    with _lock:
        entry = _offers.get(key)
    if entry is None:
        return None
    identity, recorded_at, offer = entry
    if (
        not _signed_in()
        or _account_identity() != identity
        or time.time() - recorded_at > MEMO_TTL_SECONDS
    ):
        clear()
        return None
    if offer["id"] in dismissed_ids():
        return None
    return dict(offer)


def _account_identity() -> Optional[str]:
    """Which account the stored login is, stable across token refreshes: the portal plus the
    JWT's user and org, or for a guest its anon credential (its access token is minted on
    demand, so may be absent). None when there is no identity to key on."""
    from hermes_cli.anon_auth import is_guest_state
    from hermes_cli.auth import get_provider_auth_state
    from hermes_cli.auth_constants import _decode_jwt_claims
    from hermes_cli.nous_account import _portal_base_url

    state = get_provider_auth_state("nous") or {}
    portal = _portal_base_url(state) or ""
    if is_guest_state(state):
        anon = state.get("anon_token")
        if isinstance(anon, str) and anon:
            return f"{portal}:guest:{hashlib.sha256(anon.encode()).hexdigest()}"
    token = state.get("access_token")
    if not isinstance(token, str) or not token:
        return None
    claims = _decode_jwt_claims(token)
    sub, org = claims.get("sub"), claims.get("org_id")
    if isinstance(sub, str) and sub and isinstance(org, str) and org:
        return f"{portal}:jwt:{sub}:{org}"
    return f"{portal}:token:{hashlib.sha256(token.encode()).hexdigest()}"


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


def refresh_from_portal() -> Optional[dict[str, str]]:
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


def _validated(value: Any) -> Optional[dict[str, str]]:
    """The five wire fields as non-empty strings, with an https CTA, or None.

    Shape-checks rather than trusting the JSON: the payload crosses a network
    boundary, and a half-formed offer must never reach a dialog. The CTA must be
    an https URL with a host: the desktop's external opener also opens local
    paths and ``file:`` URLs, and the Portal only guarantees https at create.
    """
    if not isinstance(value, dict):
        return None
    checked: dict[str, str] = {}
    for field_name in _REQUIRED_STRINGS:
        text = value.get(field_name)
        if not isinstance(text, str) or not text.strip():
            return None
        checked[field_name] = text
    url = urllib.parse.urlsplit(checked["cta_url"].strip())
    if url.scheme != "https" or not url.hostname:
        return None
    checked["cta_url"] = checked["cta_url"].strip()
    # Optional: when the offer's promo stops being claimable, for "Claim by ..." copy. Kept only as
    # a parseable ISO instant; anything else is dropped (the offer itself still shows).
    until = value.get("claimable_until")
    if isinstance(until, str) and _iso_instant(until):
        checked["claimable_until"] = until
    return checked


def _iso_instant(text: str) -> bool:
    from datetime import datetime

    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


#: What the client tells the Portal about an offer, for the campaign's counts.
REPORTABLE_EVENTS = ("shown", "clicked")


def report_event(offer_id: str, event: str) -> bool:
    """Tell the Portal this offer was shown or clicked (``POST /api/oauth/account/in-app-offer-event``).

    Best-effort. The Portal counts each person once per offer and event, so a repeat from
    another surface or a retry is harmless. Uses the stored token as it is, never refreshing it
    (the same rule as ``refresh_from_portal``); a guest whose short-lived access token has lapsed
    sends its anon credential, which the route also accepts. Reports nothing when signed out or
    for an offer this profile was not served. Returns whether the Portal took it.
    """
    from hermes_cli.anon_auth import guest_without_free_tier, held_connectors_token, is_guest_state
    from hermes_cli.auth import get_provider_auth_state
    from hermes_cli.nous_account import _portal_base_url

    if event not in REPORTABLE_EVENTS:
        return False
    served = pending_offer()
    # Only an offer the Portal actually served this profile: the RPC carries a client-chosen
    # id, and the counter must not become a way to bump an arbitrary campaign.
    if served is None or served.get("id") != offer_id:
        return False
    state = get_provider_auth_state("nous") or {}
    if guest_without_free_tier(state):
        return False
    token = held_connectors_token(state, skew_seconds=60)
    if not token and is_guest_state(state):
        anon = state.get("anon_token")
        token = anon if isinstance(anon, str) and anon else None
    if not token:
        return False
    base = (_portal_base_url(state) or "https://portal.nousresearch.com").rstrip("/")
    req = urllib.request.Request(
        f"{base}/api/oauth/account/in-app-offer-event",
        data=json.dumps({"offer_id": offer_id, "event": event}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return 200 <= resp.status < 300
    except OSError as exc:
        # urllib's network and HTTP errors are OSErrors; a count is never worth an error.
        logger.info("in-app offer %s report failed: %s", event, exc)
        return False

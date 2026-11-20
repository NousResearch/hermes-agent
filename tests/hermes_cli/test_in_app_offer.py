"""``hermes_cli/in_app_offer.py``: the per-profile memo of the Portal's
``in_app_offer`` campaign answer, as served by the account payload."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone, UTC

import pytest

from hermes_cli import in_app_offer

OFFER = {
    "id": "offer_1",
    "title": "Welcome back",
    "body": "Here is an offer.",
    "cta_label": "See the offer",
    "cta_url": "https://portal.nousresearch.com/billing",
}


@pytest.fixture(autouse=True)
def _clear_memo():
    """The memo is module state; without this, an offer recorded here leaks
    into every later test that reads `free_tier.status` for the same
    profile key."""
    yield

    in_app_offer._offers.clear()


@pytest.fixture(autouse=True)
def _signed_in(monkeypatch):
    """A signed-in Nous login by default: the memo only answers for one.
    Tests about signing out override the auth state themselves."""
    from hermes_cli import auth

    state = {"nous": {"access_token": "account-token", "auth_method": "oauth"}}
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda provider: state.get(provider))
    return state


def test_records_a_valid_offer():
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    assert in_app_offer.pending_offer() == OFFER


def test_a_missing_key_clears_the_memo():
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    in_app_offer.record_from_account_payload({})

    assert in_app_offer.pending_offer() is None


def test_a_malformed_offer_never_lands():
    in_app_offer.record_from_account_payload({"in_app_offer": {"id": "x", "title": "T"}})

    assert in_app_offer.pending_offer() is None


def test_a_malformed_offer_clears_a_earlier_valid_one():
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    in_app_offer.record_from_account_payload({"in_app_offer": "broken"})

    assert in_app_offer.pending_offer() is None


def test_records_are_scoped_per_profile(monkeypatch):
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    monkeypatch.setattr(in_app_offer, "_mint_memo_key", lambda: "other-home")

    assert in_app_offer.pending_offer() is None

    in_app_offer.record_from_account_payload({"in_app_offer": {**OFFER, "id": "offer_2"}})

    other_profile = in_app_offer.pending_offer()
    assert other_profile is not None and other_profile["id"] == "offer_2"

    monkeypatch.setattr(in_app_offer, "_mint_memo_key", lambda: "")

    assert in_app_offer.pending_offer() == OFFER


def test_clear_forgets_this_profiles_offer():
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    in_app_offer.clear()

    assert in_app_offer.pending_offer() is None


def test_signing_out_withdraws_the_offer(_signed_in):
    """However the login went (``hermes logout``, the desktop's disconnect, a
    quarantined token), an offer evaluated for that account stops answering."""
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    assert in_app_offer.pending_offer() == OFFER

    del _signed_in["nous"]

    assert in_app_offer.pending_offer() is None
    # And it is forgotten, not just hidden: signing back in does not revive it.
    _signed_in["nous"] = {"access_token": "account-token", "auth_method": "oauth"}
    assert in_app_offer.pending_offer() is None


def test_a_guest_without_an_account_never_sees_an_offer(_signed_in, monkeypatch):
    from hermes_cli import anon_auth

    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    monkeypatch.setattr(anon_auth, "guest_without_free_tier", lambda state: True)

    assert in_app_offer.pending_offer() is None


def test_refresh_returns_the_offer_the_fetch_recorded(monkeypatch):
    """A successful fetch records through the account normalizer on its way
    back; the refresh then reads the memo."""
    from hermes_cli import nous_account

    def fake_fetch():
        in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
        return object()

    monkeypatch.setattr(nous_account, "stored_token_account_info", fake_fetch)

    assert in_app_offer.refresh_from_portal() == OFFER


def test_refresh_clears_the_offer_when_signed_out(_signed_in, monkeypatch):
    from hermes_cli import nous_account

    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    del _signed_in["nous"]
    monkeypatch.setattr(nous_account, "stored_token_account_info", lambda: None)

    assert in_app_offer.refresh_from_portal() is None
    assert in_app_offer.pending_offer() is None


def test_refresh_keeps_the_offer_when_the_token_is_too_close_to_expiry(monkeypatch):
    """Signed in, but the stored token expires inside the read's skew, so no
    fetch happened (this read never refreshes the token). That is no new
    answer, not a withdrawal: an idle user's token sits exactly here."""
    from hermes_cli import nous_account

    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    monkeypatch.setattr(nous_account, "stored_token_account_info", lambda: None)

    assert in_app_offer.refresh_from_portal() == OFFER
    assert in_app_offer.pending_offer() == OFFER


def test_refresh_keeps_the_last_answer_when_the_fetch_fails(monkeypatch):
    """A transient portal error must not withdraw an offer the user is about
    to see."""
    from hermes_cli import nous_account

    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    def boom():
        raise OSError("portal unreachable")

    monkeypatch.setattr(nous_account, "stored_token_account_info", boom)

    assert in_app_offer.refresh_from_portal() == OFFER


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def _posts(monkeypatch):
    """Capture the report POSTs instead of sending them."""
    sent = []

    def fake_urlopen(req, timeout):
        sent.append({"url": req.full_url, "method": req.get_method(), "body": req.data,
                     "auth": req.get_header("Authorization"), "timeout": timeout})
        return _Resp(204)

    monkeypatch.setattr(in_app_offer.urllib.request, "urlopen", fake_urlopen)
    return sent


def _live_token(state, seconds=3600):
    """The default fixture's token carries no expiry, which reads as expiring; a report needs a
    token that outlives the 60s skew, as a real login has."""
    expires = datetime.now(UTC) + timedelta(seconds=seconds)
    state["nous"] = {**state["nous"], "expires_at": expires.isoformat()}


def test_report_posts_the_served_offer_with_the_stored_token(_signed_in, _posts):
    _live_token(_signed_in)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    assert in_app_offer.report_event("offer_1", "shown") is True

    assert len(_posts) == 1
    post = _posts[0]
    assert post["method"] == "POST"
    assert post["url"].endswith("/api/oauth/account/in-app-offer-event")
    assert post["auth"] == "Bearer account-token"
    assert json.loads(post["body"]) == {"offer_id": "offer_1", "event": "shown"}


def test_report_refuses_an_offer_this_profile_was_not_served(_signed_in, _posts):
    """The RPC carries a client-chosen id: it must not bump an arbitrary campaign."""
    _live_token(_signed_in)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    assert in_app_offer.report_event("someone_elses_offer", "clicked") is False
    assert _posts == []


def test_report_refuses_an_unknown_event(_signed_in, _posts):
    _live_token(_signed_in)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    assert in_app_offer.report_event("offer_1", "viewed") is False
    assert _posts == []


def test_report_sends_nothing_when_signed_out(_signed_in, _posts):
    _live_token(_signed_in)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    _signed_in["nous"] = None

    assert in_app_offer.report_event("offer_1", "shown") is False
    assert _posts == []


def test_report_sends_nothing_with_a_token_about_to_expire(_signed_in, _posts):
    """Never a refresh: a token inside the 60s skew means no report, not a token write."""
    _live_token(_signed_in, seconds=30)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    assert in_app_offer.report_event("offer_1", "shown") is False
    assert _posts == []


def test_report_never_raises_on_a_network_failure(_signed_in, monkeypatch):
    _live_token(_signed_in)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    def boom(req, timeout):
        raise OSError("connection reset")

    monkeypatch.setattr(in_app_offer.urllib.request, "urlopen", boom)

    assert in_app_offer.report_event("offer_1", "shown") is False


def test_another_account_on_the_profile_never_sees_the_offer(_signed_in):
    """A free account's offer must not reach a paid account that logs in on the same profile."""
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    _signed_in["nous"] = {"access_token": "another-accounts-token", "auth_method": "oauth"}

    assert in_app_offer.pending_offer() is None


def test_a_token_refresh_for_the_same_account_keeps_the_offer(_signed_in, monkeypatch):
    """The identity is the JWT's user and org, so a refreshed token is the same account."""
    from hermes_cli import auth_constants

    monkeypatch.setattr(auth_constants, "_decode_jwt_claims", lambda token: {"sub": "u1", "org_id": "o1"})
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    _signed_in["nous"] = {"access_token": "refreshed-token", "auth_method": "oauth"}

    assert in_app_offer.pending_offer() == OFFER


def test_an_offer_not_reconfirmed_for_a_day_is_dropped(monkeypatch):
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})
    later = in_app_offer.time.time() + in_app_offer.MEMO_TTL_SECONDS + 1
    monkeypatch.setattr(in_app_offer.time, "time", lambda: later)

    assert in_app_offer.pending_offer() is None


@pytest.mark.parametrize(
    "url",
    ["http://portal.nousresearch.com/x", "file:///etc/passwd", "/Users/me/secret", "javascript:alert(1)", "https://"],
)
def test_a_non_https_cta_never_lands(url):
    """The desktop's external opener also opens local paths and file: URLs."""
    in_app_offer.record_from_account_payload({"in_app_offer": {**OFFER, "cta_url": url}})

    assert in_app_offer.pending_offer() is None


def test_a_guest_with_a_lapsed_access_token_reports_with_its_anon_credential(_signed_in, _posts, monkeypatch):
    """A guest's access token lives 15 minutes; the dialog opens long after the last fetch."""
    from hermes_cli import anon_auth

    _signed_in["nous"] = {"auth_method": anon_auth.ANON_AUTH_METHOD, "anon_token": "anon_abc"}
    monkeypatch.setattr(anon_auth, "guest_without_free_tier", lambda state: False)
    in_app_offer.record_from_account_payload({"in_app_offer": dict(OFFER)})

    assert in_app_offer.report_event("offer_1", "shown") is True
    assert _posts[0]["auth"] == "Bearer anon_abc"

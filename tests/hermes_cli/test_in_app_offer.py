"""``hermes_cli/in_app_offer.py``: the per-profile memo of the Portal's
``in_app_offer`` campaign answer, as served by the account payload."""

from __future__ import annotations

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

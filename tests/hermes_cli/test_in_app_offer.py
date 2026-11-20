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

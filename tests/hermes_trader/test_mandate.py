"""Tests for hermes_trader.risk.mandate."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hermes_trader.risk.mandate import (
    Mandate,
    load_mandate,
    save_mandate,
    sign_mandate,
    validate_mandate,
)

TEST_KEY = b"test-mandate-secret-key"


@pytest.fixture
def wallet():
    return "0xAbCdEf1234567890abcdef1234567890AbCdEf12"


def _signed(wallet: str, **kwargs) -> Mandate:
    return sign_mandate(wallet, signing_key=TEST_KEY, **kwargs)


def test_sign_and_validate_round_trip(wallet):
    mandate = _signed(wallet)
    ok, err = validate_mandate(mandate, expected_wallet=wallet, signing_key=TEST_KEY)
    assert ok, err


def test_validate_rejects_tampered_signature(wallet):
    mandate = _signed(wallet)
    tampered = Mandate(
        version=mandate.version,
        wallet_address=mandate.wallet_address,
        signed_at=mandate.signed_at,
        expires_at=mandate.expires_at,
        signature="0" * 64,
    )
    ok, err = validate_mandate(tampered, signing_key=TEST_KEY)
    assert not ok
    assert "signature" in err


def test_validate_rejects_wallet_mismatch(wallet):
    mandate = _signed(wallet)
    ok, err = validate_mandate(
        mandate,
        expected_wallet="0x0000000000000000000000000000000000000001",
        signing_key=TEST_KEY,
    )
    assert not ok
    assert "wallet" in err


def test_validate_rejects_expired_mandate(wallet):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mandate = _signed(wallet, expires_at=past)
    ok, err = validate_mandate(mandate, signing_key=TEST_KEY, now=datetime.now(timezone.utc))
    assert not ok
    assert "expired" in err


def test_load_and_save_mandate(tmp_path, wallet):
    path = tmp_path / "mandate.json"
    mandate = _signed(wallet)
    save_mandate(mandate, path)
    loaded = load_mandate(path)
    assert loaded is not None
    ok, err = validate_mandate(loaded, expected_wallet=wallet, signing_key=TEST_KEY)
    assert ok, err


def test_load_missing_returns_none(tmp_path):
    assert load_mandate(tmp_path / "missing.json") is None


def test_sign_requires_key(wallet, monkeypatch):
    monkeypatch.delenv("HERMES_TRADER_MANDATE_SECRET", raising=False)
    monkeypatch.delenv("USER_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="Cannot sign mandate"):
        sign_mandate(wallet)


def test_mandate_json_round_trip(wallet):
    mandate = _signed(wallet)
    data = json.loads(json.dumps(mandate.to_dict()))
    restored = Mandate.from_mapping(data)
    ok, err = validate_mandate(restored, signing_key=TEST_KEY)
    assert ok, err
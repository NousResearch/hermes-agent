"""Tests for Telegram Mini App ``initData`` HMAC verification."""

import hashlib
import hmac
import json
import time
from urllib.parse import quote

from plugins.dashboard_auth.telegram_miniapp.initdata import (
    extract_user_id,
    verify_init_data,
)

BOT_TOKEN = "123456:AAFakeTestTokenNotReal-abcdefghijklmno"


def _build_init_data(fields: dict, *, bot_token: str = BOT_TOKEN) -> str:
    """Build a correctly-signed initData string for *fields* (excluding hash)."""
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    all_fields = {**fields, "hash": computed_hash}
    return "&".join(f"{k}={quote(str(v), safe='')}" for k, v in all_fields.items())


def _valid_fields(user_id: int = 42, auth_date: int | None = None) -> dict:
    if auth_date is None:
        auth_date = int(time.time())
    return {
        "auth_date": str(auth_date),
        "query_id": "AAFakeQueryId",
        # Real Telegram clients always send `signature` (ed25519, for
        # third-party validation) alongside `hash`. Include it in the
        # default shape so every test exercises what production actually
        # receives -- an earlier version of this suite omitted it, which
        # let a check-string bug (excluding signature from the HMAC) pass
        # every test while failing every real client.
        "signature": "fake-ed25519-signature-value",
        "user": json.dumps({"id": user_id, "first_name": "Test", "username": "testuser"}),
    }


# --------------------------------------------------------------------------
# verify_init_data
# --------------------------------------------------------------------------


def test_valid_init_data_verifies():
    init_data = _build_init_data(_valid_fields())
    fields = verify_init_data(init_data, bot_token=BOT_TOKEN)
    assert fields is not None
    assert "hash" in fields


def test_empty_init_data_rejected():
    assert verify_init_data("", bot_token=BOT_TOKEN) is None


def test_empty_bot_token_rejected():
    init_data = _build_init_data(_valid_fields())
    assert verify_init_data(init_data, bot_token="") is None


def test_missing_hash_rejected():
    fields = _valid_fields()
    raw = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in fields.items())
    assert verify_init_data(raw, bot_token=BOT_TOKEN) is None


def test_tampered_field_rejected():
    init_data = _build_init_data(_valid_fields())
    # Flip a signed field's value after signing — hash no longer matches.
    tampered = init_data.replace("query_id=AAFakeQueryId", "query_id=Tampered")
    assert tampered != init_data
    assert verify_init_data(tampered, bot_token=BOT_TOKEN) is None


def test_wrong_bot_token_rejected():
    init_data = _build_init_data(_valid_fields())
    assert verify_init_data(init_data, bot_token="999999:WrongToken") is None


def test_malformed_query_string_rejected():
    assert verify_init_data("not a valid ; query ==", bot_token=BOT_TOKEN) is None


def test_missing_auth_date_rejected():
    fields = _valid_fields()
    del fields["auth_date"]
    init_data = _build_init_data(fields)
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is None


def test_malformed_auth_date_rejected():
    fields = _valid_fields()
    fields["auth_date"] = "not-a-number"
    init_data = _build_init_data(fields)
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is None


def test_expired_auth_date_rejected():
    old_auth_date = int(time.time()) - 7200  # 2h ago, past the 60min default
    init_data = _build_init_data(_valid_fields(auth_date=old_auth_date))
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is None


def test_default_max_age_is_60_minutes():
    """Pins the default window. NOT tight-per-request: a Mini App reuses one
    initData for its whole open session (Telegram gives no way to refresh it
    mid-session), so the window is a session-lifetime bound. An earlier 60s
    default broke every call ~a minute after the app opened; a later 24h
    default was judged too loose. 60 minutes is a deliberate middle ground --
    long enough for a normal session, meaningfully tighter than a full day.
    TLS is the real transport protection; this window is defense-in-depth.
    """
    from plugins.dashboard_auth.telegram_miniapp.initdata import DEFAULT_MAX_AGE_SECONDS

    assert DEFAULT_MAX_AGE_SECONDS == 3600


def test_default_window_accepts_45_minutes_old():
    # A realistic mid-session reuse: the app has been open 45 minutes and is
    # still making calls with the same initData. Must be accepted.
    init_data = _build_init_data(_valid_fields(auth_date=int(time.time()) - 2700))
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is not None


def test_default_window_rejects_2_hours_old():
    init_data = _build_init_data(_valid_fields(auth_date=int(time.time()) - 7200))
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is None


def test_future_auth_date_rejected():
    # Beyond the 60min window in the FUTURE direction -- an auth_date this
    # far ahead is implausible clock skew a real client never sends, so it's
    # rejected rather than trusted.
    future_auth_date = int(time.time()) + 7200
    init_data = _build_init_data(_valid_fields(auth_date=future_auth_date))
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is None


def test_auth_date_within_custom_window_accepted():
    auth_date = int(time.time()) - 100
    init_data = _build_init_data(_valid_fields(auth_date=auth_date))
    assert verify_init_data(init_data, bot_token=BOT_TOKEN, max_age_seconds=200) is not None


def test_auth_date_outside_custom_window_rejected():
    auth_date = int(time.time()) - 250
    init_data = _build_init_data(_valid_fields(auth_date=auth_date))
    assert verify_init_data(init_data, bot_token=BOT_TOKEN, max_age_seconds=200) is None


def test_signature_field_is_part_of_check_string():
    """`signature` (the newer ed25519 third-party-validation field) is a
    SIGNED field like any other -- only `hash` itself is excluded from the
    data-check-string, per Telegram's docs ("a chain of all received
    fields") and confirmed against live initData from real Android and
    Windows Telegram clients.

    This test asserts the INVERSE of what an earlier version did: back
    then _data_check_string excluded `signature` too, so a signature
    appended AFTER signing still verified -- and that behavior is exactly
    the bug that made every real Telegram client's initData fail HMAC
    verification on the live deployment (real clients always send
    `signature`; the synthetic test tokens never did, so the suite stayed
    green while production was broken). Both directions pinned:
    """
    # (a) a signature INCLUDED in the signing (as real Telegram clients do)
    # verifies fine...
    fields_with_sig = {**_valid_fields(), "signature": "some-ed25519-sig"}
    init_data = _build_init_data(fields_with_sig)
    assert verify_init_data(init_data, bot_token=BOT_TOKEN) is not None

    # (b) ...and a signature appended AFTER signing (i.e. not covered by the
    # hash) breaks verification, because it IS part of the check string.
    init_data_tacked_on = _build_init_data(_valid_fields()) + "&signature=" + quote("some-ed25519-sig")
    assert verify_init_data(init_data_tacked_on, bot_token=BOT_TOKEN) is None


# --------------------------------------------------------------------------
# extract_user_id
# --------------------------------------------------------------------------


def test_extract_user_id_from_valid_fields():
    init_data = _build_init_data(_valid_fields(user_id=777))
    fields = verify_init_data(init_data, bot_token=BOT_TOKEN)
    assert extract_user_id(fields) == "777"


def test_extract_user_id_missing_user_field():
    assert extract_user_id({"auth_date": "123"}) is None


def test_extract_user_id_malformed_json():
    assert extract_user_id({"user": "{not json"}) is None


def test_extract_user_id_user_not_a_dict():
    assert extract_user_id({"user": json.dumps([1, 2, 3])}) is None


def test_extract_user_id_missing_id_key():
    assert extract_user_id({"user": json.dumps({"first_name": "Test"})}) is None

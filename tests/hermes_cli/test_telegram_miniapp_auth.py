import hashlib
import hmac
import json
import time
from urllib.parse import quote

import pytest

from hermes_cli.telegram_miniapp.auth import InitDataAuthError, verify_init_data


BOT_TOKEN = "123456:test-token"
USER = {"id": 777, "first_name": "Andrey", "username": "daiver"}


def build_init_data(*, bot_token=BOT_TOKEN, user=USER, auth_date=None, extra=None, hash_value=None):
    auth_date = int(time.time()) if auth_date is None else auth_date
    fields = {
        "auth_date": str(auth_date),
        "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
        "user": json.dumps(user, separators=(",", ":")),
    }
    if extra:
        fields.update(extra)
    data_check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    fields["hash"] = expected if hash_value is None else hash_value
    return "&".join(f"{quote(key)}={quote(value)}" for key, value in fields.items())


def build_init_data_from_fields(fields, *, bot_token=BOT_TOKEN):
    data_check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    signed = dict(fields)
    signed["hash"] = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    return "&".join(f"{quote(key)}={quote(value)}" for key, value in signed.items())


def test_verify_init_data_accepts_valid_allowlisted_user():
    result = verify_init_data(
        build_init_data(auth_date=1_700_000_000),
        bot_token=BOT_TOKEN,
        allowed_users={"777"},
        now=1_700_000_100,
    )

    assert result.user.id == "777"
    assert result.user.username == "daiver"
    assert result.fingerprint
    assert result.raw_init_data is None


@pytest.mark.parametrize(
    "init_data",
    [
        "",
        "auth_date=1700000000&user={}",
        build_init_data(hash_value="not-hex"),
        build_init_data(hash_value="a" * 63),
        build_init_data(hash_value="0" * 64),
        build_init_data(auth_date="not-int"),
        build_init_data(user={"first_name": "No id"}),
        build_init_data(user={"id": []}),
        "auth_date=1700000000&bad=%ZZ&hash=" + "a" * 64,
    ],
)
def test_verify_init_data_rejects_invalid_payloads(init_data):
    with pytest.raises(InitDataAuthError):
        verify_init_data(init_data, bot_token=BOT_TOKEN, allowed_users={"777"}, now=1_700_000_100)


def test_verify_init_data_rejects_duplicate_keys():
    init_data = build_init_data(auth_date=1_700_000_000) + "&auth_date=1700000000"

    with pytest.raises(InitDataAuthError):
        verify_init_data(init_data, bot_token=BOT_TOKEN, allowed_users={"777"}, now=1_700_000_100)


def test_verify_init_data_rejects_signed_missing_or_malformed_user():
    with pytest.raises(InitDataAuthError):
        verify_init_data(
            build_init_data_from_fields({"auth_date": "1700000000"}),
            bot_token=BOT_TOKEN,
            allowed_users={"777"},
            now=1_700_000_100,
        )

    with pytest.raises(InitDataAuthError):
        verify_init_data(
            build_init_data_from_fields({"auth_date": "1700000000", "user": "{"}),
            bot_token=BOT_TOKEN,
            allowed_users={"777"},
            now=1_700_000_100,
        )


def test_verify_init_data_rejects_expired_and_future_auth_date():
    with pytest.raises(InitDataAuthError):
        verify_init_data(
            build_init_data(auth_date=1_700_000_000),
            bot_token=BOT_TOKEN,
            allowed_users={"777"},
            now=1_700_001_000,
            ttl_seconds=300,
        )

    with pytest.raises(InitDataAuthError):
        verify_init_data(
            build_init_data(auth_date=1_700_000_500),
            bot_token=BOT_TOKEN,
            allowed_users={"777"},
            now=1_700_000_000,
            future_skew_seconds=60,
        )


def test_verify_init_data_fails_closed_for_empty_or_non_matching_allowlist():
    valid = build_init_data(auth_date=1_700_000_000)

    with pytest.raises(InitDataAuthError):
        verify_init_data(valid, bot_token=BOT_TOKEN, allowed_users=set(), now=1_700_000_100)

    with pytest.raises(InitDataAuthError):
        verify_init_data(valid, bot_token=BOT_TOKEN, allowed_users={"999"}, now=1_700_000_100)


def test_verify_init_data_does_not_expose_raw_token_or_init_data_in_error():
    init_data = build_init_data(bot_token=BOT_TOKEN, auth_date=1_700_000_000)

    with pytest.raises(InitDataAuthError) as exc:
        verify_init_data(init_data, bot_token="wrong-token", allowed_users={"777"}, now=1_700_000_100)

    message = str(exc.value)
    assert BOT_TOKEN not in message
    assert init_data not in message

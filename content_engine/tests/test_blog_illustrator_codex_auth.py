"""Tests for the Codex CLI auth payload builder (blog_illustrator).

Covers the "missing field id_token" CLI regression: pool entries carry only
access/refresh, so the payload builder must fetch a full token set via the
refresher and never drop identity claims that are present.
"""

from types import SimpleNamespace

from blog.blog_illustrator import (
    _account_id_from_id_token,
    _codex_auth_payload,
    _output_shows_auth_failure,
)


def _entry(**overrides):
    base = {
        "id": "oauth-3",
        "access_token": "access-abc",
        "refresh_token": "refresh-def",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_payload_prefers_refreshed_tokens_even_when_entry_looks_complete():
    """A complete-looking entry still routes through the refresher.

    Stored grants are single-use rotated by other consumers (Codex CLI,
    parallel Hermes processes), so "looks fine" is not a safe skip: the
    refresh path is the single place that can self-heal via
    ~/.codex/auth.json adoption before the CLI burns a dead grant.
    """
    entry = _entry(id_token="id-xyz", account_id="acct-1")
    calls = []

    def refresher(_entry):
        calls.append(_entry)
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "id_token": "fresh-id",
            "account_id": "acct-1",
        }

    payload = _codex_auth_payload(entry, refresher=refresher)
    assert payload["tokens"]["access_token"] == "fresh-access"
    assert payload["tokens"]["refresh_token"] == "fresh-refresh"
    assert payload["tokens"]["id_token"] == "fresh-id"
    assert calls == [entry]


def test_payload_refreshes_when_access_token_expiring(monkeypatch):
    """id_token present but access token expired must still refresh.

    Regression: handing the CLI an expired access token made it attempt its
    own refresh with an already-consumed rotation -> 401 'refresh token was
    already used'.
    """
    import hermes_cli.auth as A

    monkeypatch.setattr(A, "_codex_access_token_is_expiring",
                        lambda token, skew=0: True)
    entry = _entry(id_token="stale-id", account_id="acct-1")

    def refresher(_entry):
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "id_token": "fresh-id",
            "account_id": "acct-1",
        }

    payload = _codex_auth_payload(entry, refresher=refresher)
    assert payload["tokens"]["access_token"] == "fresh-access"
    assert payload["tokens"]["refresh_token"] == "fresh-refresh"
    assert payload["tokens"]["id_token"] == "fresh-id"


def test_payload_refreshes_when_id_token_missing():
    entry = _entry()

    def refresher(_entry):
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "id_token": "fresh-id",
            "account_id": "acct-9",
        }

    payload = _codex_auth_payload(entry, refresher=refresher)
    assert payload["tokens"] == {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
        "id_token": "fresh-id",
        "account_id": "acct-9",
    }


def test_payload_degrades_when_refresh_fails():
    entry = _entry()

    def refresher(_entry):
        return None

    payload = _codex_auth_payload(entry, refresher=refresher)
    # Partial payload: the CLI will fail closed with a clear parse error
    # instead of silently using a wrong account's id_token.
    assert payload == {
        "tokens": {
            "access_token": "access-abc",
            "refresh_token": "refresh-def",
        }
    }


def test_payload_degrades_when_refresher_raises():
    entry = _entry()

    def refresher(_entry):
        raise RuntimeError("network down")

    payload = _codex_auth_payload(entry, refresher=refresher)
    assert payload["tokens"]["access_token"] == "access-abc"
    assert payload["tokens"]["refresh_token"] == "refresh-def"


def test_output_shows_auth_failure_markers():
    dead = ("ERROR: Your access token could not be refreshed because your "
            "refresh token was already used. Please log out and sign in again.")
    assert _output_shows_auth_failure(dead)
    assert _output_shows_auth_failure(
        "failed to connect to websocket: HTTP error: 401 Unauthorized")
    assert not _output_shows_auth_failure("")
    assert not _output_shows_auth_failure(
        "usage limit reached; try again at Aug 24th, 2026 7:45 PM")
    assert not _output_shows_auth_failure("transient 500 from backend")
    assert not _output_shows_auth_failure(None)


def _jwt(claims: dict) -> str:
    """Unsigned JWT with the given claims (the CLI never verifies the sig)."""
    import base64
    import json

    def segment(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{segment({'alg': 'none'})}.{segment(claims)}.sig"


def test_account_id_derived_from_id_token():
    """The CLI needs an explicit account_id; the id_token claim supplies it.

    Without the field the websocket handshake 401s and the CLI misreports the
    session as superseded, masking real usage-limit errors.
    """
    id_token = _jwt(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-42"}}
    )
    assert _account_id_from_id_token(id_token) == "acct-42"
    payload = _codex_auth_payload(_entry(id_token=id_token))
    assert payload["tokens"]["account_id"] == "acct-42"


def test_account_id_derivation_absent_when_claim_missing():
    assert _account_id_from_id_token(_jwt({"sub": "user-1"})) is None
    assert _account_id_from_id_token(None) is None
    assert _account_id_from_id_token("not-a-jwt") is None


def test_account_id_not_overwritten_when_present():
    id_token = _jwt(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-42"}}
    )
    entry = _entry(id_token=id_token, account_id="acct-explicit")
    payload = _codex_auth_payload(entry)
    assert payload["tokens"]["account_id"] == "acct-explicit"

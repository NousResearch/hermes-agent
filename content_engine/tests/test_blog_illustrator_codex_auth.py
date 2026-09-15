"""Tests for the Codex CLI auth payload builder (blog_illustrator).

Covers the "missing field id_token" CLI regression: pool entries carry only
access/refresh, so the payload builder must fetch a full token set via the
refresher and never drop identity claims that are present.
"""

from types import SimpleNamespace

from blog.blog_illustrator import _codex_auth_payload


def _entry(**overrides):
    base = {
        "id": "oauth-3",
        "access_token": "access-abc",
        "refresh_token": "refresh-def",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_payload_uses_entry_fields_when_complete():
    entry = _entry(id_token="id-xyz", account_id="acct-1")
    calls = []

    def refresher(_entry):
        calls.append(_entry)
        return {"access_token": "should-not-be-used"}

    payload = _codex_auth_payload(entry, refresher=refresher)
    assert payload == {
        "tokens": {
            "access_token": "access-abc",
            "refresh_token": "refresh-def",
            "account_id": "acct-1",
            "id_token": "id-xyz",
        }
    }
    assert calls == []


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

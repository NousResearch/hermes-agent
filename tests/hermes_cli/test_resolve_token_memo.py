"""Tests for the resolve_nous_access_token startup-burst memo (PR #66016).

The memo collapses the startup burst of managed-tool check_fn calls into a
single expensive resolution: within the short TTL, repeat calls return the
cached token without re-entering _provider_state_transaction (two
cross-process file locks + state reads) or triggering a network refresh.
"""

import json
import time

import pytest

import hermes_cli.auth as auth


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    monkeypatch.setattr(auth, "_RESOLVE_TOKEN_CACHE", None)
    yield


def _write_valid_auth_file(tmp_path, token="memo-token"):
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "nous",
                "providers": {
                    "nous": {
                        "access_token": token,
                        "refresh_token": "r",
                        "client_id": "hermes-cli-vps",
                        "expires_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + 3600)
                        ),
                    }
                },
            }
        )
    )


def _count_transactions(monkeypatch):
    calls = {"n": 0}
    real = auth._provider_state_transaction

    def _counting(provider):
        calls["n"] += 1
        return real(provider)

    monkeypatch.setattr(auth, "_provider_state_transaction", _counting)
    return calls


def test_repeat_calls_within_ttl_hit_memo(monkeypatch, tmp_path):
    _write_valid_auth_file(tmp_path)
    calls = _count_transactions(monkeypatch)

    first = auth.resolve_nous_access_token()
    second = auth.resolve_nous_access_token()
    third = auth.resolve_nous_access_token()

    assert first == second == third == "memo-token"
    assert calls["n"] == 1, (
        "repeat calls within the TTL must not re-enter the state transaction"
    )


def test_memo_expires_after_ttl(monkeypatch, tmp_path):
    _write_valid_auth_file(tmp_path)
    calls = _count_transactions(monkeypatch)

    auth.resolve_nous_access_token()
    cached_at, tok = auth._RESOLVE_TOKEN_CACHE
    monkeypatch.setattr(
        auth,
        "_RESOLVE_TOKEN_CACHE",
        (cached_at - auth._RESOLVE_TOKEN_CACHE_TTL_S - 1.0, tok),
    )
    auth.resolve_nous_access_token()

    assert calls["n"] == 2, "an expired memo must re-resolve"


def test_insecure_callers_bypass_memo(monkeypatch, tmp_path):
    _write_valid_auth_file(tmp_path)
    calls = _count_transactions(monkeypatch)

    auth.resolve_nous_access_token()
    auth.resolve_nous_access_token(insecure=True)

    assert calls["n"] == 2, "insecure callers must bypass the memo entirely"

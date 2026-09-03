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
    monkeypatch.setattr(auth, "_RESOLVE_TOKEN_CACHE", {})
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
    cache_key = str(auth.get_hermes_home())
    cached_at, tok = auth._RESOLVE_TOKEN_CACHE[cache_key]
    monkeypatch.setattr(
        auth,
        "_RESOLVE_TOKEN_CACHE",
        {cache_key: (cached_at - auth._RESOLVE_TOKEN_CACHE_TTL_S - 1.0, tok)},
    )
    auth.resolve_nous_access_token()

    assert calls["n"] == 2, "an expired memo must re-resolve"


def test_insecure_callers_bypass_memo(monkeypatch, tmp_path):
    _write_valid_auth_file(tmp_path)
    calls = _count_transactions(monkeypatch)

    auth.resolve_nous_access_token()
    auth.resolve_nous_access_token(insecure=True)

    assert calls["n"] == 2, "insecure callers must bypass the memo entirely"


def test_memo_does_not_leak_across_multiplex_profile_contexts(tmp_path):
    """A multiplex gateway scopes each profile's context via
    hermes_constants.set_hermes_home_override (gateway/run.py,
    tui_gateway/server.py), not the HERMES_HOME env var — the memo must key
    on that same resolved home, or one profile's context can read another
    profile's already-cached Nous access token for up to the TTL window.
    """
    import hermes_constants

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    _write_valid_auth_file(profile_a, token="token-a")
    _write_valid_auth_file(profile_b, token="token-b")

    token_a = token_b = None
    reset_token = hermes_constants.set_hermes_home_override(str(profile_a))
    try:
        token_a = auth.resolve_nous_access_token()
    finally:
        hermes_constants.reset_hermes_home_override(reset_token)

    # Still well within the 5s TTL — this is exactly the race window: profile
    # B's context calls resolve_nous_access_token() shortly after profile A's.
    reset_token = hermes_constants.set_hermes_home_override(str(profile_b))
    try:
        token_b = auth.resolve_nous_access_token()
    finally:
        hermes_constants.reset_hermes_home_override(reset_token)

    assert token_a == "token-a"
    assert token_b == "token-b", (
        "profile B's context must not receive profile A's cached access token"
    )

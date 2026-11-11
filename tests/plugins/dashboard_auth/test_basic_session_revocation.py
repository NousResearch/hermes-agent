"""BasicAuthProvider sessions end on logout and on a credential change (#77186).

Tokens are stateless, so before the fix a copied token outlived logout for its whole TTL and a
password rotation with the same explicit ``secret`` left every old session valid.
"""

from __future__ import annotations

import secrets

import pytest

from hermes_cli.dashboard_auth import RefreshExpiredError
from plugins.dashboard_auth import basic

SECRET = secrets.token_bytes(32)


def _provider(password: str = "old-pw", **kw) -> basic.BasicAuthProvider:
    return basic.BasicAuthProvider(username="admin", password_hash=basic.hash_password(password), secret=SECRET, **kw)


def test_logout_ends_the_login_and_its_renewals_in_this_and_a_restarted_process():
    p = _provider()
    s = p.complete_password_login(username="admin", password="old-pw")
    renewed = p.refresh_session(refresh_token=s.refresh_token)
    other = p.complete_password_login(username="admin", password="old-pw")

    p.revoke_session(refresh_token=s.refresh_token)

    restarted = basic.BasicAuthProvider(username="admin", password_hash=p._password_hash, secret=SECRET)
    for provider in (p, restarted):
        assert provider.verify_session(access_token=s.access_token) is None
        assert provider.verify_session(access_token=renewed.access_token) is None
        for rt in (s.refresh_token, renewed.refresh_token):
            with pytest.raises(RefreshExpiredError):
                provider.refresh_session(refresh_token=rt)
        # Another login of the same user is not signed out.
        assert provider.verify_session(access_token=other.access_token) is not None


def test_revoking_a_foreign_or_garbage_token_does_not_raise():
    p = _provider()
    s = p.complete_password_login(username="admin", password="old-pw")
    p.revoke_session(refresh_token="garbage")
    p.revoke_session(refresh_token="")
    assert p.verify_session(access_token=s.access_token) is not None


@pytest.mark.parametrize("configured", ["plaintext", "hash"])
def test_rotating_the_password_with_the_same_secret_ends_old_sessions(monkeypatch, configured):
    monkeypatch.setattr(basic, "_load_config_basic_auth_section", dict)
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "admin")
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", SECRET.hex())

    def start(password):
        if configured == "plaintext":
            monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", password)
        else:
            monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH", basic.hash_password(password))
        return basic.BasicAuthProvider(**basic._settings())

    old = start("old-pw")
    s = old.complete_password_login(username="admin", password="old-pw")
    if configured == "plaintext":
        # A restart with the unchanged plaintext (re-hashed with a new salt) keeps the session.
        assert start("old-pw").verify_session(access_token=s.access_token) is not None

    rotated = start("new-pw")
    assert rotated.verify_session(access_token=s.access_token) is None
    with pytest.raises(RefreshExpiredError):
        rotated.refresh_session(refresh_token=s.refresh_token)
    fresh = rotated.complete_password_login(username="admin", password="new-pw")
    assert rotated.verify_session(access_token=fresh.access_token) is not None

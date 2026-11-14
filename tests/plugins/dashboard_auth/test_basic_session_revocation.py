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


def test_logout_outlasts_a_session_ttl_longer_than_the_refresh_ttl(monkeypatch):
    now = [1_000_000_000.0]
    monkeypatch.setattr(basic.time, "time", lambda: now[0])
    day = 24 * 60 * 60
    p = _provider(ttl_seconds=60 * day)
    s = p.complete_password_login(username="admin", password="old-pw")
    now[0] += day
    p.revoke_session(refresh_token=s.refresh_token)

    now[0] += 45 * day  # past the refresh TTL, inside the access token's lifetime
    assert p.verify_session(access_token=s.access_token) is None


def test_a_worker_sees_a_logout_whose_file_kept_the_old_mtime():
    """Coarse-timestamp filesystems: a replacement can leave ``st_mtime`` unchanged."""
    import os

    worker_a = _provider()
    worker_b = basic.BasicAuthProvider(username="admin", password_hash=worker_a._password_hash, secret=SECRET)
    first = worker_a.complete_password_login(username="admin", password="old-pw")
    s = worker_a.complete_password_login(username="admin", password="old-pw")
    worker_b.revoke_session(refresh_token=first.refresh_token)
    path = worker_b._revoked_path
    before = path.stat()
    assert worker_a.verify_session(access_token=s.access_token) is not None  # A has read the file

    worker_b.revoke_session(refresh_token=s.refresh_token)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert worker_a.verify_session(access_token=s.access_token) is None
    # A's own next logout must not write B's revocation away.
    worker_a.revoke_session(refresh_token=worker_a.complete_password_login(
        username="admin", password="old-pw").refresh_token)
    assert worker_b.verify_session(access_token=s.access_token) is None


def test_a_logout_keeps_the_revocations_other_workers_wrote_since_this_one_last_read():
    worker_a = _provider()
    worker_b = basic.BasicAuthProvider(username="admin", password_hash=worker_a._password_hash, secret=SECRET)
    first = worker_a.complete_password_login(username="admin", password="old-pw")
    second = worker_a.complete_password_login(username="admin", password="old-pw")
    worker_b.revoke_session(refresh_token=first.refresh_token)
    worker_a.revoke_session(refresh_token=second.refresh_token)  # A has not read the file since B wrote it

    restarted = basic.BasicAuthProvider(username="admin", password_hash=worker_a._password_hash, secret=SECRET)
    assert restarted.verify_session(access_token=first.access_token) is None
    assert restarted.verify_session(access_token=second.access_token) is None


def test_access_token_revocation_ends_its_login_and_ignores_other_tokens(monkeypatch):
    """Past day 30 a long-TTL access token has outlived its refresh token (and cookie), so the
    access token is all a logout can present; the configured lifetime is kept."""
    now = [1_000_000_000.0]
    monkeypatch.setattr(basic.time, "time", lambda: now[0])
    day = 24 * 60 * 60
    p = _provider(ttl_seconds=60 * day)
    s = p.complete_password_login(username="admin", password="old-pw")
    other = p.complete_password_login(username="admin", password="old-pw")
    now[0] += 31 * day
    assert p.verify_session(access_token=s.access_token) is not None  # the configured 60 days hold
    p.revoke_access_session(access_token=s.refresh_token)  # a refresh token is not an access token
    p.revoke_access_session(access_token="garbage")
    assert p.verify_session(access_token=s.access_token) is not None

    p.revoke_access_session(access_token=s.access_token)
    now[0] += 28 * day
    assert p.verify_session(access_token=s.access_token) is None
    assert p.verify_session(access_token=other.access_token) is not None


@pytest.fixture
def gated_basic():
    from fastapi.testclient import TestClient

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth import refresh_singleflight

    provider = _provider()
    clear_providers()
    register_provider(provider)
    state = web_server.app.state
    prev = {k: getattr(state, k, None) for k in ("bound_host", "bound_port", "auth_required")}
    state.bound_host, state.bound_port, state.auth_required = "fly-app.fly.dev", 443, True
    yield provider, TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    clear_providers()
    with refresh_singleflight._guard:
        refresh_singleflight._cache.clear()
    for k, v in prev.items():
        setattr(state, k, v)


def test_logout_ends_a_refresh_the_gate_cached_just_before_it(gated_basic):
    """A copied RT replayed within the refresh replay window must not get the cached rotation of a
    login that has since signed out."""
    provider, client = gated_basic
    s = provider.complete_password_login(username="admin", password="old-pw")
    copied = {"hermes_session_at": "lapsed", "hermes_session_rt": s.refresh_token,
              "hermes_session_provider": "basic"}

    def me():
        client.cookies.clear()
        client.cookies.update(copied)
        return client.get("/api/auth/me")

    assert me().status_code == 200  # the gate rotated the copied RT and cached the result
    client.cookies.clear()
    client.cookies.update(copied)
    out = client.post("/auth/logout", headers={"Origin": "https://fly-app.fly.dev"}, follow_redirects=False)
    assert out.status_code == 302, out.text

    assert me().status_code == 401


def _cookie_request(client, cookies, method="get", path="/api/auth/me", **kw):
    client.cookies.clear()
    client.cookies.update({"hermes_session_provider": "basic", **cookies})
    return getattr(client, method)(path, **kw)


def _logout(client, cookies):
    out = _cookie_request(client, cookies, "post", "/auth/logout",
                          headers={"Origin": "https://fly-app.fly.dev"}, follow_redirects=False)
    assert out.status_code == 302, out.text


def test_logout_with_the_browsers_current_rt_ends_a_copied_rts_cached_replay(gated_basic):
    """The browser rotated outside the replay window, so the copied RT's cached rotation shares no
    token with the RT the logout presents; it shares only the login."""
    provider, client = gated_basic
    s = provider.complete_password_login(username="admin", password="old-pw")
    browser_rt = provider.refresh_session(refresh_token=s.refresh_token).refresh_token
    copied = {"hermes_session_at": "lapsed", "hermes_session_rt": s.refresh_token}

    assert _cookie_request(client, copied).status_code == 200  # replayed just before logout
    _logout(client, {"hermes_session_at": "lapsed", "hermes_session_rt": browser_rt})

    assert _cookie_request(client, copied).status_code == 401


def test_logout_after_the_refresh_cookie_expired_ends_a_long_lived_access_token(gated_basic, monkeypatch):
    provider, client = gated_basic
    now = [1_000_000_000.0]
    monkeypatch.setattr(basic.time, "time", lambda: now[0])
    day = 24 * 60 * 60
    provider._ttl = 60 * day  # session_ttl_seconds above the 30-day refresh lifetime
    s = provider.complete_password_login(username="admin", password="old-pw")
    now[0] += 31 * day  # the refresh token and its 30-day cookie are gone; only the AT cookie is left
    copied = {"hermes_session_at": s.access_token}
    assert _cookie_request(client, copied).status_code == 200

    _logout(client, copied)

    assert _cookie_request(client, copied).status_code == 401

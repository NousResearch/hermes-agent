from hermes_cli import nous_auth_keepalive as keepalive
from hermes_cli.auth import ACCESS_TOKEN_REFRESH_SKEW_SECONDS

NOUS_ACCESS_TOKEN_TTL_SECONDS = 3600


def test_keepalive_interval_fits_inside_the_token_lifetime():
    """The tick must land before the hour rolls over, or refresh stays reactive.

    A tick at or above TTL - skew can miss the refresh window entirely, which
    is what made every hour expire into a 401 plus a re-auth round trip.
    """
    assert (
        keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
        < NOUS_ACCESS_TOKEN_TTL_SECONDS - ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    )


def test_interval_precedence_and_disable(monkeypatch):
    monkeypatch.delenv(keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_ENV, raising=False)
    assert (
        keepalive._interval_seconds(None)
        == keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    )

    monkeypatch.setenv(keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_ENV, "600")
    assert keepalive._interval_seconds(None) == 600
    # An explicit argument still outranks the environment.
    assert keepalive._interval_seconds(300) == 300

    # A malformed override falls back to the default rather than disabling.
    monkeypatch.setenv(keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_ENV, "not-a-number")
    assert (
        keepalive._interval_seconds(None)
        == keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    )

    # Zero remains the documented way to turn the keepalive off.
    monkeypatch.setenv(keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_ENV, "0")
    assert keepalive._interval_seconds(None) == 0
    assert keepalive.start_nous_auth_keepalive() is None


def test_keepalive_refreshes_stale_pool_entry(monkeypatch):
    class _Entry:
        access_token = "pooled-access-token"
        expires_at = "2000-01-01T00:00:00+00:00"
        agent_key = ""
        agent_key_expires_at = None
        scope = "inference:invoke"

    class _Pool:
        refreshed = False

        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

        def try_refresh_current(self):
            self.refreshed = True
            return _Entry()

    pool = _Pool()
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

    assert keepalive.refresh_nous_auth_keepalive_once() is True
    assert pool.refreshed is True


def test_keepalive_falls_back_to_singleton_state(monkeypatch):
    calls = []

    class _Pool:
        def has_credentials(self):
            return False

    def _resolve_nous_runtime_credentials(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "nous",
            "api_key": "fresh-agent-key",
            "base_url": "https://inference-api.nousresearch.com/v1",
        }

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _Pool())
    monkeypatch.setattr(
        keepalive,
        "get_provider_auth_state",
        lambda provider: {"access_token": "stored-access-token"},
    )
    monkeypatch.setattr(
        keepalive,
        "resolve_nous_runtime_credentials",
        _resolve_nous_runtime_credentials,
    )

    assert keepalive.refresh_nous_auth_keepalive_once(timeout_seconds=15.0) is True
    assert calls == [{"timeout_seconds": 15.0}]

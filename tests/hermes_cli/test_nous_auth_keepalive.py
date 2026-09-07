from hermes_cli import nous_auth_keepalive as keepalive

# Both lifetimes have been observed on real installs.
OBSERVED_LIFETIMES_SECONDS = (3594, 899)


def test_refresh_always_fires_before_expiry_for_observed_lifetimes():
    """Simulate the tick schedule and assert no credential expires unrefreshed.

    This is the property that actually matters: for every lifetime, some tick
    must decide to refresh while the credential is still valid. Ticking faster
    alone does not guarantee it -- the refresh horizon has to cover the gap
    between ticks too.
    """
    for lifetime in OBSERVED_LIFETIMES_SECONDS:
        tick = keepalive._tick_seconds(
            keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS, lifetime
        )
        horizon = keepalive._refresh_horizon_seconds(
            tick, keepalive.NOUS_INVOKE_JWT_MIN_TTL_SECONDS
        )

        # Walk the ticks and find the first one that refreshes.
        refreshed_at = None
        elapsed = 0
        while elapsed <= lifetime:
            if lifetime - elapsed <= horizon:
                refreshed_at = elapsed
                break
            elapsed += tick

        assert refreshed_at is not None, f"never refreshed for lifetime={lifetime}"
        assert refreshed_at < lifetime, (
            f"refresh at {refreshed_at}s came at/after expiry {lifetime}s "
            f"(tick={tick}, horizon={horizon})"
        )


def test_interval_precedence_and_disable(monkeypatch):
    def _config(section):
        monkeypatch.setattr(keepalive, "_nous_config", lambda: section)

    # An absent section leaves the module default in place.
    _config({})
    assert (
        keepalive._interval_seconds(None)
        == keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    )

    _config({keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_CONFIG_KEY: 600})
    assert keepalive._interval_seconds(None) == 600
    # An explicit argument still outranks config.yaml.
    assert keepalive._interval_seconds(300) == 300

    # A malformed value falls back to the default rather than disabling.
    _config({keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_CONFIG_KEY: "not-a-number"})
    assert (
        keepalive._interval_seconds(None)
        == keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    )

    # Zero remains the documented way to turn the keepalive off.
    _config({keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_CONFIG_KEY: 0})
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

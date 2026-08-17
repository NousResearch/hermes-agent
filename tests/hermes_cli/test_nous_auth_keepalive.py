from hermes_cli import nous_auth_keepalive as keepalive
from hermes_cli.auth import ACCESS_TOKEN_REFRESH_SKEW_SECONDS

# Both lifetimes have been observed on real installs.
OBSERVED_LIFETIMES_SECONDS = (3594, 899)


def test_resolved_tick_fits_inside_the_token_lifetime():
    """The tick actually used must land before the credential rolls over.

    A tick at or above TTL - skew can miss the refresh window entirely, which
    is what made every hour expire into a 401 plus a re-auth round trip.

    This asserts on the derived tick rather than the configured constant,
    because the constant is only a ceiling -- the schedule that ships is
    whatever the derivation produces. It therefore fails if the derivation
    constants regress (a lower TICKS_PER_LIFETIME or a higher
    MIN_INTERVAL_SECONDS both break it), which a bare inequality against the
    default interval cannot catch.
    """
    for lifetime in OBSERVED_LIFETIMES_SECONDS:
        tick = keepalive._tick_seconds(
            keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS, lifetime
        )
        assert tick < lifetime - ACCESS_TOKEN_REFRESH_SKEW_SECONDS, (
            f"tick={tick}s leaves no room to refresh inside a {lifetime}s "
            f"lifetime (skew={ACCESS_TOKEN_REFRESH_SKEW_SECONDS}s)"
        )


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


def test_tick_derives_from_observed_lifetime():
    default = keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS

    # A short-lived credential pulls the tick down below the configured default.
    assert keepalive._tick_seconds(default, 899) == 899 // 4

    # A long-lived one is still capped by the configured interval.
    assert keepalive._tick_seconds(default, 86400) == default

    # A missing or nonsensical lifetime leaves the configured tick alone.
    assert keepalive._tick_seconds(default, None) == default
    assert keepalive._tick_seconds(default, 0) == default

    # A pathological lifetime cannot spin the thread.
    assert (
        keepalive._tick_seconds(default, 4)
        == keepalive.NOUS_AUTH_KEEPALIVE_MIN_INTERVAL_SECONDS
    )


def test_refresh_horizon_covers_the_gap_between_ticks():
    # A credential that will not survive until the next tick refreshes now.
    assert keepalive._refresh_horizon_seconds(900, 120) == 900 + (
        ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    )
    # The caller's floor still applies when ticks are very frequent.
    assert keepalive._refresh_horizon_seconds(10, 5000) == 5000


def test_observed_lifetime_takes_the_shorter_credential(monkeypatch):
    monkeypatch.setattr(
        keepalive,
        "get_provider_auth_state",
        lambda provider: {"expires_in": 3594, "agent_key_expires_in": 899},
    )
    assert keepalive._observed_lifetime_seconds() == 899

    monkeypatch.setattr(keepalive, "get_provider_auth_state", lambda provider: {})
    assert keepalive._observed_lifetime_seconds() is None

    monkeypatch.setattr(
        keepalive,
        "get_provider_auth_state",
        lambda provider: {"expires_in": "nonsense"},
    )
    assert keepalive._observed_lifetime_seconds() is None


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


def test_interval_survives_an_unreadable_config(monkeypatch):
    """A broken config.yaml must not take the keepalive thread down with it."""

    def _boom():
        raise RuntimeError("config.yaml is unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    assert keepalive._nous_config() == {}
    assert (
        keepalive._interval_seconds(None)
        == keepalive.NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    )


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

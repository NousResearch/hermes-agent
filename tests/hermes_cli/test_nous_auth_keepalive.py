import threading
from pathlib import Path

from hermes_cli import nous_auth_keepalive as keepalive
from hermes_constants import get_hermes_home

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


# ---------------------------------------------------------------------------
# Multiplex profile scoping (#-, raw threading.Thread does not inherit the
# spawning thread's contextvars, so an unscoped keepalive loop only ever
# resolves get_hermes_home() as the default profile).
# ---------------------------------------------------------------------------


def test_profile_homes_single_profile_gateway_is_unscoped(monkeypatch):
    """No multiplexing -> legacy shape: one entry, home=None (run unscoped)."""

    class _Config:
        multiplex_profiles = False

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _Config())
    monkeypatch.setattr(keepalive, "_active_profile_name", lambda: "default")

    assert keepalive._keepalive_profile_homes() == [("default", None)]


def test_profile_homes_multiplex_returns_every_profile_home(monkeypatch):
    """Multiplexed -> every profile gateway.run._multiplex_profile_homes yields."""

    class _Config:
        multiplex_profiles = True
        multiplex_profile_allowlist = None

    homes = [("default", Path("/h")), ("worker", Path("/h/profiles/worker"))]
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _Config())
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", lambda _cfg: homes)

    assert keepalive._keepalive_profile_homes() == homes


def test_profile_homes_degrades_to_active_profile_when_resolution_raises(monkeypatch):
    """A broken config/profile resolver must not disable the keepalive entirely."""

    def _boom():
        raise RuntimeError("config.yaml unreadable")

    monkeypatch.setattr("gateway.config.load_gateway_config", _boom)
    monkeypatch.setattr(keepalive, "_active_profile_name", lambda: "default")

    assert keepalive._keepalive_profile_homes() == [("default", None)]


def test_tick_one_profile_enters_that_profiles_runtime_scope(
    tmp_path, monkeypatch
):
    """The whole point of the fix: the refresh call sees THAT profile's home.

    Asserting on the observed ``get_hermes_home()`` (not just the call count)
    keeps this mutation-survivable -- dropping the ``with
    _profile_runtime_scope(...)`` still calls ``refresh_nous_auth_keepalive_once``
    once, but it would see the process-default home instead of the profile's.
    """
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    seen = []

    def _fake_refresh(**_kwargs):
        seen.append(get_hermes_home())
        return True

    monkeypatch.setattr(keepalive, "refresh_nous_auth_keepalive_once", _fake_refresh)
    monkeypatch.setattr(keepalive, "_observed_lifetime_seconds", lambda: None)

    tick = keepalive._keepalive_tick_one_profile(
        "worker",
        profile_home,
        interval_seconds=900,
        min_key_ttl_seconds=60,
        timeout_seconds=None,
    )

    assert seen == [profile_home]
    assert tick == 900
    # Scope must not leak past the call.
    assert get_hermes_home() != profile_home


def test_tick_one_profile_none_home_runs_unscoped(monkeypatch):
    """``profile_home=None`` is the legacy single-profile path: no scope entered."""
    ambient_home = get_hermes_home()
    seen = []

    def _fake_refresh(**_kwargs):
        seen.append(get_hermes_home())
        return True

    monkeypatch.setattr(keepalive, "refresh_nous_auth_keepalive_once", _fake_refresh)
    monkeypatch.setattr(keepalive, "_observed_lifetime_seconds", lambda: None)

    keepalive._keepalive_tick_one_profile(
        "default",
        None,
        interval_seconds=900,
        min_key_ttl_seconds=60,
        timeout_seconds=None,
    )

    assert seen == [ambient_home]


class _StopAfterFirstPass:
    """Fake ``threading.Event``: lets ``_keepalive_loop`` run exactly one pass."""

    def __init__(self):
        self._set = False

    def wait(self, _timeout=None):
        self._set = True
        return self._set

    def is_set(self):
        return self._set


def test_keepalive_loop_ticks_every_multiplex_profile_home_per_pass(
    tmp_path, monkeypatch
):
    """End-to-end: one loop pass must refresh every configured profile's own home.

    This is the regression this whole fix exists for: before it, the loop
    called ``refresh_nous_auth_keepalive_once`` exactly once per pass, always
    under whatever home was ambient at thread-spawn time (the default
    profile) -- a secondary profile's independent Nous OAuth login was never
    proactively refreshed.
    """
    homes = [
        ("default", tmp_path / "default"),
        ("worker", tmp_path / "worker"),
    ]
    for _name, home in homes:
        home.mkdir()

    seen = []

    def _fake_refresh(**_kwargs):
        seen.append((get_hermes_home(), threading.current_thread().name))
        return True

    monkeypatch.setattr(keepalive, "_keepalive_profile_homes", lambda: homes)
    monkeypatch.setattr(keepalive, "refresh_nous_auth_keepalive_once", _fake_refresh)
    monkeypatch.setattr(keepalive, "_observed_lifetime_seconds", lambda: None)

    keepalive._keepalive_loop(
        _StopAfterFirstPass(),
        interval_seconds=900,
        initial_delay_seconds=0,
        min_key_ttl_seconds=60,
        timeout_seconds=None,
    )

    assert [home for home, _thread in seen] == [home for _name, home in homes]

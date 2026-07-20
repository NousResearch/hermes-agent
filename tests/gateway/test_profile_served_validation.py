"""``_require_served_profile_home`` — refusing unknown/stale profiles.

``_resolve_profile_home_for_source`` is deliberately lenient: an explicit
profile that does not exist logs a warning and falls back to the global
HERMES_HOME. That is survivable on a single-profile gateway, where the global
home IS the only profile's home.

Under multiplexing it is not. A stale ``profile_routes`` entry, a renamed or
deleted profile, or a source stamped by anything other than our own startup
wiring would be served under the MULTIPLEXER's config, skills, memory and
credentials. ``_require_served_profile_home`` is the validating wrapper used on
the authorization and runtime-entry paths; it raises ``ProfileNotServedError``
where the plain resolver would silently fall back.

These tests drive the REAL resolver underneath the real wrapper — nothing about
resolution itself is mocked out, so route precedence and owner-stamp behavior
are exercised as they ship.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner, ProfileNotServedError
from gateway.session import SessionSource


def _runner(multiplex=True, served=(), routes=(), active="default"):
    """Runner with the REAL resolver + wrapper bound, serving ``served``."""
    runner = object.__new__(GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=multiplex, profile_routes=list(routes))
    runner._profile_adapters = {name: {} for name in served if name != active}
    runner._active_profile_name = lambda: active
    return runner


def _source(profile=None, chat_id="C1", platform=Platform.SLACK, **kw):
    return SessionSource(platform=platform, chat_id=chat_id, profile=profile, **kw)


class TestUnknownProfileRefused:
    def test_unserved_profile_raises(self, tmp_path, monkeypatch):
        """A stamped profile this gateway does not serve must not resolve."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(served=["default", "coder"])

        with pytest.raises(ProfileNotServedError) as exc:
            runner._require_served_profile_home(_source(profile="ghost"))

        assert "ghost" in str(exc.value)

    def test_unserved_profile_does_not_reach_global_home(self, tmp_path, monkeypatch):
        """Contrast with the lenient resolver, on the identical source.

        The plain resolver returns the global HERMES_HOME — that IS the defect.
        The wrapper must not.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(served=["default"])
        source = _source(profile="ghost")

        # The lenient path still behaves as documented...
        assert runner._resolve_profile_home_for_source(source) == Path(str(tmp_path))
        # ...but the validating path refuses instead of entering it.
        with pytest.raises(ProfileNotServedError):
            runner._require_served_profile_home(source)

    def test_served_but_deleted_profile_raises(self, tmp_path, monkeypatch):
        """A profile that is served but no longer on disk is stale."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(served=["default", "coder"])

        with patch("hermes_cli.profiles.profile_exists", return_value=False):
            with pytest.raises(ProfileNotServedError) as exc:
                runner._require_served_profile_home(_source(profile="coder"))

        assert "no longer exists" in str(exc.value)

    def test_stale_route_target_raises(self, tmp_path, monkeypatch):
        """A route pointing at an unserved profile is refused too.

        The profile arrives via routing rather than a stamp, but it is just as
        explicit — and just as unserveable.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        route = ProfileRoute(name="r", platform="slack", chat_id="C1", profile="retired")
        runner = _runner(served=["default"], routes=[route])

        with pytest.raises(ProfileNotServedError) as exc:
            runner._require_served_profile_home(_source(chat_id="C1"))

        assert "retired" in str(exc.value)


class TestValidProfilesUnaffected:
    """The wrapper must be a pass-through for everything legitimate."""

    def test_served_profile_resolves_to_its_own_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(served=["default", "coder"])

        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            home = runner._require_served_profile_home(_source(profile="coder"))
            # Identical to what the plain resolver would have returned — the
            # wrapper only changes the failure mode, never a valid answer.
            lenient = runner._resolve_profile_home_for_source(_source(profile="coder"))

        assert home == lenient
        assert "coder" in str(home)

    def test_active_profile_is_always_served(self, tmp_path, monkeypatch):
        """The multiplexer's own profile needs no adapter-map entry."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(served=[], active="default")

        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            assert runner._require_served_profile_home(_source(profile="default"))

    def test_route_precedence_preserved(self, tmp_path, monkeypatch):
        """An explicit stamp still wins over a matching route.

        Same precedence the plain resolver documents — validating must not
        reorder resolution, only reject unserveable answers.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        route = ProfileRoute(name="r", platform="slack", chat_id="C1", profile="routed")
        runner = _runner(served=["default", "routed", "stamped"], routes=[route])

        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            home = runner._require_served_profile_home(
                _source(profile="stamped", chat_id="C1")
            )

        assert "stamped" in str(home)
        assert "routed" not in str(home)

    def test_valid_route_resolves_when_served(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        route = ProfileRoute(name="r", platform="slack", chat_id="C1", profile="routed")
        runner = _runner(served=["default", "routed"], routes=[route])

        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            home = runner._require_served_profile_home(_source(chat_id="C1"))

        assert "routed" in str(home)

    def test_unstamped_unrouted_source_uses_active(self, tmp_path, monkeypatch):
        """No profile asserted → active profile, no validation, no raise."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(served=["default"])

        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            assert runner._require_served_profile_home(_source()) is not None


class TestNonMultiplexUnchanged:
    """Single-profile gateways keep the lenient fallback."""

    def test_unknown_profile_falls_back_without_multiplexing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner = _runner(multiplex=False, served=["default"])

        home = runner._require_served_profile_home(_source(profile="ghost"))

        assert home == Path(str(tmp_path))

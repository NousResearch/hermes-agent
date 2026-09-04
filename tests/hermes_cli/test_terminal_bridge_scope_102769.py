"""Regression for #102769: the terminal-config bridge guard must compare
against the TRUE process launch home, not the context-overridden home.

Under multi-profile serve, a routed profile's turn runs with
``HERMES_HOME``-style context overrides pointing at the routed profile's
home. ``load_hermes_dotenv(hermes_home=<routed>)`` then calls
``_reapply_terminal_config_bridge(<routed>)``. The old guard compared
against the context-overridden ``get_hermes_home()`` — which ALSO pointed
at the routed profile — so the guard passed and the shared bridge bridged
the ROUTED profile's terminal config (e.g. ``TERMINAL_ENV=docker`` with its
volumes) into the shared ``os.environ``. The launch profile's next
unscoped turn then executed in the routed profile's Docker backend.

Contract: the bridge may run ONLY when ``home_path`` equals the process
launch home (``get_process_hermes_home``), regardless of any active
context override.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import hermes_constants
from hermes_cli import env_loader


@pytest.fixture(autouse=True)
def _launch_home(tmp_path, monkeypatch):
    """Pin the process launch home and the context override to distinct dirs."""
    launch = tmp_path / "launch-home"
    routed = tmp_path / "routed-profile"
    launch.mkdir()
    routed.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    return SimpleNamespace(launch=launch, routed=routed)


class TestTerminalBridgeScopedToLaunchHome:
    def test_routed_profile_home_does_not_bridge(self, _launch_home):
        """The routed profile's dotenv reload must NOT re-apply the terminal
        bridge — that would bridge the routed profile's terminal config into
        the shared process env (#102769's hijack)."""
        applied = []

        with mock.patch.object(
            hermes_constants, "get_process_hermes_home", return_value=_launch_home.launch
        ), mock.patch(
            "hermes_cli.config.apply_terminal_config_to_env",
            side_effect=lambda env=None: applied.append(env),
        ):
            env_loader._reapply_terminal_config_bridge(_launch_home.routed)

        assert applied == []

    def test_launch_home_still_bridges(self, _launch_home):
        """Control: the launch profile's own dotenv reload still re-applies the
        bridge (the stale-``.env`` fix of #29186/#67323 is preserved)."""
        applied = []

        with mock.patch.object(
            hermes_constants, "get_process_hermes_home", return_value=_launch_home.launch
        ), mock.patch(
            "hermes_cli.config.apply_terminal_config_to_env",
            side_effect=lambda env=None: applied.append(env),
        ):
            env_loader._reapply_terminal_config_bridge(_launch_home.launch)

        assert applied == [None]

    def test_context_override_does_not_change_the_verdict(self, _launch_home):
        """Even when a context override points at the routed home (the exact
        multi-profile-serve shape), the routed home is still refused — the
        guard reads the launch home, not the context home."""
        applied = []

        token = hermes_constants._HERMES_HOME_OVERRIDE.set(str(_launch_home.routed))
        try:
            assert hermes_constants.get_hermes_home() == _launch_home.routed

            with mock.patch.object(
                hermes_constants, "get_process_hermes_home", return_value=_launch_home.launch
            ), mock.patch(
                "hermes_cli.config.apply_terminal_config_to_env",
                side_effect=lambda env=None: applied.append(env),
            ):
                env_loader._reapply_terminal_config_bridge(_launch_home.routed)
        finally:
            hermes_constants._HERMES_HOME_OVERRIDE.reset(token)

        assert applied == []

    def test_fail_open_on_guard_error(self, _launch_home):
        """A broken guard lookup must not break dotenv loading (fail-open)."""
        with mock.patch.object(
            hermes_constants,
            "get_process_hermes_home",
            side_effect=RuntimeError("boom"),
        ):
            env_loader._reapply_terminal_config_bridge(_launch_home.launch)  # must not raise

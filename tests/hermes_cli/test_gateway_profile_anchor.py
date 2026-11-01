"""Launcher profile anchoring: default-home launch argv must pin ``--profile default``.

``hermes_cli.gateway._profile_arg`` builds the child argv for every detached
gateway spawn, service unit and Scheduled Task launcher. For the default home
it historically returned "" — the spawned child then re-resolves its profile
at boot, and a sticky non-default ``active_profile`` re-homed the HOST gateway
to that profile, which the one-gateway-per-host multiplexer refuses. That is
the post-update gateway outage shape: the updater restarts the gateway without
an anchor, the child adopts the sticky profile, and the host gateway never
comes back up until someone runs it with an explicit ``-p default``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: tmp_path / ".hermes")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return tmp_path / ".hermes"


def _make_profile(root: Path, name: str) -> None:
    (root / "profiles" / name).mkdir(parents=True, exist_ok=True)
    (root / "profiles" / name / "config.yaml").write_text("{}\n")  # identity marker


class TestProfileArgAnchorsDefaultHome:
    def test_default_home_with_sticky_profile_returns_anchor(self, profile_env):
        from hermes_cli.gateway import _profile_arg
        from hermes_cli.profiles import set_active_profile

        _make_profile(profile_env, "pleroma")
        set_active_profile("pleroma")

        assert _profile_arg() == "--profile default"

    def test_default_home_without_sticky_profile_returns_empty(self, profile_env):
        from hermes_cli.gateway import _profile_arg

        assert _profile_arg() == ""

    def test_named_profile_home_keeps_its_flag(self, profile_env):
        from hermes_cli.gateway import _profile_arg

        _make_profile(profile_env, "pleroma")

        assert _profile_arg(
            hermes_home=str(profile_env / "profiles" / "pleroma")
        ) == "--profile pleroma"

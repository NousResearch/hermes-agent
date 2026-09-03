"""Named-profile cron list must see a live shared/default-home gateway (#99631).

``hermes -p triage cron list`` sets HERMES_HOME to the profile directory.
Current-home lock/pid probes therefore miss the gateway process that lives
in the default home and actually ticks that profile's cron store.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_cli import cron as cron_cli


@pytest.fixture()
def named_profile_home(tmp_path, monkeypatch):
    """``hermes -p triage`` layout: HERMES_HOME is the profile dir, not the root."""
    default = tmp_path / ".hermes"
    profile = default / "profiles" / "triage"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda *a, **k: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    hermes_constants._default_hermes_root_memo = None
    yield SimpleNamespace(root=tmp_path, default=default, profile=profile)
    hermes_constants._default_hermes_root_memo = None


def _patch_current_home_dead(monkeypatch) -> None:
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        "hermes_cli.gateway.named_profile_served_by_running_multiplexer",
        lambda: False,
    )


def _path_aware_lock(default_home: Path, *, default_live: bool):
    def _lock_active(lock_path=None):
        from gateway.status import _get_gateway_lock_path

        if lock_path is None:
            return False
        default_lock = _get_gateway_lock_path(default_home / "gateway.pid")
        try:
            return default_live and Path(lock_path).resolve() == default_lock.resolve()
        except OSError:
            return False

    return _lock_active


class TestNamedProfileSeesSharedGateway:
    def test_default_home_lock_live_is_not_false(
        self, named_profile_home, monkeypatch, capsys
    ):
        assert not (named_profile_home.profile / "gateway.pid").exists()
        _patch_current_home_dead(monkeypatch)
        monkeypatch.setattr(
            "gateway.status.is_gateway_runtime_lock_active",
            _path_aware_lock(named_profile_home.default, default_live=True),
        )

        assert cron_cli._builtin_gateway_liveness() is not False
        cron_cli._warn_if_gateway_not_running()
        assert "Gateway is not running" not in capsys.readouterr().out

    def test_default_home_pid_live_is_not_false(
        self, named_profile_home, monkeypatch, capsys
    ):
        assert not (named_profile_home.profile / "gateway.pid").exists()
        (named_profile_home.default / "gateway.pid").write_text(
            str(os.getpid()), encoding="utf-8"
        )
        _patch_current_home_dead(monkeypatch)
        monkeypatch.setattr(
            "gateway.status.is_gateway_runtime_lock_active",
            _path_aware_lock(named_profile_home.default, default_live=False),
        )

        assert cron_cli._builtin_gateway_liveness() is not False
        cron_cli._warn_if_gateway_not_running()
        assert "Gateway is not running" not in capsys.readouterr().out

    def test_neither_default_nor_profile_live_warns(
        self, named_profile_home, monkeypatch, capsys
    ):
        assert not (named_profile_home.profile / "gateway.pid").exists()
        assert not (named_profile_home.default / "gateway.pid").exists()
        _patch_current_home_dead(monkeypatch)
        monkeypatch.setattr(
            "gateway.status.is_gateway_runtime_lock_active",
            _path_aware_lock(named_profile_home.default, default_live=False),
        )

        assert cron_cli._builtin_gateway_liveness() is False
        cron_cli._warn_if_gateway_not_running()
        assert "Gateway is not running" in capsys.readouterr().out

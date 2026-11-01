"""Busy-input mode hot-refresh for the default profile.

``/busy`` persists ``display.busy_input_mode`` to config.yaml, and users edit
that file directly, but the gateway latched the value at startup — so a changed
mode did not take effect until a full gateway restart.

These tests pin the refresh behavior AND its boundary:
routed (multiplexed) profiles must keep using their startup snapshot, which is
what ``tests/gateway/test_multiplex_busy_input_mode.py`` asserts.
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SessionSource
from gateway.run import GatewayRunner


def _runner(*, multiplex: bool, default_mode: str = "interrupt") -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner._busy_input_mode = default_mode
    runner._busy_text_mode = "queue" if default_mode == "queue" else "interrupt"
    runner._profile_adapters = {}
    runner.adapters = {}
    runner._sessions = {}
    return runner


def _source(profile: str = "") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        profile=profile,
    )


def _write_config(home, mode: str) -> None:
    (home / "config.yaml").write_text(f"display:\n  busy_input_mode: {mode}\n", encoding="utf-8")


@pytest.fixture
def gateway_home(tmp_path, monkeypatch):
    """Point the gateway config loader at a temp home."""
    import gateway.run as gateway_run

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: home)
    # Startup env bridge must not win over the persisted file.
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INPUT_MODE", raising=False)
    return home


def test_default_profile_picks_up_config_change_without_restart(gateway_home):
    """The whole point: editing config.yaml takes effect on the next lookup."""
    _write_config(gateway_home, "interrupt")
    runner = _runner(multiplex=False, default_mode="interrupt")
    assert runner._effective_busy_input_mode(_source()) == "interrupt"

    _write_config(gateway_home, "queue")

    assert runner._effective_busy_input_mode(_source()) == "queue", (
        "busy mode did not hot-refresh after config.yaml changed"
    )
    # busy_input_mode is the source of truth for the text mode too.
    assert runner._effective_busy_text_mode(_source()) == "queue"


def test_stale_startup_env_bridge_loses_to_persisted_config(gateway_home, monkeypatch):
    """A stale startup env value must not override the persisted file."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "interrupt")
    _write_config(gateway_home, "steer")
    runner = _runner(multiplex=False, default_mode="interrupt")

    assert runner._effective_busy_input_mode(_source()) == "steer"


def test_unchanged_config_is_not_reread(gateway_home, monkeypatch):
    """Hot path stays cheap: no YAML re-read while the file is untouched."""
    import gateway.run as gateway_run

    _write_config(gateway_home, "queue")
    runner = _runner(multiplex=False, default_mode="interrupt")
    assert runner._effective_busy_input_mode(_source()) == "queue"

    calls = {"n": 0}
    real = gateway_run._load_gateway_config

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(gateway_run, "_load_gateway_config", counting)

    for _ in range(5):
        assert runner._effective_busy_input_mode(_source()) == "queue"

    assert calls["n"] == 0, f"config was re-read {calls['n']}x despite no file change"


def test_routed_profile_still_uses_startup_snapshot(gateway_home, monkeypatch):
    """Upstream contract preserved: multiplexed lookups do NOT re-read config.

    tests/gateway/test_multiplex_busy_input_mode.py asserts this explicitly, so
    the refresh must stay confined to the default path.
    """
    import gateway.run as gateway_run

    _write_config(gateway_home, "queue")
    runner = _runner(multiplex=True, default_mode="interrupt")
    runner.__dict__["_busy_input_modes_by_profile"] = {"research": "steer"}
    runner.__dict__["_busy_text_modes_by_profile"] = {"research": "interrupt"}

    def fail_config_read():
        raise AssertionError("routed busy-mode lookup reread config after startup")

    monkeypatch.setattr(gateway_run, "_load_gateway_config", fail_config_read)

    assert runner._effective_busy_input_mode(_source("research")) == "steer"
    assert runner._effective_busy_text_mode(_source("research")) == "interrupt"


def test_missing_config_falls_back_to_current_mode(gateway_home):
    """A missing/unreadable config must not crash or wipe the live mode."""
    runner = _runner(multiplex=False, default_mode="queue")

    assert runner._effective_busy_input_mode(_source()) == "queue"

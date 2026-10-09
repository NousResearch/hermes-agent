"""Gateway startup names every served profile whose config still declares a retired timed reset."""
import logging
from types import SimpleNamespace

import pytest

from gateway.run_startup import GatewayStartupMixin
from hermes_cli import session_reset_retirement


@pytest.mark.parametrize("plugin_enabled", [False, True])
def test_startup_warns_on_timed_session_reset_unless_plugin_enabled(tmp_path, monkeypatch, caplog, plugin_enabled):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("session_reset:\n  mode: idle\n  idle_minutes: 60\n", encoding="utf-8")
    monkeypatch.setattr("gateway.profile_serving.profiles_to_serve", lambda multiplex: [("default", home)])
    monkeypatch.setattr(session_reset_retirement, "reset_plugin_enabled", lambda: plugin_enabled)
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=False)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._start_log_retired_session_reset()

    warned = [r.getMessage() for r in caplog.records if "session_reset.mode: idle" in r.getMessage()]
    assert (warned == []) is plugin_enabled
    if not plugin_enabled:
        assert session_reset_retirement.PLUGIN_NAME in warned[0]


def test_only_timed_modes_count_as_retired_policy():
    find = session_reset_retirement.retired_reset_policy
    assert find({"session_reset": {"mode": "both"}}) == ("session_reset", "both")
    assert find({"gateway": {"session_reset": {"mode": "Daily"}}}) == ("gateway.session_reset", "daily")
    assert find({"session_reset": {"mode": "none"}}) is None
    assert find({"session_reset": {"idle_minutes": 60}}) is None
    assert find({}) is None

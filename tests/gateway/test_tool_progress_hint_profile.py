"""The long-tool /verbose tip must mark itself seen in the config it was read from.

A gateway serving several profiles reads the flag from the routed profile's config.yaml. Writing it
to the launch home's instead meant the tip came back on every long tool run (#154977, #156007).
"""

import queue
from types import SimpleNamespace

import hermes_yaml as yaml

import gateway.run as _gr
from agent.onboarding import TOOL_PROGRESS_FLAG, is_seen
from gateway.run_turn_progress import GatewayTurnProgressMixin
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _Progress(GatewayTurnProgressMixin):
    def __init__(self, ctx):
        self._ctx = ctx


def test_tool_progress_hint_flag_lands_in_the_routed_profile_config(tmp_path, monkeypatch):
    launch_home, profile_home = tmp_path / "launch", tmp_path / "profiles" / "coder"
    launch_home.mkdir()
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("display:\n  tool_progress_command: true\n")
    monkeypatch.setattr(_gr, "_hermes_home", launch_home)

    def _load():
        path = _gr._gateway_config_home() / "config.yaml"
        return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}

    monkeypatch.setattr(_gr, "_load_gateway_config", _load)

    ctx = SimpleNamespace(
        _LONG_TOOL_THRESHOLD_S=10, progress_mode="all",
        long_tool_hint_fired=[False], progress_queue=queue.Queue(),
    )
    token = set_hermes_home_override(profile_home)
    try:
        _Progress(ctx)._progress_onboarding_hint({"duration": 30})
        seen_on_next_run = is_seen(_gr._load_gateway_config(), TOOL_PROGRESS_FLAG)
    finally:
        reset_hermes_home_override(token)

    assert ctx.long_tool_hint_fired[0] is True
    assert ctx.progress_queue.qsize() == 1
    assert seen_on_next_run is True
    assert not (launch_home / "config.yaml").exists()

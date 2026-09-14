"""Reset metadata resolves read-only inside the serving profile."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.parametrize("delegation, expected", [
    ({}, "model: inherited from main; effort: inherited from main"),
    ({"model": "worker-model", "reasoning_effort": False}, "model: worker-model (configured); effort: off (configured)"),
    ({"reasoning_effort": "invalid"}, "model: inherited from main; effort: inherited from main (invalid setting)"),
])
def test_reset_metadata_uses_serving_profile(tmp_path, monkeypatch, delegation, expected):
    import yaml
    from hermes_cli import __version__
    profile = tmp_path / "profiles" / "planner"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "main-model", "context_length": 2000},
        "agent": {"reasoning_effort": "high", "service_tier": "fast"},
        "delegation": delegation, "approvals": {"mode": "manual"},
    }))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    source = SessionSource(platform=Platform.MATRIX, chat_id="room", profile="planner")
    with patch.object(runner, "_resolve_profile_home_for_source", return_value=profile), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
         patch("tools.approval._YOLO_MODE_FROZEN", True):
        info = runner._reset_notice_session_info(source)
    assert f"Profile: planner · Hermes {__version__}" in info
    assert "Main reasoning: high" in info
    assert "Service tier (requested): priority" in info
    assert f"Delegation default — {expected}" in info
    assert "Tool approval: off (runtime override)" in info
    assert "main-model" in info


def test_metadata_failure_does_not_hide_existing_model_info():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    source = SessionSource(platform=Platform.MATRIX, chat_id="room")
    with patch.object(runner, "_format_session_info", return_value="existing model block"), \
         patch.object(runner, "_resolve_session_agent_runtime", return_value=("model", {})), \
         patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("private detail")):
        info = runner._reset_notice_session_info(source)
    assert "existing model block" in info
    assert "Session settings: unknown" in info
    assert "private detail" not in info

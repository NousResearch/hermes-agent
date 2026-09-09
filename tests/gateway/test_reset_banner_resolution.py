"""Behavior contracts for new-session metadata, without provider requests."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, ChannelOverride
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.parametrize("channel", [False, True])
def test_new_defaults_follow_resolved_route(tmp_path, monkeypatch, channel):
    import yaml
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "base", "context_length": 2000},
        "agent": {"reasoning_effort": "low", "service_tier": "fast",
                  "reasoning_overrides": {"fallback": "high", "channel": "xhigh"}},
    }))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.MATRIX: PlatformConfig(
        channel_overrides={"room": ChannelOverride(model="channel")} if channel else {})})
    source = SessionSource(platform=Platform.MATRIX, chat_id="room")
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"model": "fallback", "provider": "custom"}), \
         patch("agent.model_metadata.get_model_context_length", return_value=2000):
        info = runner._reset_notice_session_info(source)
    assert f"Model: `{'channel' if channel else 'fallback'}`" in info
    assert f"Main reasoning: {'xhigh' if channel else 'high'}" in info
    assert "Service tier (requested): priority" in info
    assert not runner._sessions_map()


@pytest.mark.parametrize("mode,expected", [("smart", "smart"), ("off", "off"), ("bogus", "manual"), (None, "manual")])
def test_approval_defaults(tmp_path, monkeypatch, mode, expected):
    import yaml
    from gateway.session_banner import format_reset_settings
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"approvals": {"mode": mode}}))
    runner = SimpleNamespace(_load_reasoning_config=lambda model="": None, _load_service_tier=lambda: None)
    with patch("tools.approval._YOLO_MODE_FROZEN", False):
        text = format_reset_settings(runner, model="model")
    assert f"Tool approval: {expected}" in text


@pytest.mark.parametrize("bad", [[], "invalid", 42])
def test_malformed_delegation_is_unknown_not_exception(tmp_path, monkeypatch, bad):
    import yaml
    from gateway.session_banner import format_reset_settings
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"delegation": bad}))
    runner = SimpleNamespace(_load_reasoning_config=lambda model="": None, _load_service_tier=lambda: None)
    assert "Delegation default: unknown" in format_reset_settings(runner, model="model")


def test_short_id_is_unambiguous_and_resolvable(tmp_path):
    from hermes_state import SessionDB
    from gateway.session_banner import session_identifier
    db = SessionDB(tmp_path / "state.db")
    try:
        first = "20260101_120000_abcdef"
        second = "20260101_120000_abcdff"
        db.create_session(first, "matrix")
        db.create_session(second, "matrix")
        short = session_identifier(first, db.resolve_session_id)
        assert len(short) < len(first)
        assert db.resolve_session_id(short) == first
        assert session_identifier(first, lambda prefix: None) == first
        assert session_identifier(first, None) == first
    finally:
        db.close()

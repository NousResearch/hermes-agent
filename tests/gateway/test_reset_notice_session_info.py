"""reset_notice_session_info: keep model/provider out of reset notices.

The manual /new-//reset reply and the auto-reset notification both append a
session-info block (model, provider, context window) resolved by
``GatewayRunner._reset_notice_session_info``. That block is emitted by the
gateway itself, so no system-prompt instruction can suppress it — operators
of multi-user group chats need a config switch to keep the runtime identity
private. These tests cover the config plumbing and the single choke point.
"""

from types import SimpleNamespace

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def test_config_defaults_to_showing_session_info():
    cfg = GatewayConfig.from_dict({})
    assert cfg.reset_notice_session_info is True


def test_config_parses_disabled_value():
    cfg = GatewayConfig.from_dict({"reset_notice_session_info": False})
    assert cfg.reset_notice_session_info is False
    # String forms coming from hand-edited YAML must coerce too.
    cfg = GatewayConfig.from_dict({"reset_notice_session_info": "false"})
    assert cfg.reset_notice_session_info is False


def test_config_round_trips_through_to_dict():
    cfg = GatewayConfig.from_dict({"reset_notice_session_info": False})
    assert cfg.to_dict()["reset_notice_session_info"] is False


def _stub_runner(reset_notice_session_info: bool) -> SimpleNamespace:
    """Minimal stand-in exercising the real unbound method."""
    stub = SimpleNamespace(
        config=SimpleNamespace(
            reset_notice_session_info=reset_notice_session_info,
            multiplex_profiles=False,
        ),
        _format_session_info=lambda: "◆ Model: `some-model`",
    )
    return stub


def test_session_info_returned_when_enabled():
    stub = _stub_runner(reset_notice_session_info=True)
    result = GatewayRunner._reset_notice_session_info(stub, source=None)
    assert "some-model" in result


def test_session_info_suppressed_when_disabled():
    stub = _stub_runner(reset_notice_session_info=False)
    result = GatewayRunner._reset_notice_session_info(stub, source=None)
    assert result == ""

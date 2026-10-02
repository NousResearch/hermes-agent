"""Native fixture configuration preserves feedback and mode settings."""

import inspect
from pathlib import Path

import pytest

import hermes_yaml as yaml
from gateway.config import Platform, load_gateway_config
from gateway.config_loader import read_yaml_layers
from plugins.platforms.matrix.adapter_feedback import MatrixFeedbackPolicy, ReadReceiptMode
from tests.fakes.fake_llm_provider import write_hermes_home
from tests.integration.matrix_live import conftest as live_fixtures


@pytest.mark.parametrize("feedback,module_feedback,extra_feedback,expected_feedback", [
    pytest.param(
        live_fixtures.MatrixFeedbackSettings(), None, None,
        MatrixFeedbackPolicy(ReadReceiptMode.IMMEDIATE, False), id="immediate-default",
    ),
    pytest.param(
        live_fixtures.MatrixFeedbackSettings(read_receipts="after_processing", reactions=True),
        None, None, MatrixFeedbackPolicy(ReadReceiptMode.AFTER_PROCESSING, True),
        id="after-processing-default",
    ),
    pytest.param(
        live_fixtures.MatrixFeedbackSettings(read_receipts="disabled"), None, None,
        MatrixFeedbackPolicy(ReadReceiptMode.DISABLED, False), id="disabled-default",
    ),
    pytest.param(
        live_fixtures.MatrixFeedbackSettings(),
        live_fixtures.MatrixFeedbackSettings(read_receipts="after_processing", reactions=True),
        None, MatrixFeedbackPolicy(ReadReceiptMode.AFTER_PROCESSING, True),
        id="module-overrides-defaults",
    ),
    pytest.param(
        live_fixtures.MatrixFeedbackSettings(read_receipts="disabled"),
        live_fixtures.MatrixFeedbackSettings(read_receipts="after_processing", reactions=True),
        live_fixtures.MatrixFeedbackSettings(),
        MatrixFeedbackPolicy(ReadReceiptMode.IMMEDIATE, False), id="extra-overrides-module",
    ),
])
def test_queued_context_receives_feedback_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    feedback: live_fixtures.MatrixFeedbackSettings,
    module_feedback: live_fixtures.MatrixFeedbackSettings | None,
    extra_feedback: live_fixtures.MatrixFeedbackSettings | None,
    expected_feedback: MatrixFeedbackPolicy,
) -> None:
    authored_config = yaml.safe_load(inspect.unwrap(live_fixtures.gateway_config)())
    if module_feedback is not None:
        authored_config["platforms"]["matrix"].update({
            "read_receipts": module_feedback.read_receipts,
            "reactions": module_feedback.reactions,
        })
    extra = {
        "display": {"status": "compact"},
        "plugins": {"enabled": ["module-plugin"], "directory": "module-plugins"},
        "auxiliary": {"title_generation": {"enabled": False}},
        "platform_toolsets": {"matrix": ["hermes-matrix", "matrix_admin"]},
        "matrix": {"require_mention": False},
    }
    if extra_feedback is not None:
        extra["platforms"] = {"matrix": {
            "read_receipts": extra_feedback.read_receipts,
            "reactions": extra_feedback.reactions,
        }}
    composed = live_fixtures._gateway_yaml_config(
        yaml.safe_dump(authored_config), feedback,
        live_fixtures.GatewaySettings(mode="pause-queued-context"),
        "!feedback:matrix.test", "interrupt", yaml.safe_dump(extra),
    )
    write_hermes_home(tmp_path, "http://127.0.0.1:1/v1", extra_config=composed)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = read_yaml_layers(tmp_path)
    config = load_gateway_config()
    expected_matrix = {
        **authored_config["platforms"]["matrix"],
        "read_receipts": expected_feedback.read_receipts.value,
        "reactions": expected_feedback.reactions,
    }

    assert {
        "platforms": raw["platforms"],
        "display": raw["display"],
        "plugins": raw["plugins"],
        "auxiliary": raw["auxiliary"],
        "approvals": raw["approvals"],
        "updates": raw["updates"],
        "matrix_enabled": config.platforms[Platform.MATRIX].enabled,
        "feedback_policy": MatrixFeedbackPolicy.from_config(config.platforms[Platform.MATRIX]),
        "platform_toolsets": raw["platform_toolsets"],
        "matrix": raw["matrix"],
    } == {
        "platforms": {"matrix": expected_matrix},
        "display": {"status": "compact", "busy_input_mode": "queue",
                    "busy_ack_enabled": False, "busy_text_mode": "interrupt"},
        "plugins": {"enabled": ["module-plugin", "matrix-live-context"],
                    "directory": "module-plugins"},
        "auxiliary": extra["auxiliary"],
        "approvals": {"mode": "manual", "timeout": 15},
        "updates": authored_config["updates"],
        "matrix_enabled": True,
        "feedback_policy": expected_feedback,
        "platform_toolsets": extra["platform_toolsets"],
        "matrix": extra["matrix"],
    }

"""Runtime seams for the extracted conversation error classifier."""

from importlib import import_module
from unittest.mock import Mock

import pytest

classifiers = import_module("agent.conversation_error_classifiers")
conversation_loop = import_module("agent.conversation_loop")


@pytest.mark.parametrize(
    ("status_code", "message", "expected"),
    [
        (400, "model_not_available_for_integrator", True),
        (400, "MODEL_NOT_SUPPORTED", True),
        (None, "error code: 400; the requested model is not supported", True),
        (401, "model_not_supported", False),
        (400, "the model name is invalid", False),
        (None, "temporary upstream failure", False),
    ],
)
def test_stale_copilot_credential_classifier_behavior(status_code, message, expected):
    assert classifiers._is_stale_copilot_credential_error(status_code, message) is expected


def test_legacy_namespace_exports_the_same_callable():
    assert conversation_loop._is_stale_copilot_credential_error is classifiers._is_stale_copilot_credential_error


def test_legacy_namespace_binding_remains_patchable(monkeypatch):
    replacement = Mock(return_value="patched")
    monkeypatch.setattr(conversation_loop, "_is_stale_copilot_credential_error", replacement)

    assert conversation_loop._is_stale_copilot_credential_error(400, "anything") == "patched"
    replacement.assert_called_once_with(400, "anything")

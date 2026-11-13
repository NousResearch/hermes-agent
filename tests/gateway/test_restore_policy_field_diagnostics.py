"""#148540: a refused receipt must say WHICH check refused it.

``restore_policy`` used to collapse every validation into one bare ``ValueError('invalid policy')``,
so both the client frame and the gateway log carried only ``storage_unavailable`` plus a line number.
``RuntimeStoreError`` already carries ``field``/``detail`` for exactly this (and
``session_local_migration`` uses them), so each check now names itself. These tests pin the field
names, keep the wire code unchanged, and pin that the log line names the field too.
"""
import json

import pytest

from gateway.session_policy import restore_policy
from hermes_state_runtime import RuntimeStoreError

CWD = "/tmp/restore-policy-field-diagnostics"


def stored_policy(**overrides):
    """A receipt as our own writers persist it — valid unless a test overrides one field."""
    data = {
        "source": "cli",
        "platform": "cli",
        "cwd": CWD,
        "model": "gemma-4",
        "toolsets": ["terminal"],
        "config_json": json.dumps({"model": {"default": "gemma-4"}}),
        "request_json": "{}",
        "terminal_json": json.dumps({"TERMINAL_CWD": CWD}),
    }
    data.update(overrides)
    return data


def refuse(**overrides):
    with pytest.raises(RuntimeStoreError) as caught:
        restore_policy(stored_policy(**overrides))
    return caught.value


def test_valid_receipt_still_restores():
    restored = restore_policy(stored_policy())
    assert restored.model == "gemma-4"
    assert restored.toolsets == ("terminal",)


@pytest.mark.parametrize("overrides, expected_field", [
    ({"model": None}, "model"),
    ({"model": ""}, "model"),
    ({"model": "   "}, "model"),
    ({"model": 42}, "model"),
    ({"cwd": "relative/path"}, "cwd"),
    ({"cwd": None}, "cwd"),
    ({"platform": "tui"}, "source"),
    ({"source": "nowhere", "platform": "nowhere"}, "source"),
    ({"toolsets": "terminal"}, "toolsets"),
    ({"toolsets": [1, 2]}, "toolsets"),
    ({"safe_mode": "yes"}, "safe_mode"),
    ({"ignore_user_config": "yes"}, "ignore_user_config"),
    ({"safe_mode": True}, "safe_mode"),
    ({"skills_prompt": 5}, "skills_prompt"),
    ({"coding_posture": "yes"}, "coding_posture"),
    ({"client_env_json": "[]"}, "client_env_json"),
    ({"client_env_remote": "yes"}, "client_env_remote"),
    ({"client_env_rules_json": "[]"}, "client_env_rules_json"),
    ({"edit_approval_policy": "always"}, "edit_approval_policy"),
    ({"edit_approval_policy": "session"}, "edit_approval_policy"),
    ({"config_json": "[]"}, "config_json"),
    ({"request_json": "[]"}, "request_json"),
    ({"terminal_json": "[]"}, "terminal_json"),
    ({"terminal_json": json.dumps({"TERMINAL_CWD": "/somewhere/else"})}, "terminal_json"),
])
def test_refusal_names_the_failing_field(overrides, expected_field):
    error = refuse(**overrides)
    assert error.reason == "storage_unavailable", error.reason
    assert error.field == expected_field, (error.field, error.detail)
    assert error.detail, "a named field without a detail tells the operator nothing"


def test_detail_stays_one_line_and_bounded():
    error = refuse(source="weird\nsource", platform="x")
    assert "\n" not in (error.detail or "")
    assert len(error.field) <= 80 and len(error.detail) <= 160


def test_log_line_names_the_field_not_just_the_line():
    """``_refused_check`` is what the gateway logs for a refused receipt (#140099)."""
    from gateway.session_local_recovery import _refused_check

    with pytest.raises(RuntimeStoreError) as caught:
        restore_policy(stored_policy(model=None))
    logged = _refused_check(caught.value)
    assert "storage_unavailable[model]" in logged, logged
    assert "names no model" in logged, logged

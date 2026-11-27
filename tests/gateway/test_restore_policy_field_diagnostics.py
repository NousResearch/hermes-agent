"""#148540: a refused receipt must say WHICH check refused it.

``restore_policy`` used to collapse every validation into one bare ``ValueError('invalid policy')``,
so both the client frame and the gateway log carried only ``storage_unavailable`` plus a line number.
``AuthorityRefusal`` already carries ``field``/``detail`` for exactly this, so each check now names
itself; the wire code stays ``storage_unavailable``.
"""
import json
import logging

import pytest

from gateway.session_policy import restore_policy
from persistence.core.errors import AuthorityRefusal

CWD = "/srv/restore-policy-field-diagnostics"


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


@pytest.mark.parametrize("overrides, expected_field", [
    ({"model": "   "}, "model"),
    ({"model": 42}, "model"),
    ({"cwd": "relative/path"}, "cwd"),
    ({"cwd": None}, "cwd"),
    ({"platform": "tui"}, "source"),
    ({"source": "nowhere", "platform": "nowhere"}, "source"),
    ({"source": []}, "source"),
    ({"source": {}}, "source"),
    ({"source": "local", "platform": []}, "source"),
    ({"edit_approval_policy": ["ask"]}, "edit_approval_policy"),
    ({"toolsets": [1, 2]}, "toolsets"),
    ({"safe_mode": "yes"}, "safe_mode"),
    ({"ignore_user_config": "yes"}, "ignore_user_config"),
    ({"safe_mode": True}, "safe_mode"),
    ({"skills_prompt": 5}, "skills_prompt"),
    ({"coding_posture": "yes"}, "coding_posture"),
    ({"client_env_json": "{"}, "client_env_json"),
    ({"client_env_remote": "yes"}, "client_env_remote"),
    ({"client_env_rules_json": "[]"}, "client_env_rules_json"),
    ({"edit_approval_policy": "session"}, "edit_approval_policy"),
    ({"config_json": "{"}, "config_json"),
    ({"request_json": "[]"}, "request_json"),
    ({"terminal_json": json.dumps({"TERMINAL_CWD": "/somewhere/else"})}, "terminal_json"),
])
def test_refusal_names_the_failing_field(overrides, expected_field):
    from gateway.session_local_recovery import _refused_check

    with pytest.raises(AuthorityRefusal) as caught:
        restore_policy(stored_policy(**overrides))
    error = caught.value
    assert error.reason == "storage_unavailable", error.reason
    assert error.field == expected_field, (error.field, error.detail)
    assert error.detail, "a named field without a detail tells the operator nothing"
    assert f"storage_unavailable[{expected_field}]: {error.detail}" in _refused_check(error)


@pytest.mark.parametrize("not_an_object", [None, [], "policy"])
def test_a_receipt_whose_policy_is_not_an_object_names_it(not_an_object):
    with pytest.raises(AuthorityRefusal) as caught:
        restore_policy(not_an_object)
    assert (caught.value.reason, caught.value.field) == ("storage_unavailable", "policy")


@pytest.mark.parametrize("missing", ["model", "cwd", "toolsets"])
def test_a_receipt_missing_a_required_field_names_it(missing):
    stored = stored_policy()
    del stored[missing]
    with pytest.raises(AuthorityRefusal) as caught:
        restore_policy(stored)
    assert (caught.value.reason, caught.value.field) == ("storage_unavailable", missing)


def test_a_named_storage_refusal_warns_once_per_check(caplog, monkeypatch):
    """Desktop reopens a refused chat dozens of times: the gateway names the check at WARNING once,
    and the identical repeats stay at debug, as ``_log_storage_failure`` keeps one traceback."""
    from gateway import session_controls

    monkeypatch.setattr(session_controls, "_STORAGE_REFUSALS", set())
    with pytest.raises(AuthorityRefusal) as caught:
        restore_policy(stored_policy(cwd="relative/path"))
    with caplog.at_level(logging.DEBUG, logger="gateway.session_controls"):
        for _ in range(3):
            session_controls._log_refusal("default", "session.resume", caught.value)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warned == ["session.resume refused for profile=default: storage_unavailable field=cwd "
                      "detail=the stored working directory is missing or not absolute"], warned
    assert sum(r.levelno == logging.DEBUG for r in caplog.records) == 2

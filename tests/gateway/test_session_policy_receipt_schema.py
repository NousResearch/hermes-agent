"""A stored receipt written on the other side of a release boundary still restores (#151667).

``restore_policy`` constructed ``LocalSessionPolicy(**data)`` verbatim, so a receipt carrying one
key this build's dataclass does not define raised ``TypeError`` and was answered
``storage_unavailable`` — the same code as a disk fault — on every restart. The session could never
be re-adopted, and a cron job delivering into that Bot Chat reported ``delivery_failed ... unverified:
storage_unavailable`` for over a day.

These tests pin both halves: unknown keys are adopted away (and named in the log — keys only, a
receipt's values may hold credentials), while a receipt that is genuinely incomplete still refuses.
"""
import json

import pytest

from gateway.session_policy import restore_policy
from hermes_state_runtime import RuntimeStoreError


def _receipt(cwd):
    """A stored receipt carrying exactly the fields this build's dataclass defines."""
    return {"source": "tui", "platform": "tui", "cwd": cwd, "model": "fixture",
            "toolsets": ["terminal"], "config_json": "{}", "request_json": "{}",
            "terminal_json": json.dumps({"TERMINAL_CWD": cwd})}


def test_a_receipt_with_a_key_this_build_dropped_still_restores(tmp_path):
    policy = restore_policy({**_receipt(str(tmp_path)), "fallback_providers": ["fixture"]})
    assert (policy.cwd, policy.model) == (str(tmp_path), "fixture")
    assert not hasattr(policy, "fallback_providers")


def test_every_unknown_key_is_dropped_not_just_the_first(tmp_path):
    policy = restore_policy({**_receipt(str(tmp_path)), "fallback_providers": [],
                             "some_future_field": 1})
    assert (policy.source, policy.platform) == ("tui", "tui")


def test_the_dropped_keys_are_named_in_the_log_but_their_values_are_not(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="gateway.session_policy"):
        restore_policy({**_receipt(str(tmp_path)), "fallback_providers": ["a-secret-endpoint"]})
    lines = [record.getMessage() for record in caplog.records]
    assert any("fallback_providers" in line for line in lines), lines
    assert not any("a-secret-endpoint" in line for line in lines), lines


def test_a_genuinely_incomplete_receipt_still_refuses(tmp_path):
    """Adopting unknown keys must not weaken "reject incomplete private policy"."""
    incomplete = {k: v for k, v in _receipt(str(tmp_path)).items() if k != "model"}
    with pytest.raises(RuntimeStoreError) as caught:
        restore_policy({**incomplete, "model_name": "fixture"})
    assert caught.value.reason == "storage_unavailable"


@pytest.mark.parametrize("payload", [None, [], "x"])
def test_a_non_mapping_payload_keeps_refusing(payload):
    with pytest.raises(RuntimeStoreError) as caught:
        restore_policy(payload)
    assert caught.value.reason == "storage_unavailable"

"""A local automation whose ``completion_wakes_parent`` is anything but ``True`` is refused as invalid params."""
from types import SimpleNamespace

import pytest

from gateway import session_automation
from persistence import AuthorityRefusal


@pytest.mark.parametrize("value", [False, "yes", 1, None])
def test_a_wakes_parent_flag_other_than_true_is_an_invalid_params_refusal(monkeypatch, value):
    monkeypatch.setattr(session_automation, "_local_automation_owner",
                        lambda *_a: (SimpleNamespace(session_id="parent"), None))
    event = SimpleNamespace(metadata={"completion_wakes_parent": value})
    entry = SimpleNamespace(session_key="route", session_id="parent")
    with pytest.raises(AuthorityRefusal) as refused:
        session_automation.snapshot_local_automation(None, None, event, "identity", entry)
    assert refused.value.reason == "invalid_params"

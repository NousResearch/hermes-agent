"""With admin-only approval buttons on, a non-admin's own DM can't answer an approval prompt."""

from types import SimpleNamespace

import pytest

from gateway.run_turn_runner_approval import _ADMIN_GATE_REASON, unanswerable_approval_reason


def _adapter(**extra):
    return SimpleNamespace(config=SimpleNamespace(extra=extra))


def _source(user_id="222", chat_type="dm"):
    return SimpleNamespace(user_id=user_id, chat_type=chat_type, chat_id=user_id)


GATED = {"require_admin_for_exec_approval": True, "allow_admin_from": ["111"]}


@pytest.mark.parametrize("extra,source,expected", [
    (GATED, _source("222"), _ADMIN_GATE_REASON),                        # non-admin, own DM
    (GATED, _source("111"), None),                                      # admin's own DM
    (GATED, _source("222", chat_type="group"), None),                   # an admin may be in the group
    ({"allow_admin_from": ["111"]}, _source("222"), None),              # toggle off (default)
    ({"require_admin_for_exec_approval": "true"}, _source("222"), _ADMIN_GATE_REASON),  # no admins: fail closed
    ({}, _source("222"), None),
])
def test_admin_gate_reason(extra, source, expected):
    assert unanswerable_approval_reason(_adapter(**extra), source) == expected


def test_adapter_reason_takes_precedence():
    class _Guest:
        config = SimpleNamespace(extra=GATED)

        def exec_approval_unanswerable(self, source):
            return "this chat can't show approval prompts."

    assert unanswerable_approval_reason(_Guest(), _source("222")) == "this chat can't show approval prompts."


def test_adapter_without_config_is_answerable():
    assert unanswerable_approval_reason(object(), _source("222")) is None

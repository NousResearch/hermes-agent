"""The core approval callback stays available and tracks its work."""

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from tests.gateway.test_group_home_consent import command, home
from tests.gateway.group_chat_picker_fixtures import picker_home, choose_first


@pytest.mark.asyncio
async def test_picker_home_approval_is_counted_during_maintenance(picker_home, monkeypatch):
    await command(picker_home, "/group")
    await command(picker_home, "/group confirm")
    picker_home.service.service = None
    picker_home.service.room_status["pending_actions"] = [{
        "kind": "approval", "member_id": "default", "task_id": "task-1",
        "request_id": "request-1", "execution_generation": 1,
        "authority_gateway_id": "install:test-gateway", "authority_epoch": 1,
        "approval": {"description": "Run local check", "command": "echo test",
                     "choices": ["once", "deny"]},
    }]
    counts = []
    original = approvals.submit_room_approval

    def observed(*args, **kwargs):
        counts.append(picker_home.runner._active_deferred_agent_worker_count())
        return original(*args, **kwargs)

    monkeypatch.setattr(approvals, "submit_room_approval", observed)
    assert await command(picker_home, "/group 1 approvals") is None
    picker_home.runner._external_drain_active = True
    result = await choose_first(picker_home)
    assert counts == [1]
    assert picker_home.runner._active_deferred_agent_worker_count() == 0
    assert "Decision sent" in str(result)

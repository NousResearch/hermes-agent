"""Membership surrounds content projection and the final resource identity."""
import base64
import json
import threading

import pytest

from gateway.task_read import TaskReadDenied, TaskReadService
from gateway.work_presentation import TrustedWorkAudience


def selector():
    raw = json.dumps(["default", "default", "t_12345678", 1], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def reader():
    service = TaskReadService.__new__(TaskReadService)
    service.lock = threading.RLock()
    bot = object()
    service._authorize = lambda app, raw, selected: (
        "policy", None, None, bot, None, None, 99, 123, False)
    return service


@pytest.mark.asyncio
async def test_membership_precedes_projection_and_final_identity_is_rechecked():
    service = reader()
    audience = TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)
    identity = ["path", 1, 2, 1, 4]
    events = []

    def admission(selected):
        events.append("audience")
        return audience, tuple(identity)

    def project(selected):
        events.append("project")
        return {"title": "safe", "status": "running", "incarnation": 1,
                "revision": 4, "updated_at": 1}, audience, tuple(identity)

    async def member(bot, actor, bot_id, admitted):
        events.append("member")

    service._audience, service._project, service._member = admission, project, member
    result = await service.detail(object(), "signed", selector())
    assert result["title"] == "safe"
    assert events.index("member") < events.index("project")
    assert events.count("member") == 2 and events[-1] == "audience"


@pytest.mark.asyncio
async def test_nonmember_never_reaches_content_projection():
    service = reader()
    audience = TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)
    service._audience = lambda selected: (audience, ("resource", 1))

    async def denied(*args):
        raise TaskReadDenied("unavailable")

    service._member = denied
    service._project = lambda selected: pytest.fail("content projected before membership")
    with pytest.raises(TaskReadDenied) as error:
        await service.detail(object(), "signed", selector())
    assert error.value.reason == "unavailable"


@pytest.mark.asyncio
async def test_resource_change_during_second_membership_denies_prior_projection():
    service = reader()
    audience = TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)
    identity = ["resource", 1]
    service._audience = lambda selected: (audience, tuple(identity))
    service._project = lambda selected: ({"title": "old"}, audience, tuple(identity))
    calls = 0

    async def member(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            identity[-1] = 2

    service._member = member
    with pytest.raises(TaskReadDenied) as error:
        await service.detail(object(), "signed", selector())
    assert error.value.reason == "unavailable"

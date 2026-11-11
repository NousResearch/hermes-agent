"""Real guarded-client lifecycle contracts for persistent native plan snapshots."""
import asyncio
from types import SimpleNamespace

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse

from gateway.slack_task_stream import RichTaskCardSession
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


@pytest.mark.asyncio
async def test_guarded_rate_limit_stays_on_native_lane_and_idle_flush_recovers(monkeypatch):
    from gateway import slack_task_stream, slack_plan_snapshot
    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(slack_task_stream, "time", clock)
    monkeypatch.setattr(slack_plan_snapshot, "time", clock)
    adapter, client = direct_adapter()
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    session.main.ROLLOVER_MAX_AGE_S = 1
    original = client.api_call
    reject = [True]
    async def controlled(method, *, json):
        if method == "chat.update" and reject[0]:
            reject[0] = False
            response = SlackResponse(client=None, http_verb="POST", api_url="https://slack.com/api/chat.update",
                req_args={}, data={"ok": False, "error": "ratelimited"}, headers={"Retry-After": "7"}, status_code=429)
            raise SlackApiError("fixture", response)
        return await original(method, json=json)
    client.api_call = controlled
    assert (await session.publish([{"type": "tool.started", "tool_call_id": "a", "tool_name": "read_file"}])).success
    now[0] += 2
    result = await session.publish([{"type": "tool.completed", "tool_call_id": "a", "tool_name": "read_file"}])
    assert result.success, "a known idempotent plan retry must not trigger a new text lane"
    assert client.opens == 1
    now[0] += 7
    await session.flush_pending()
    updates = [p for method, p in client.calls if method == "chat.update"]
    assert updates and updates[-1]["blocks"][0]["tasks"][0]["status"] == "complete"
    await session.stop()


@pytest.mark.asyncio
async def test_stop_during_capacity_open_closes_new_stream_without_content(monkeypatch):
    from gateway import slack_task_stream, slack_plan_snapshot
    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(slack_task_stream, "time", clock)
    monkeypatch.setattr(slack_plan_snapshot, "time", clock)
    adapter, client = direct_adapter()
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    card = session.main
    card.ROLLOVER_MAX_AGE_S = 1
    card._plan.MAX_TASKS = 1
    original = client.api_call
    opening = asyncio.Event()
    release = asyncio.Event()
    async def controlled(method, *, json):
        if method == "chat.startStream" and client.opens == 1:
            opening.set()
            await release.wait()
        return await original(method, json=json)
    client.api_call = controlled
    await card.task_started("a", "read_file")
    now[0] += 10
    publishing = asyncio.create_task(card.task_started("b", "read_file"))
    await opening.wait()
    stopping = asyncio.create_task(card.stop(flush_reasoning=False))
    await asyncio.sleep(0)
    release.set()
    await publishing
    await stopping
    assert client.opens == 2
    assert any(method == "chat.stopStream" and p["ts"] == "stream-2" for method, p in client.calls)
    assert not any(method in {"chat.update", "chat.appendStream"} and p.get("ts") == "stream-2" for method, p in client.calls)

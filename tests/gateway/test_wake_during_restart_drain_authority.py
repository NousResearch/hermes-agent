"""A wake that arrives while the gateway drains must be refused retryably on the authority route too.

Once a chat has a committed human turn, ``deliver_wake`` commits a Kanban wake to the session
authority's FIFO (``admit_route_wake``) instead of the classic adapter path the sibling file covers.
That FIFO's admission gate already refuses during a drain, re-checked inside the write; these pin it
so both routes keep the same contract: a refused wake carries ``runtime_draining``, which the
notifier retries, and nothing is half-admitted. Real GatewayRunner, DiscordAdapter and
SessionAuthority over SQLite.
"""

import asyncio

import pytest

from tests.waiting import async_wait_until

WAKE = "[Kanban task completed] t_1"


class Hold:
    """Keeps the human turn running until ``finish`` is set."""

    def __init__(self):
        self.running, self.finish = asyncio.Event(), asyncio.Event()


async def authority_route(monkeypatch, executed, hold=None):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    monkeypatch.setattr(adapter, "send", send)
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if hold is not None and event.text == "A":
            hold.running.set()
            await hold.finish.wait()
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply " + event.text, "messages": []}
        return "reply " + event.text

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    return runner, adapter, authority, adapter.build_source(chat_id="42", chat_type="dm", user_id="42")


def admissions(runner, authority, source):
    from hermes_state_runtime import list_session_admissions

    owner = runner.session_store.peek_session_id(runner.session_store._generate_session_key(source))
    rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
    return [(row["principal_id"].split(":")[0], row["status"]) for row in rows]


async def start_human_turn(runner, adapter, authority, source, hold):
    from gateway.platforms.event import MessageEvent

    await adapter.handle_message(MessageEvent(text="A", source=source, message_id="A"))
    if hold is not None:
        await asyncio.wait_for(hold.running.wait(), 10)
    else:
        await async_wait_until(
            lambda: admissions(runner, authority, source) == [("messaging", "terminal")],
            what="the human turn to settle",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_authority_route_refuses_a_wake_while_draining(monkeypatch, busy):
    from gateway.wake import WakeNotAccepted, deliver_wake

    executed, hold = [], Hold() if busy else None
    runner, adapter, authority, source = await authority_route(monkeypatch, executed, hold)
    await start_human_turn(runner, adapter, authority, source, hold)
    runner._draining = True  # /restart is waiting for other sessions to finish
    with pytest.raises(WakeNotAccepted) as refused:
        await asyncio.wait_for(deliver_wake(adapter, text=WAKE, source=source), 10)
    assert refused.value.reason == "runtime_draining" and not refused.value.permanent
    if hold is not None:
        hold.finish.set()
        await async_wait_until(
            lambda: admissions(runner, authority, source) == [("messaging", "terminal")],
            what="the human turn to settle",
        )
    assert admissions(runner, authority, source) == [("messaging", "terminal")]  # no wake row
    assert executed == ["A"]


@pytest.mark.asyncio
async def test_authority_route_refuses_a_wake_when_the_drain_starts_during_admission(monkeypatch):
    import gateway.session_automation as automation
    from gateway.wake import WakeNotAccepted, deliver_wake

    executed = []
    runner, adapter, authority, source = await authority_route(monkeypatch, executed)
    await start_human_turn(runner, adapter, authority, source, None)
    snapshot = automation.snapshot_automation

    def snapshot_then_drain(*args, **kwargs):
        # admit_automation has passed its first admission check; /restart begins before the write.
        result = snapshot(*args, **kwargs)
        runner._draining = True
        return result

    monkeypatch.setattr(automation, "snapshot_automation", snapshot_then_drain)
    with pytest.raises(WakeNotAccepted) as refused:
        await asyncio.wait_for(deliver_wake(adapter, text=WAKE, source=source), 10)
    assert refused.value.reason == "runtime_draining" and not refused.value.permanent
    assert admissions(runner, authority, source) == [("messaging", "terminal")]
    assert executed == ["A"]

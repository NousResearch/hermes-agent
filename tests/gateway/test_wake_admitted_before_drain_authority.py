"""A wake receipted before a restart drain must run exactly once on the authority route too.

Once a chat has a committed human turn, ``deliver_wake`` commits a Kanban wake to the session
authority's FIFO (``admit_route_wake``) instead of the classic adapter path the sibling file covers.
The receipt lets the notifier advance its cursor, so the wake is never re-sent: a drain must leave
the committed row queued (or run it), and the restarted gateway's startup recovery must run it, never
twice. Real GatewayRunner, DiscordAdapter and SessionAuthority over SQLite; the restart is a second
runner and authority over the same state DB.
"""

import asyncio

import pytest

from tests.waiting import async_wait_until

WAKE = "[Kanban task completed] t_1"


class Hold:
    """Keeps the human turn running until ``finish`` is set."""

    def __init__(self):
        self.running, self.finish = asyncio.Event(), asyncio.Event()


async def authority_route(monkeypatch, instance_id, executed, hold=None):
    from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
    from hermes_agent.gateway.platforms.base import SendResult
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.session_results import execution_result
    from hermes_agent.plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    monkeypatch.setattr(adapter, "send", send)
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id=instance_id)
    runner._wire_adapter_handlers(adapter)

    async def turn(event, source, quick_key, run_generation):
        executed.append((instance_id, event.text))
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
    from hermes_agent.hermes_state_runtime import list_session_admissions

    owner = runner.session_store.peek_session_id(runner.session_store._generate_session_key(source))
    rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
    return [(row["principal_id"].split(":")[0], row["status"], row.get("outcome")) for row in rows]


def wake_runs(executed):
    return [owner for owner, text in executed if text == WAKE]


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_authority_route_runs_a_wake_receipted_before_drain_once_after_restart(monkeypatch, busy):
    from hermes_agent.gateway.platforms.event import MessageEvent
    from hermes_agent.gateway.run_runtime import recover_profile_native_sessions, settle_gateway_runtime
    from hermes_agent.gateway.wake import deliver_wake

    executed, hold = [], Hold() if busy else None
    runner, adapter, authority, source = await authority_route(monkeypatch, "owner", executed, hold)
    await adapter.handle_message(MessageEvent(text="A", source=source, message_id="A"))
    if hold is not None:
        await asyncio.wait_for(hold.running.wait(), 10)
    else:
        await async_wait_until(
            lambda: admissions(runner, authority, source) == [("messaging", "terminal", "completed")],
            what="the human turn to settle",
        )
    await asyncio.wait_for(deliver_wake(adapter, text=WAKE, source=source), 10)  # receipt: no retry
    runner._draining = True  # /restart begins; a claim whose guard already passed may still commit
    if hold is not None:
        hold.finish.set()
    # A real stop joins the old owner's authority tasks before the successor can take the profile: a
    # wake claimed just before the drain runs (and settles) here, not under the successor's epoch.
    await settle_gateway_runtime(runner)
    assert admissions(runner, authority, source)[0] == ("messaging", "terminal", "completed")
    wake_row = admissions(runner, authority, source)[-1]
    assert wake_row[0] == "automation"
    assert wake_row[1] == "queued" or (wake_row[1:] == ("terminal", "completed") and wake_runs(executed) == ["owner"])

    successor, _adapter, successor_authority, _source = await authority_route(monkeypatch, "successor", executed)
    await recover_profile_native_sessions(successor, successor_authority)  # a real gateway's startup step
    await async_wait_until(
        lambda: admissions(successor, successor_authority, source)[-1] == ("automation", "terminal", "completed"),
        what="the wake to run",
    )
    assert len(wake_runs(executed)) == 1  # ran once, on whichever gateway claimed it


@pytest.mark.asyncio
async def test_a_wake_claimed_just_before_a_drain_runs_rather_than_settling_unrun(monkeypatch):
    """The drain begins after the wake's claim committed (its admission gate already passed): the
    old owner holds the only claim, so it must run the wake, not answer it with the drain notice."""
    from hermes_agent.gateway.platforms.event import MessageEvent
    from hermes_agent.gateway.run_runtime import settle_gateway_runtime
    from hermes_agent.gateway.wake import deliver_wake

    executed = []
    runner, adapter, authority, source = await authority_route(monkeypatch, "owner", executed)
    await adapter.handle_message(MessageEvent(text="A", source=source, message_id="A"))
    await async_wait_until(
        lambda: admissions(runner, authority, source) == [("messaging", "terminal", "completed")],
        what="the human turn to settle",
    )
    stamp = authority._stamp_claim

    def drain_begins_after_commit(ref, live, row):
        if row:
            runner._draining = True
        return stamp(ref, live, row)

    monkeypatch.setattr(authority, "_stamp_claim", drain_begins_after_commit)
    await asyncio.wait_for(deliver_wake(adapter, text=WAKE, source=source), 10)
    await settle_gateway_runtime(runner)
    assert admissions(runner, authority, source)[-1] == ("automation", "terminal", "completed")
    assert wake_runs(executed) == ["owner"]

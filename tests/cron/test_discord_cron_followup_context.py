"""Continuable Discord briefs reach admitted follow-ups, not an active turn.

Real scheduler, Discord send, SQLite persistence, busy routing, queue recursion,
and AIAgent turn execution against a temporary HERMES_HOME. Discord and model
transport boundaries are offline doubles; no external service is contacted.
"""

import asyncio
import copy
import threading
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from plugins.platforms.discord.adapter import DiscordAdapter
from run_agent import AIAgent

BRIEF = "Scheduled briefing: widget validation is broken; open an issue for it."
FOLLOWUP = "Aren't you opening an issue?"


@pytest.fixture(params=["dm", "thread"])
def conversation(tmp_path, monkeypatch, request):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  supports_vision: true\nagent:\n  image_input_mode: native\ncron:\n  wrap_response: false\n"
        "platforms:\n  discord:\n    enabled: true\n    token: offline-test\n",
        encoding="utf-8",
    )
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, token="offline-test")
    runner = GatewayRunner(config=config)
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="1001", chat_type=request.param, user_id="2001",
        thread_id="1001" if request.param == "thread" else None,
    )
    session = runner.session_store.get_or_create_session(source)
    history = [
        {"role": "user", "content": "Review the README pull request."},
        {"role": "assistant", "content": "The README pull request only needs a wording change."},
    ]
    for message in history:
        runner.session_store.append_to_transcript(session.session_id, message)

    # DiscordAdapter.send and create_handoff_thread remain real. DMs cannot
    # host a child thread; an explicitly targeted existing thread is retained.
    import discord

    channel = MagicMock(spec=discord.DMChannel if request.param == "dm" else discord.Thread)
    channel.id = int(source.chat_id)
    channel.send = AsyncMock(return_value=SimpleNamespace(id=3001))
    adapter = DiscordAdapter(config.platforms[Platform.DISCORD])
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    adapter.set_session_store(runner.session_store)
    runner.adapters[Platform.DISCORD] = adapter
    assert get_hermes_home() == tmp_path
    return runner, source, session, adapter, channel, history


async def deliver(conversation, attach):
    runner, source, session, adapter, channel, history = conversation
    job = {"id": "offline-brief", "name": "Widget briefing", "deliver": "origin", "origin": asdict(source)}
    job["origin"]["platform"] = source.platform.value
    if attach is not None:
        job["attach_to_session"] = attach
    error = await asyncio.to_thread(
        _deliver_result, job, BRIEF, {Platform.DISCORD: adapter}, asyncio.get_running_loop(),
    )
    assert error is None
    channel.send.assert_awaited_once()
    assert channel.send.await_args.kwargs["content"] == BRIEF
    reply_session = runner.session_store.get_or_create_session(source)
    assert reply_session.session_id == session.session_id
    return runner.session_store._db.get_messages(reply_session.session_id, include_inactive=True)


def make_agent(conversation, monkeypatch, respond):
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=respond(kwargs), tool_calls=None,
                                        reasoning=None, reasoning_content=None),
                finish_reason="stop")], usage=None, model="test-model")

    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])
    runner, source, session, *_ = conversation
    agent = AIAgent(model="test-model", api_key="offline", base_url="http://localhost:1/v1",
                    platform="discord", session_id=session.session_id, session_db=runner.session_store._db,
                    max_iterations=2, quiet_mode=True, skip_memory=True, skip_context_files=True)
    agent._disable_streaming = True
    agent._cached_system_prompt = "Offline test system prompt; preserve these bytes."
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("attach", [None, False, True])
@pytest.mark.parametrize("boundary", ["same", "compressed"])
@pytest.mark.parametrize("input_kind", ["string", "text-list", "image", "image-override", "image-text-override"])
async def test_idle_followup_observes_only_opted_in_delivery(conversation, attach, boundary, input_kind, monkeypatch, tmp_path):
    from hermes_state import SessionDB

    runner, source, session, *_ = conversation
    db = runner.session_store._db
    # Same chat / different user and a sibling thread must never contribute context.
    for other in (
        SessionSource(platform=Platform.DISCORD, chat_id=source.chat_id, chat_type="group", user_id="9999"),
        SessionSource(platform=Platform.DISCORD, chat_id="1002", chat_type="thread", thread_id="1002", user_id="2001"),
    ):
        entry = runner.session_store.get_or_create_session(other)
        db.append_pending_delivery(entry.session_id, "UNRELATED PRIVATE BRIEF", source="cron")
    foreign = SessionDB(tmp_path / "foreign-profile" / "state.db")
    foreign.create_session(session.session_id, source="discord")
    foreign.append_pending_delivery(session.session_id, "FOREIGN PROFILE BRIEF", source="cron")
    # The existing lease key, rather than all parent links, defines continuity.
    for reason, config in (("reset", {}), ("compression", {"_branched_from": "compression-parent"})):
        parent, child = reason + "-parent", reason + "-child"
        db.create_session(parent, source="discord")
        db.append_pending_delivery(parent, "UNRELATED ANCESTOR BRIEF", source="cron")
        db.end_session(parent, reason)
        db.create_session(child, source="discord", parent_session_id=parent, model_config=config)
        assert db.pending_deliveries(child) == []
    transcript = await deliver(conversation, attach)
    assert (BRIEF in str(transcript)) is (attach is True)
    if boundary == "compressed":
        parent = session.session_id
        db.publish_compression_child(parent_session_id=parent, child_session_id="compressed-child",
                                     source="discord", messages=runner.session_store.load_transcript(parent),
                                     require_compression_lease=False)
        assert runner.session_store.advance_compression_session(session.session_key, parent, "compressed-child")
    requests = []
    agent = make_agent(conversation, monkeypatch, lambda kw: requests.append(copy.deepcopy(kw)) or "Done.")
    history = runner.session_store.load_transcript(session.session_id)
    if attach is True:
        acknowledge = db._consume_pending_deliveries

        def fail_after_ack(conn, sid, messages):
            acknowledge(conn, sid, messages)
            raise RuntimeError("offline injected transaction failure")

        from agent.turn_facade_lease import attach_pending_delivery_context
        wire_text, metadata = attach_pending_delivery_context(agent, FOLLOWUP, None)
        with monkeypatch.context() as patcher:
            patcher.setattr(db, "_consume_pending_deliveries", fail_after_ack)
            with pytest.raises(RuntimeError, match="offline injected transaction failure"):
                db.append_messages_batch(session.session_id, [{
                    "role": "user", "content": FOLLOWUP, "api_content": wire_text, "display_metadata": metadata,
                }])
        assert runner.session_store.load_transcript(session.session_id) == history
        assert BRIEF in str(db.pending_deliveries(session.session_id)), "ack must roll back with the user row"
        if source.chat_type == "dm" and boundary == "same" and input_kind == "string":
            db.create_session("bounded-references", source="discord")
            bounded_agent = SimpleNamespace(_session_db=db, session_id="bounded-references")
            for index in range(17):
                db.append_pending_delivery(bounded_agent.session_id, f"reference-{index}", source="cron")
            wire, bounded_metadata = attach_pending_delivery_context(bounded_agent, FOLLOWUP, None)
            assert len(bounded_metadata["pending_delivery_ids"]) == 16
            db.append_messages_batch(bounded_agent.session_id, [{
                "role": "user", "content": FOLLOWUP, "api_content": wire, "display_metadata": bounded_metadata,
            }])
            assert [row["content"] for row in db.pending_deliveries(bounded_agent.session_id)] == ["reference-16"]
            db.create_session("oversized-reference", source="discord")
            bounded_agent.session_id = "oversized-reference"
            db.append_pending_delivery(bounded_agent.session_id, "x" * 32001, source="cron")
            wire, _ = attach_pending_delivery_context(bounded_agent, FOLLOWUP, None)
            assert "x" * 32000 in wire and "x" * 32001 not in wire
            assert "truncated" in wire
            assert db.pending_deliveries(bounded_agent.session_id)[0]["size"] == 32001
    user_input = FOLLOWUP if input_kind == "string" else [{"type": "text", "text": FOLLOWUP}]
    if input_kind.startswith("image"):
        user_input.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN1sAAAAASUVORK5CYII="}})
    original_input = copy.deepcopy(user_input)
    override = [{"type": "text", "text": "Original display caption"}] if input_kind == "image-override" else None
    if input_kind == "image-text-override":
        override = "Original display caption"
    result = await asyncio.to_thread(agent.run_conversation, user_input, conversation_history=history,
                                    persist_user_message=override)
    assert not result.get("failed"), result
    assert user_input == original_input
    assert (BRIEF in str(requests)) is (attach is True)
    if input_kind.startswith("image"):
        assert "image_url" in str(requests[-1]["messages"][-1]["content"])
    reloaded = runner.session_store.load_transcript(session.session_id)
    carrier = next(m for m in reversed(reloaded) if m["role"] == "user")
    assert BRIEF not in str(carrier["content"])
    display_overridden = override and (isinstance(override, list) or attach is True)
    assert ("Original display caption" if display_overridden else FOLLOWUP) in str(carrier["content"])
    assert (BRIEF in str(carrier.get("api_content"))) is (attach is True)
    sent = copy.deepcopy(requests[-1]["messages"])
    again = await asyncio.to_thread(agent.run_conversation, "Thanks.", conversation_history=reloaded)
    assert not again.get("failed"), again
    if attach is True:
        assert requests[-1]["messages"][:len(sent)] == sent
        assert str(requests[-1]).count(BRIEF) == 1
    assert agent._cached_system_prompt == "Offline test system prompt; preserve these bytes."
    assert "UNRELATED" not in str(requests) and "FOREIGN PROFILE" not in str(requests)
    assert foreign.pending_deliveries(conversation[2].session_id if boundary == "same" else parent)
    foreign.close()
    assert db.pending_deliveries(session.session_id) == []
    assert all(a["role"] != b["role"] for a, b in zip(requests[-1]["messages"], requests[-1]["messages"][1:]))


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["normal", "priority", "steer", "priority-steer", "slash-steer"])
@pytest.mark.parametrize("runtime", ["chat", "native"])
async def test_active_followup_observes_opted_in_delivery(conversation, route, runtime, monkeypatch):
    runner, source, session, adapter, channel, history = conversation
    entered, finish = threading.Event(), threading.Event()
    requests = []

    def respond(kwargs):
        requests.append(copy.deepcopy(kwargs))
        if len(requests) == 1:
            entered.set()
            assert finish.wait(30), "test did not release active request"
        return "README review complete." if len(requests) == 1 else "Briefing follow-up complete."

    agent = make_agent(conversation, monkeypatch, respond)
    if runtime == "native":
        from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult

        def native_turn(_session, user_input, **kwargs):
            response = respond({"user_input": user_input})
            tool_id = f"read-{len(requests)}"
            return TurnResult(final_text=response, projected_messages=[
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": tool_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": tool_id, "content": "Unpersisted README tool result."},
                {"role": "assistant", "content": response},
            ], tool_iterations=1, turn_id=tool_id, thread_id="offline-native-thread")

        monkeypatch.setattr(CodexAppServerSession, "run_turn", native_turn)
        monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "offline-native-thread")
        agent.api_mode = "codex_app_server"
    runner._running_agents[session.session_key] = agent
    active = asyncio.create_task(asyncio.to_thread(
        agent.run_conversation, "Continue reviewing that README pull request.",
        conversation_history=runner.session_store.load_transcript(session.session_id),
    ))
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        cached_prefix = copy.deepcopy(requests[0].get("messages"))
        transcript = await deliver(conversation, True)
        assert BRIEF in str(transcript), "confirmed delivery must be durable in this session"
        assert BRIEF not in str(requests)
        event = MessageEvent(text=FOLLOWUP, source=source, message_id="4001")
        assert event.reply_to_message_id is None and event.reply_to_text is None
        if route in ("normal", "steer"):
            outcome = await runner._resolve_busy_steer_or_redirect(
                event, session.session_key, "steer" if route == "steer" else "interrupt", agent)
            assert outcome.effective_mode == "queue"
            assert not outcome.redirected and not outcome.steered
            runner._queue_or_replace_pending_event(session.session_key, event)
        elif route == "priority-steer":
            runner._hm_busy_steer(event, agent, session.session_key)
        elif route == "slash-steer":
            event.text = "/steer " + FOLLOWUP
            reply = await runner._busy_steer_command(event, session.session_key, source)
            assert "next turn" in reply
        else:
            await runner._hm_busy_interrupt(event, source, agent, session.session_key)
        assert agent._drain_pending_redirect() is None
        assert not agent._interrupt_requested
        assert adapter._pending_messages[session.session_key].text == FOLLOWUP
    finally:
        finish.set()
        first = await active
    assert not first.get("failed"), first
    assert BRIEF not in str(first["messages"])

    # Exercise the real queued boundary, which recurses using the live result,
    # not a fresh full-history load. Only turn execution's outer wrapper is replaced.
    async def run_followup(**kwargs):
        return await asyncio.to_thread(agent.run_conversation, kwargs["message"],
                                       conversation_history=kwargs["history"])

    monkeypatch.setattr(runner, "_run_agent", run_followup)
    monkeypatch.setattr(runner, "_run_agent_deliver_first_response", AsyncMock())
    adapter.send_typing = AsyncMock()
    ctx = SimpleNamespace(source=source, session_id=session.session_id, session_key=session.session_key,
                          run_generation=None, _interrupt_depth=0, history=history,
                          _status_thread_metadata={}, context_prompt="", result_holder=[first])
    pending = adapter._pending_messages.pop(session.session_key)
    result = await runner._run_agent_queued_followup(ctx, adapter, pending.text, pending, first, first, None)
    assert not result.get("failed"), result
    if runtime == "chat":
        second = requests[-1]["messages"]
        assert second[:len(cached_prefix)] == cached_prefix
        user_input = second[-1]["content"]
    else:
        user_input = requests[-1]["user_input"]
        second = result["messages"]
        assert second[:len(first["messages"])] == first["messages"]
        assert any(m.get("tool_call_id") == "read-1" for m in second)
    assert BRIEF in user_input and FOLLOWUP in user_input
    assert agent._cached_system_prompt == "Offline test system prompt; preserve these bytes."
    assert all(a["role"] != b["role"] for a, b in zip(second, second[1:]))
    assert not adapter._pending_messages
    reloaded = runner.session_store.load_transcript(session.session_id)
    wire = [m.get("api_content") or m.get("content") or "" for m in reloaded]
    assert "\n".join(wire).count(BRIEF) == 1
    assert "\n".join(wire).count(FOLLOWUP) == 1
    again = await asyncio.to_thread(agent.run_conversation, "Thanks.", conversation_history=reloaded)
    assert not again.get("failed"), again
    last_input = requests[-1]["messages"][-1]["content"] if runtime == "chat" else requests[-1]["user_input"]
    assert BRIEF not in last_input
    assert str([m.get("api_content", m.get("content")) for m in
                runner.session_store.load_transcript(session.session_id)]).count(BRIEF) == 1

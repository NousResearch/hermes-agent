"""Completion-admitted turns own native typing, including the workers they launch."""

import asyncio
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until

pytest.importorskip("discord")


@asynccontextmanager
async def _completion_owner(tmp_path, monkeypatch):
    import aiohttp
    from aiohttp import web
    import discord.http
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import claim_session_input, settle_session_input
    from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner

    runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    adapter.config.typing_indicator = True
    posts = asyncio.Queue()

    async def typing(request):
        posts.put_nowait(request.match_info["channel_id"])
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/channels/{channel_id}/typing", typing)
    server = web.AppRunner(app)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", 0).start()
    host, port = server.addresses[0]
    monkeypatch.setattr(discord.http.Route, "BASE", f"http://{host}:{port}")
    try:
        async with aiohttp.ClientSession() as session:
            http = discord.http.HTTPClient(asyncio.get_running_loop())
            http._global_over = asyncio.Event()
            http._global_over.set()
            http._HTTPClient__session = session
            adapter._client = SimpleNamespace(http=http)
            receipt = await _admit_human(runner, authority, MessageEvent(
                text="human", source=source, message_id="human"))
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row["admission_id"],
                                 generation=row["generation"], outcome="completed")
            yield runner, adapter, authority, source, receipt.ref, posts
    finally:
        await adapter.cancel_background_tasks()
        await server.cleanup()
        authority.db.close()


@asynccontextmanager
async def _recovered_owner(adapter, authority, ref):
    """Reopen the committed database with a fresh runner, adapter, authority and runtime epoch."""
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB
    from plugins.platforms.discord.adapter import DiscordAdapter

    await adapter.cancel_background_tasks()
    runner = GatewayRunner(GatewayConfig())
    recovered_adapter = DiscordAdapter(adapter.config)
    recovered_adapter._client = adapter._client
    runner.adapters = {Platform.DISCORD: recovered_adapter}
    db = SessionDB(authority.db.db_path)
    recovered = await initialize_session_authority(
        runner, profile_id=authority.profile_id, instance_id="recovered-owner", db=db)
    runner._wire_adapter_handlers(recovered_adapter)
    source = recovered_adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    try:
        assert recovered.epoch != authority.epoch and recovered.db is not authority.db
        assert await recovered.recover_native_sessions([(ref.session_id, source, recovered_adapter)]) == {
            ref.session_id: "ready"}
        yield runner, recovered_adapter, recovered, source
    finally:
        await recovered_adapter.cancel_background_tasks()
        db.close()
        await runner._session_db.close()


@asynccontextmanager
async def _workers(monkeypatch, ref, key):
    """Real spawn/continue dispatch and persisted child history; only the model is scripted."""
    import json
    from tests.tools.test_delegate_continue import _ScriptedModel, _completion, _parent
    from tools import async_delegation as delegation
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.delegate_tool import delegate_task

    delegation._reset_for_tests()
    release = threading.Event()
    entered = threading.Event()
    model = _ScriptedModel()
    parent = _parent(session_id=ref.session_id)
    parent.platform = "discord"

    def build(**kwargs):
        child = model(**kwargs)
        if len(model.builds) > 1:
            run = child.run_conversation.side_effect

            def blocked(*args, **kw):
                entered.set()
                assert release.wait(30), "worker was not released"
                return run(*args, **kw)

            child.run_conversation.side_effect = blocked
            child.interrupt.side_effect = lambda *args, **kw: release.set()
        return child

    monkeypatch.setattr("run_agent.AIAgent", build)
    token = set_current_session_key(key)
    try:
        first = json.loads(await asyncio.to_thread(delegate_task, goal="initial work", background=True,
                                                  parent_agent=parent))
        assert first["status"] == "dispatched", first
        completion = await asyncio.to_thread(_completion, first["delegation_id"])

        async def start(action):
            args = {"action": "continue", "subagent_id": first["subagent_ids"][0], "message": "follow-up"} if (
                action == "continue") else {"goal": "follow-up"}
            result = json.loads(await asyncio.to_thread(delegate_task, **args, background=True, parent_agent=parent))
            assert result["status"] == "dispatched", result
            assert (result["subagent_ids"] == first["subagent_ids"]) is (action == "continue")
            assert await asyncio.to_thread(entered.wait, 5)
            assert delegation.has_live_for_session(session_key=key)

        yield completion, start, release
    finally:
        release.set()
        if delegation._executor is not None:
            await asyncio.to_thread(delegation._executor.shutdown, wait=True)
        reset_current_session_key(token)
        parent._session_db.close()
        delegation._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["spawn", "continue"])
@pytest.mark.parametrize("restart", [False, True], ids=["live", "recovered"])
async def test_completion_keeps_native_typing_until_worker_finishes(tmp_path, monkeypatch, action, restart):
    from unittest.mock import AsyncMock
    from gateway.platforms.base import SendResult
    from gateway.session_ingress import deliver_settled, execute_admission
    from hermes_state_runtime import claim_session_input, list_session_admissions, settle_session_input
    from tests.gateway.test_completion_admission import pending

    async with (_completion_owner(tmp_path, monkeypatch) as (runner, adapter, authority, source, ref, posts),
                AsyncExitStack() as recovery):
        key = adapter._source_session_key(source)
        async with _workers(monkeypatch, ref, key) as (completion, start, release):
            group = [completion]
            if action == "spawn":
                group.append(dict(pending(key, "batch-sibling"), parent_session_id=ref.session_id))
            assert await runner._deliver_async_delegation_group(group) is True
            committed = list_session_admissions(authority.db, session_id=ref.session_id)[-1]
            assert committed["status"] == "queued" and "automation" in committed["payload"]["native_text_v1"]
            if restart:
                runner, adapter, authority, source = await recovery.enter_async_context(
                    _recovered_owner(adapter, authority, ref))
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
            assert row["admission_id"] == committed["admission_id"] and row["payload"] == committed["payload"]
            adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))

            async def handler(event):
                await start(action)
                assert await asyncio.wait_for(posts.get(), 8) == source.chat_id
                return "follow-up dispatched"

            monkeypatch.setattr(runner, "_handle_message", handler)
            await execute_admission(authority, ref, row)
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row["admission_id"],
                                 generation=row["generation"], outcome="completed")
            await deliver_settled(authority, row["admission_id"])
            assert source.chat_id in adapter._typing_tasks, "completion lost its worker's typing owner"
            assert await asyncio.wait_for(posts.get(), 8) == source.chat_id
            # Neither the producer replay nor the settlement replay sends a second reply.
            assert await runner._deliver_async_delegation_group(group) is None
            await deliver_settled(authority, row["admission_id"])
            adapter.send.assert_awaited_once()
            assert len(list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)) == 2
            assert not authority.producer_turns
            release.set()
            await async_wait_until(lambda: not adapter._typing_tasks, timeout=8,
                                   what="completion worker retired native typing")
            assert not adapter._typing_owners


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [
    "failed", "interrupted", "raised", "cancelled", "delivery_failed", "stop", "new", "shutdown", "no_worker", "disabled",
])
async def test_completion_teardown_retires_native_typing(tmp_path, monkeypatch, ending):
    from unittest.mock import AsyncMock
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress import deliver_settled, execute_admission
    from gateway.session_results import execution_result
    from hermes_state_runtime import claim_session_input, settle_session_input

    async with _completion_owner(tmp_path, monkeypatch) as (runner, adapter, authority, source, ref, posts):
        key = adapter._source_session_key(source)
        async with _workers(monkeypatch, ref, key) as (completion, start, _release):
            adapter.config.typing_indicator = ending != "disabled"
            assert await runner._deliver_async_delegation_group([completion]) is True
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
            adapter.send = AsyncMock(return_value=SendResult(success=ending != "delivery_failed", message_id="reply"))
            entered = asyncio.Event()

            async def handler(event):
                if ending != "no_worker":
                    await start("continue")
                if ending != "disabled":
                    assert await asyncio.wait_for(posts.get(), 8) == source.chat_id
                entered.set()
                if ending == "cancelled":
                    await asyncio.Event().wait()
                if ending == "raised":
                    raise RuntimeError("completion failed")
                if ending in {"failed", "interrupted"}:
                    execution_result.get()["result"] = {ending: True}
                return "follow-up dispatched"

            monkeypatch.setattr(runner, "_handle_message", handler)
            executing = asyncio.create_task(execute_admission(authority, ref, row))
            try:
                await asyncio.wait_for(entered.wait(), 10)
                if ending == "cancelled":
                    executing.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await executing
                elif ending == "raised":
                    with pytest.raises(RuntimeError, match="completion failed"):
                        await executing
                else:
                    await executing
                outcome = {"failed": "failed", "raised": "failed", "cancelled": "interrupted",
                           "interrupted": "interrupted"}.get(ending, "completed")
                settle_session_input(authority.db, epoch=authority.epoch, admission_id=row["admission_id"],
                                     generation=row["generation"], outcome=outcome)
                await deliver_settled(authority, row["admission_id"])
                if ending in {"stop", "new", "shutdown"}:
                    assert source.chat_id in adapter._typing_tasks
                    if ending == "shutdown":
                        await adapter.cancel_background_tasks()
                    elif ending == "stop":
                        await runner._handle_stop_command(MessageEvent(text="/stop", source=source))
                    else:
                        await runner._handle_reset_command(MessageEvent(text="/new", source=source))
                assert not adapter._typing_tasks, "completion teardown left a native transport running"
                assert not adapter._typing_owners
                assert not authority.producer_turns
                if ending == "disabled":
                    assert posts.empty()
            finally:
                executing.cancel()
                await asyncio.gather(executing, return_exceptions=True)

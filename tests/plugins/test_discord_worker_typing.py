"""Native Discord typing must cover the whole conversation's live work."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["refresh", "stop", "cancel_stop"])
async def test_real_http_global_backoff_does_not_strand_other_sends(monkeypatch, ending):
    import discord.http

    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    http = discord.http.HTTPClient(asyncio.get_running_loop())
    http._global_over = asyncio.Event()
    http._global_over.set()
    backoff_started = asyncio.Event()
    backoff_finished = asyncio.Event()
    real_sleep = asyncio.sleep

    async def sleep(delay):
        backoff_started.set()
        await real_sleep(delay)
        backoff_finished.set()

    monkeypatch.setattr(discord.http, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if not name.startswith("_")},
    ))
    monkeypatch.setattr(discord.http.asyncio, "sleep", sleep)
    calls = []
    request_tasks = set()

    class Response:
        def __init__(self, status, body):
            self.status, self.body = status, body
            self.reason = "test"
            self.headers = {"content-type": "application/json", "Via": "test"}

        async def text(self, **kwargs):
            return json.dumps(self.body)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def request(method, url, **kwargs):
        request_tasks.add(asyncio.current_task())
        calls.append(url)
        if len(calls) == 1:
            return Response(429, {"retry_after": 4.2, "global": True})
        assert backoff_finished.is_set(), "request escaped the allowed global backoff"
        return Response(200, {"id": "sent"})

    http._HTTPClient__session = SimpleNamespace(request=request)
    adapter._client = SimpleNamespace(http=http)
    stopping = None
    send = None
    try:
        await adapter.send_typing("100")
        typing = adapter._typing_tasks["100"]
        await asyncio.wait_for(backoff_started.wait(), 5)
        send = asyncio.create_task(http.request(discord.http.Route(
            "POST", "/channels/{channel_id}/messages", channel_id="200")))
        if ending != "refresh":
            stopping = asyncio.create_task(adapter.stop_typing("100"))
            await real_sleep(0)
            if ending == "cancel_stop":
                stopping.cancel()
                await real_sleep(0)
                stopping.cancel()
        # Includes the old 4s outer deadline and Discord's allowed 4.2s backoff.
        done, _ = await asyncio.wait({send}, timeout=7)
        assert send in done, "typing cancellation stranded Discord's global HTTP gate"
        assert send.result() == {"id": "sent"}
        assert backoff_finished.is_set()
        if stopping is not None:
            await asyncio.wait_for(asyncio.gather(stopping, return_exceptions=True), 5)
            assert typing.done()
            assert not adapter._typing_tasks
    finally:
        if send is not None:
            send.cancel()
            await asyncio.gather(send, return_exceptions=True)
        await adapter.cancel_background_tasks()
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
    assert all(task.done() for task in request_tasks), "HTTP cleanup left an orphan request"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["successor", "bounded_shutdown", "fatal_disconnect"])
async def test_real_http_backoff_hands_off_typing_ownership(monkeypatch, typing_clock, ending):
    import discord.http
    from gateway.platforms.event import MessageEvent
    from tools import process_registry as processes

    ticks, resume = typing_clock
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    http = discord.http.HTTPClient(asyncio.get_running_loop())
    http._global_over = asyncio.Event()
    http._global_over.set()
    entered, release = asyncio.Event(), asyncio.Event()

    async def backoff(delay):
        entered.set()
        await release.wait()

    monkeypatch.setattr(discord.http, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if not name.startswith("_")},
    ))
    monkeypatch.setattr(discord.http.asyncio, "sleep", backoff)
    calls = []

    class Response:
        reason = "test"
        headers = {"content-type": "application/json", "Via": "test"}

        def __init__(self, status, body):
            self.status, self.body = status, body

        async def text(self, **kwargs):
            return json.dumps(self.body)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def request(method, url, **kwargs):
        calls.append((method, url, asyncio.current_task()))
        if len(calls) == 1:
            return Response(429, {"retry_after": 20, "global": True})
        assert release.is_set(), "request escaped the allowed global backoff"
        return Response(204, {})

    http._HTTPClient__session = SimpleNamespace(request=request)
    adapter._client = SimpleNamespace(http=http, close=AsyncMock())
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    event = MessageEvent(text="work", source=adapter.build_source(
        chat_id="100", chat_type="group", user_id="1"))
    successor = MessageEvent(text="work", source=adapter.build_source(
        chat_id="100", chat_type="group", user_id="2"))
    key = adapter._event_session_key(event)
    successor_key = adapter._event_session_key(successor)
    assert key != successor_key
    old = processes.ProcessSession(id="old", command="build", session_key=key, notify_on_complete=True)
    new = processes.ProcessSession(id="new", command="test", session_key=successor_key, notify_on_complete=True)
    stopping = None
    try:
        async def first_handler(event):
            if ending != "successor":
                registry._running[old.id] = old
            await asyncio.wait_for(entered.wait(), 5)

        adapter.set_message_handler(first_handler)
        await adapter.handle_message(event)
        if ending == "successor":
            stopping = adapter._session_tasks[key]
            await asyncio.wait_for(entered.wait(), 5)
            async with asyncio.timeout(5):
                while not adapter._typing_tasks["100"].cancelling():
                    await asyncio.sleep(0)
        else:
            await asyncio.gather(*list(adapter._background_tasks))
        old_typing = adapter._typing_tasks["100"]
        if ending != "successor":
            from gateway.config import Platform
            from gateway.run_adapters import GatewayAdapterLifecycleMixin

            lifecycle = GatewayAdapterLifecycleMixin()
            monkeypatch.setattr(lifecycle, "_adapter_disconnect_timeout_secs", lambda: 0.05)
            client = adapter._client
            if ending == "bounded_shutdown":
                await lifecycle._bounded_adapter_teardown(adapter, Platform.DISCORD)
            else:
                await lifecycle._safe_adapter_disconnect(adapter, Platform.DISCORD)
            assert not old_typing.done()
            assert not http._global_over.is_set()
            release.set()
            async with asyncio.timeout(5):
                while adapter._client is not None:
                    await asyncio.sleep(0.01)
            assert old_typing.done()
            assert not adapter._typing_tasks
            assert not adapter._typing_owners
            assert http._global_over.is_set()
            client.close.assert_awaited_once()
            await adapter.send_typing("100")
            assert not adapter._typing_tasks
            return

        assert not stopping.done()

        async def successor_handler(event):
            registry._running[new.id] = new
            await adapter.send_typing("100")
            await asyncio.sleep(0)

        adapter.set_message_handler(successor_handler)
        await adapter.handle_message(successor)
        await adapter._session_tasks[successor_key]
        assert registry.has_completion_work_for_session(successor_key)
        assert adapter._typing_owners["100"][successor_key][0].done()
        assert adapter._typing_active("100")
        assert adapter._typing_tasks["100"] is old_typing
        assert not http._global_over.is_set()
        release.set()
        await asyncio.wait_for(stopping, 5)
        assert "100" in adapter._typing_tasks, "live successor worker lost native typing"
        replacement = adapter._typing_tasks["100"]
        assert replacement is not old_typing
        assert old_typing.done()
        assert http._global_over.is_set()
        # The old stop finalizer must preserve the replacement, which must POST
        # through the real HTTPClient, not merely occupy the task registry.
        await asyncio.wait_for(ticks.get(), 5)
        assert any(method == "POST" and url.endswith("/channels/100/typing")
                   and task is not calls[0][2] for method, url, task in calls)
        new.exited = True
        resume.put_nowait(None)
        await asyncio.wait_for(replacement, 5)
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        old.exited = new.exited = True
        release.set()
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        await adapter.cancel_background_tasks()
    assert all(task.done() for _, _, task in calls)


@pytest.mark.asyncio
async def test_stop_before_first_request_releases_typing_task():
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    await adapter.send_typing("100")
    await adapter.stop_typing("100")
    assert not adapter._typing_tasks
    assert not adapter._typing_owners
    adapter._client.http.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_prestart_stop_hands_off_to_successor_worker(monkeypatch, typing_clock):
    from gateway.platforms.event import MessageEvent
    from tools import process_registry as processes

    ticks, resume = typing_clock
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    request = AsyncMock()
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))
    event = MessageEvent(text="work", source=adapter.build_source(chat_id="100", user_id="2"))
    key = adapter._event_session_key(event)
    registry = processes.ProcessRegistry()
    worker = processes.ProcessSession(id="new", command="build", session_key=key, notify_on_complete=True)
    registry._running[worker.id] = worker
    monkeypatch.setattr(processes, "process_registry", registry)
    try:
        await adapter.send_typing("100")
        old_typing = adapter._typing_tasks["100"]
        # Admit the successor while stop awaits the never-started transport.
        asyncio.get_running_loop().call_soon(
            adapter._start_typing_refresh, event, asyncio.Event(), None)
        await adapter.stop_typing("100")
        assert old_typing.cancelled()
        request.assert_not_awaited()
        assert "100" in adapter._typing_tasks, "prestart cleanup lost the successor"
        replacement = adapter._typing_tasks["100"]
        assert replacement is not old_typing
        refresh = adapter._typing_owners["100"][key][0]
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        await asyncio.wait_for(ticks.get(), 5)
        request.assert_awaited_once()
        worker.exited = True
        resume.put_nowait(None)
        await asyncio.wait_for(replacement, 5)
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        worker.exited = True
        await adapter.cancel_background_tasks()


@pytest.fixture
def typing_clock(monkeypatch):
    import plugins.platforms.discord.adapter_typing as module

    ticks, resume = asyncio.Queue(), asyncio.Queue()

    async def sleep(delay):
        ticks.put_nowait(delay)
        await resume.get()

    monkeypatch.setattr(module, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if not name.startswith("_")},
    ))
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    return ticks, resume


@pytest.mark.asyncio
async def test_native_transport_honors_pause(typing_clock):
    ticks, resume = typing_clock
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    request = AsyncMock()
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))
    try:
        await adapter.send_typing("100")
        await asyncio.wait_for(ticks.get(), 5)
        adapter.pause_typing_for_chat("100")
        before = request.await_count
        resume.put_nowait(None)
        await asyncio.wait_for(ticks.get(), 5)
        assert request.await_count == before
        adapter.resume_typing_for_chat("100")
        resume.put_nowait(None)
        await asyncio.wait_for(ticks.get(), 5)
        assert request.await_count == before + 1
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_stalled_native_request_is_bounded(monkeypatch):
    import aiohttp
    from aiohttp import web
    import discord.http

    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    # Exercise aiohttp's actual I/O deadline, not a mock of HTTPClient.request.
    monkeypatch.setattr(adapter, "_TYPING_REQUEST_TIMEOUT", 2.0, raising=False)
    entered, release = asyncio.Event(), asyncio.Event()

    async def stalled(request):
        entered.set()
        await release.wait()
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/channels/{channel_id}/typing", stalled)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0]
    monkeypatch.setattr(discord.http.Route, "BASE", f"http://{host}:{port}")
    try:
        async with aiohttp.ClientSession() as session:
            http = discord.http.HTTPClient(asyncio.get_running_loop())
            http._global_over = asyncio.Event()
            http._global_over.set()
            http._HTTPClient__session = session
            adapter._client = SimpleNamespace(http=http)
            await adapter.send_typing("100")
            task = adapter._typing_tasks["100"]
            await asyncio.wait_for(entered.wait(), 5)
            done, _ = await asyncio.wait({task}, timeout=5)
            assert task in done, "stalled network I/O exceeded the typing deadline"
            assert not adapter._typing_tasks
    finally:
        await adapter.cancel_background_tasks()
        release.set()
        await runner.cleanup()



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "heartbeat"])
async def test_suppressed_turn_creates_no_native_typing(mode):
    from gateway.platforms.event import MessageEvent

    adapter = DiscordAdapter(PlatformConfig(enabled=True, typing_indicator=mode != "disabled"))
    request = AsyncMock()
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))
    event = MessageEvent(text="work", source=adapter.build_source(chat_id="100", user_id="1"))
    if mode == "heartbeat":
        event._heartbeat_session_id = "scheduled-heartbeat"
    adapter.set_message_handler(AsyncMock(return_value=None))
    await adapter.handle_message(event)
    await asyncio.gather(*list(adapter._background_tasks))
    request.assert_not_awaited()
    assert not adapter._typing_owners
    assert not adapter._typing_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rate_limit", "error", "cancel"])
async def test_native_transport_failure_and_shutdown_leave_no_task(typing_clock, failure):
    import discord

    ticks, resume = typing_clock
    entered, exited = asyncio.Event(), asyncio.Event()
    adapter = DiscordAdapter(PlatformConfig(enabled=True))

    async def request(route, **kwargs):
        entered.set()
        try:
            if failure == "rate_limit":
                raise discord.RateLimited(2.5)
            if failure == "error":
                raise RuntimeError("transport unavailable")
            await asyncio.Event().wait()
        finally:
            exited.set()

    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))
    try:
        await adapter.send_typing("100")
        task = adapter._typing_tasks["100"]
        await asyncio.wait_for(entered.wait(), 5)
        if failure == "rate_limit":
            assert await asyncio.wait_for(ticks.get(), 5) >= 2.5
        if failure == "error":
            await asyncio.wait_for(task, 5)
            assert not adapter._typing_tasks
            adapter._client.http.request = AsyncMock()
            await adapter.send_typing("100")
            await asyncio.wait_for(ticks.get(), 5)
            adapter._client.http.request.assert_awaited_once()
        await adapter.cancel_background_tasks()
        assert exited.is_set()
        assert task.done()
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_overlapping_turns_keep_profile_and_thread_owners(tmp_path, monkeypatch):
    import threading
    from agent import secret_scope
    from gateway.platforms.event import MessageEvent
    from gateway.run import _profile_runtime_scope
    from tools import async_delegation as delegation

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    delegation._reset_for_tests()
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    releases = [threading.Event() for _ in range(3)]
    events, keys = [], []
    for profile, chat in [("a", "100"), ("b", "100"), ("a", "200")]:
        source = adapter.build_source(chat_id=chat, thread_id=chat, chat_type="group", user_id="1")
        source.profile = profile
        event = MessageEvent(text="work", source=source)
        events.append(event)
        keys.append(adapter._event_session_key(event))
    assert len(set(keys)) == len(events)
    index = 0

    async def handler(event):
        release = releases[index]
        result = await asyncio.to_thread(
            delegation.dispatch_async_delegation,
            goal="test", context=None, toolsets=None, role="leaf", model=None,
            session_key=adapter._event_session_key(event),
            runner=lambda: (release.wait(30) and {"status": "completed"}),
            interrupt_fn=release.set)
        assert result["status"] == "dispatched"
        await adapter.send_typing(event.source.chat_id)

    adapter.set_message_handler(handler)
    foreground = asyncio.Event()
    entered = asyncio.Event()
    try:
        # A -> B -> A in one process, with the same channel shared by A and B.
        for index, event in enumerate(events):
            home = tmp_path / event.source.profile
            home.mkdir(exist_ok=True)
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                await adapter.handle_message(event)
                await asyncio.gather(*list(adapter._background_tasks))
        assert set(adapter._typing_tasks) == {"100", "200"}

        async def overlapping_handler(event):
            entered.set()
            await foreground.wait()

        adapter.set_message_handler(overlapping_handler)
        with _profile_runtime_scope(tmp_path / "a", prepared_secret_scope={}):
            await adapter.handle_message(events[0])
        await asyncio.wait_for(entered.wait(), 5)
        releases[0].set()
        async with asyncio.timeout(5):
            while delegation.has_live_for_session(session_key=keys[0]):
                await asyncio.sleep(0.01)
        await adapter.stop_typing("100")
        assert "100" in adapter._typing_tasks  # foreground OR the other profile
        await adapter.interrupt_session_activity(keys[1], "100")
        assert "100" in adapter._typing_tasks  # now only A's foreground
        assert keys[1] not in adapter._typing_owners["100"]
        foreground.set()
        await asyncio.gather(*list(adapter._background_tasks))
        assert "100" not in adapter._typing_tasks
        assert "200" in adapter._typing_tasks  # A's other thread is independent
        await adapter.interrupt_session_activity(keys[2], "200")
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        foreground.set()
        for release in releases:
            release.set()
        if delegation._executor is not None:
            await asyncio.to_thread(delegation._executor.shutdown, wait=True)
        await adapter.cancel_background_tasks()
        delegation._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["interrupt", "shutdown"])
async def test_active_turn_teardown_cannot_revive_native_typing(ending):
    from gateway.platforms.event import MessageEvent

    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    entered, cancelled = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()

    async def request(route, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))
    event = MessageEvent(text="work", source=adapter.build_source(chat_id="100", user_id="1"))
    async def handler(event):
        await release.wait()

    adapter.set_message_handler(handler)
    try:
        await adapter.handle_message(event)
        await asyncio.wait_for(entered.wait(), 5)
        tasks = list(adapter._background_tasks)
        if ending == "interrupt":
            await adapter.interrupt_session_activity(adapter._event_session_key(event), "100")
            release.set()
        else:
            await adapter.cancel_background_tasks()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert cancelled.is_set()
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_detached_worker_typing_recovers_after_transport_error(typing_clock, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from tools import process_registry as processes

    ticks, resume = typing_clock
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    event = MessageEvent(text="work", source=adapter.build_source(chat_id="100", user_id="1"))
    key = adapter._event_session_key(event)
    registry = processes.ProcessRegistry()
    worker = processes.ProcessSession(id="bounded", command="build", session_key=key, notify_on_complete=True)
    registry._running[worker.id] = worker
    monkeypatch.setattr(processes, "process_registry", registry)
    request = AsyncMock(side_effect=[None, RuntimeError("temporary failure"), None])
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))

    async def handler(event):
        await asyncio.wait_for(ticks.get(), 5)

    adapter.set_message_handler(handler)
    try:
        await adapter.handle_message(event)
        await asyncio.gather(*list(adapter._background_tasks))
        task = adapter._typing_tasks["100"]
        resume.put_nowait(None)
        await asyncio.wait_for(ticks.get(), 5)
        assert not task.done()
        resume.put_nowait(None)
        await asyncio.wait_for(ticks.get(), 5)
        assert request.await_count == 3
        worker.exited = True
        resume.put_nowait(None)
        await asyncio.wait_for(task, 5)
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        worker.exited = True
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_completed_worker_exits_during_long_rate_limit(typing_clock, monkeypatch):
    import discord
    from gateway.platforms.event import MessageEvent
    from tools import process_registry as processes

    ticks, resume = typing_clock
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    event = MessageEvent(text="work", source=adapter.build_source(chat_id="100", user_id="1"))
    registry = processes.ProcessRegistry()
    worker = processes.ProcessSession(id="bounded", command="build",
        session_key=adapter._event_session_key(event), notify_on_complete=True)
    registry._running[worker.id] = worker
    monkeypatch.setattr(processes, "process_registry", registry)
    request = AsyncMock(side_effect=discord.RateLimited(60))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request))
    first_tick = asyncio.Queue()

    async def handler(event):
        first_tick.put_nowait(await asyncio.wait_for(ticks.get(), 5))

    adapter.set_message_handler(handler)
    try:
        await adapter.handle_message(event)
        await asyncio.gather(*list(adapter._background_tasks))
        assert await first_tick.get() <= 5, "rate-limit sleep must still poll owner completion"
        task = adapter._typing_tasks["100"]
        resume.put_nowait(None)
        assert await asyncio.wait_for(ticks.get(), 5) <= 5
        request.assert_awaited_once()  # poll ownership, not Discord, before retry_after
        worker.exited = True
        resume.put_nowait(None)
        await asyncio.wait_for(task, 5)
        request.assert_awaited_once()
        assert not adapter._typing_tasks
        assert not adapter._typing_owners
    finally:
        worker.exited = True
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_native_typing_refreshes_before_discord_expiry(monkeypatch):
    import plugins.platforms.discord.adapter_typing as module

    delays = asyncio.Queue()
    release = asyncio.Event()

    async def sleep(delay):
        await delays.put(delay)
        await release.wait()

    monkeypatch.setattr(module, "asyncio", SimpleNamespace(
        **{name: getattr(asyncio, name) for name in dir(asyncio) if not name.startswith("_")},
    ))
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    try:
        await adapter.send_typing("100")
        delay = await asyncio.wait_for(delays.get(), 5)
        assert 0 < delay < 10, "Discord expires typing after about ten seconds"
        await adapter.send_typing("100")
        assert adapter._client.http.request.await_count == 1
    finally:
        await adapter.stop_typing("100")


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_kind", ["delegation", pytest.param("process", marks=pytest.mark.platforms("posix"))])
@pytest.mark.parametrize("ending", ["complete", "shutdown", "disconnect", "reset", "stop_command", "new_command"])
async def test_typing_survives_foreground_and_sibling_completion(tmp_path, monkeypatch, worker_kind, ending):
    import threading
    from gateway.platforms.event import MessageEvent
    from tools import async_delegation as delegation

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    delegation._reset_for_tests()
    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    posts = asyncio.Queue()

    async def request(route, **kwargs):
        posts.put_nowait(route.channel_id)

    adapter._client = SimpleNamespace(http=SimpleNamespace(request=request), close=AsyncMock())
    event = MessageEvent(text="work", source=adapter.build_source(
        chat_id="100", chat_type="group", thread_id="100", user_id="1"))
    key = adapter._event_session_key(event)
    releases = [threading.Event(), threading.Event()]
    process_ids = []
    process_releases = []
    from tools import process_registry as processes
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)

    def dispatch(release):
        if worker_kind == "process":
            import json
            import shlex
            import sys
            from tools.terminal_tool_background import spawn_background_process
            done = tmp_path / f"done-{len(process_ids)}"
            process_releases.append(done)
            script = "import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); deadline=time.monotonic()+60\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(0.01)"
            result = json.loads(spawn_background_process(
                command=shlex.join([sys.executable, "-c", script, str(done)]),
                env=SimpleNamespace(env={}), env_type="local", effective_task_id="test-worker",
                task_id="test-worker", session_key=key, workdir=str(tmp_path), cwd=str(tmp_path),
                effective_pty=False, notify_on_complete=True, watch_patterns=None,
                approval_note=None, pty_disabled_reason=None))
            assert result["notify_on_complete"] is True
            process_ids.append(result["session_id"])
            return {"status": "dispatched"}

        return delegation.dispatch_async_delegation(
            goal="test", context=None, toolsets=None, role="leaf", model=None,
            session_key=key, runner=lambda: (release.wait(20) and {"status": "completed"}),
            interrupt_fn=release.set)

    async def handler(_event):
        for release in releases:
            result = await asyncio.to_thread(dispatch, release)
            assert result["status"] == "dispatched"
        await asyncio.wait_for(posts.get(), 5)

    adapter.set_message_handler(handler)
    try:
        # Real base dispatch and finally, not a direct call to a liveness helper.
        await adapter.handle_message(event)
        await asyncio.gather(*list(adapter._background_tasks))
        assert key not in adapter._session_tasks
        assert "100" in adapter._typing_tasks, "foreground cleanup killed live workers' typing"
        if ending != "complete":
            if ending == "shutdown":
                await adapter.cancel_background_tasks()
            elif ending == "disconnect":
                tasks = list(adapter._typing_tasks.values())
                closing_tasks = []

                async def closing():
                    # Closing may yield to a late refresh while _client still exists.
                    await adapter.send_typing("100")
                    closing_tasks.extend(adapter._typing_tasks.values())

                adapter._client.close.side_effect = closing
                await adapter.disconnect()
                assert not closing_tasks
                assert all(task.done() for task in tasks)
                # A late foreground refresh must not resurrect disconnected owners
                # or transport tasks, even though the detached worker is still live.
                await asyncio.sleep(0)
                await adapter.send_typing("100")
                refresh = adapter._start_typing_refresh(event, asyncio.Event(), None)
                if refresh is not None:
                    refresh.cancel()
                    await asyncio.gather(refresh, return_exceptions=True)
                assert refresh is None
            elif ending == "reset":
                await adapter.interrupt_session_activity(key, "100")
            else:
                from gateway.config import GatewayConfig, Platform
                from gateway.run import GatewayRunner

                runner = GatewayRunner(config=GatewayConfig())
                runner.adapters = {Platform.DISCORD: adapter}
                await runner.async_session_store.get_or_create_session(event.source)
                command = MessageEvent(text="/stop" if ending == "stop_command" else "/new", source=event.source)
                if ending == "stop_command":
                    await runner._handle_stop_command(command)
                else:
                    await runner._handle_reset_command(command)
            assert not adapter._typing_tasks, "idle worker typing survived teardown"
            assert not adapter._typing_owners
            return
        assert await asyncio.wait_for(posts.get(), 8) == "100"
        if worker_kind == "process":
            process_releases[0].touch()
        else:
            releases[0].set()
        assert await asyncio.wait_for(posts.get(), 8) == "100"
        if worker_kind == "process":
            process_releases[1].touch()
        else:
            releases[1].set()
        async with asyncio.timeout(8):
            while "100" in adapter._typing_tasks:
                await asyncio.sleep(0.02)
    finally:
        for release in releases:
            release.set()
        if delegation._executor is not None:
            await asyncio.to_thread(delegation._executor.shutdown, wait=True)
        for process_id in process_ids:
            await asyncio.to_thread(registry.kill_process, process_id)
        await adapter.cancel_background_tasks()
        await adapter.stop_typing("100")
        delegation._reset_for_tests()

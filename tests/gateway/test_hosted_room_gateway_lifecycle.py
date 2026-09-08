"""Messaging-gateway ownership tests for the hosted Group Chat worker."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver, hosted_rooms
from gateway.run import GatewayRunner
from tui_gateway.hosted_room_service import HostedRoomService


class _RPC:
    def __init__(self) -> None:
        self.sessions = {}
        self.submits = []

    def resolve_exact(self, *, profile, title, source):
        del source
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        del source
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        del profile, source
        return {"session_id": session_id}

    def submit(self, **kwargs):
        self.submits.append(kwargs["profile"])
        kwargs["on_terminal"]({
            "status": "settled",
            "text": f"reply from {kwargs['profile']}",
        })
        return {"accepted": True}

    def history(self, **kwargs):
        del kwargs
        return []

    def info(self, **kwargs):
        del kwargs
        return {"active": False, "task_id": None}

    def interrupt(self, **kwargs):
        del kwargs
        raise AssertionError("gateway lifecycle must not interrupt room work")


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def _service(db_path, *, profiles=("default",)):
    service = HostedRoomService(_server(), db_path=db_path)
    rpc = _RPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: profiles
    return service, rpc


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not settle before timeout")


@pytest.mark.asyncio
async def test_messaging_gateway_supervisor_starts_without_dashboard(monkeypatch):
    from tui_gateway import methods_groups

    state = {"running": False, "starts": 0}

    class Runtime:
        def status(self):
            return {"running": state["running"], "stopping": False}

    service = SimpleNamespace(runtime=Runtime())

    def get_service():
        return service if state["running"] else None

    def start_service(**kwargs):
        if not state["running"]:
            state["starts"] += 1
        state["running"] = True
        return service

    monkeypatch.setattr(methods_groups, "get_hosted_room_service", get_service)
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", start_service)

    runner = GatewayRunner.__new__(GatewayRunner)
    started = await runner._ensure_hosted_room_worker()
    assert started is service
    assert state == {"running": True, "starts": 1}

    # A dead child is restarted, while a healthy one is left alone.
    await runner._ensure_hosted_room_worker()
    assert state["starts"] == 1
    state["running"] = False
    await runner._ensure_hosted_room_worker()
    assert state["starts"] == 2


@pytest.mark.asyncio
async def test_dead_room_worker_is_restarted_by_gateway_task_supervision(monkeypatch):
    from tui_gateway import methods_groups

    starts = {"count": 0}

    def fail_start(**kwargs):
        starts["count"] += 1
        if starts["count"] == 3:
            runner._running = False
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(methods_groups, "get_hosted_room_service", lambda: None)
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", fail_start)
    monkeypatch.setattr(GatewayRunner, "_MAX_SUPERVISED_RESTARTS", 1)
    monkeypatch.setattr(
        GatewayRunner,
        "_supervised_backoff",
        staticmethod(lambda _attempt: 0),
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._background_tasks = set()
    runner._spawn_supervised(
        lambda: runner._hosted_room_worker_watcher(interval=0),
        "hosted_room_worker",
    )

    for _ in range(200):
        if starts["count"] == 3 and not runner._background_tasks:
            break
        await asyncio.sleep(0.01)
    runner._running = False

    assert starts["count"] == 3
    assert runner._background_tasks == set()


def test_gateway_restart_resumes_queued_room_for_multiplexed_profile(tmp_path):
    db = tmp_path / "state.db"
    first, _ = _service(db, profiles=("default", "ops"))
    first.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {
                "member_id": "default",
                "profile": "default",
                "handle": "hermes",
            },
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    first.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    assert (
        len(hosted_room_driver.list_tasks(db, room_id="room-1", status="queued")) == 1
    )

    resumed, rpc = _service(db, profiles=("default", "ops"))
    resumed.start()
    try:
        _wait_for(
            lambda: any(
                event["kind"] == "message.member"
                for event in hosted_rooms.read_events(
                    db, room_id="room-1", since_seq=0
                )["events"]
            )
        )
    finally:
        assert resumed.stop(timeout=5.0)

    assert rpc.submits == ["ops"]
    assert hosted_room_driver.list_tasks(db, room_id="room-1", status="settled")


def test_dashboard_and_gateway_workers_share_one_fenced_execution_owner(tmp_path):
    db = tmp_path / "state.db"
    gateway, gateway_rpc = _service(db, profiles=("default", "ops"))
    dashboard, dashboard_rpc = _service(db, profiles=("default", "ops"))
    gateway.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {
                "member_id": "default",
                "profile": "default",
                "handle": "hermes",
            },
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    gateway.send(
        room_id="room-1",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )

    gateway.start()
    dashboard.start()
    try:
        _wait_for(
            lambda: any(
                event["kind"] == "message.member"
                for event in hosted_rooms.read_events(
                    db, room_id="room-1", since_seq=0
                )["events"]
            )
        )
        time.sleep(0.05)
    finally:
        assert gateway.stop(timeout=5.0)
        assert dashboard.stop(timeout=5.0)

    assert len(gateway_rpc.submits) + len(dashboard_rpc.submits) == 1
    events = hosted_rooms.read_events(db, room_id="room-1", since_seq=0)["events"]
    assert sum(event["kind"] == "message.member" for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["gateway", "web"])
async def test_native_owners_recover_companion_without_retargeting(tmp_path, monkeypatch, owner):
    from tui_gateway import methods_groups, hosted_room_service
    from plugins.platforms.telegram import hosted_room_transport as telegram

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service, rpc = _service(tmp_path / "state.db", profiles=("default", "ops"))
    service.create_room(room_id="room", name="Room", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"}])
    monkeypatch.setattr(methods_groups, "_bound_server", _server())
    monkeypatch.setattr(methods_groups, "_service", None)
    monkeypatch.setattr(methods_groups, "_transport", None)
    monkeypatch.setattr(methods_groups, "_binding", None)
    monkeypatch.setattr(hosted_room_service, "HostedRoomService", lambda *a, **kw: service)
    binding = dict(enabled=True, room_id="room", queue_db=str(tmp_path / "queue.db"),
                   chat_id=-123, owner_id=1, control_profile="default",
                   bots={"default": {"id": 1, "username": "hermes_bot"},
                         "ops": {"id": 2, "username": "ops_bot"}})
    binding_path = tmp_path / "binding.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({"gateway": {"hosted_rooms": {
        "telegram": {"binding_file": str(binding_path)}}}}))
    # Configured but invalid must fail closed, then recover on the same owner.
    binding_path.write_text("{}")
    companions = []
    release = threading.Event()

    class Companion(telegram.Transport):
        def __init__(self, *args):
            super().__init__(*args)
            companions.append(self)

        def _run(self):
            if len(companions) == 1:
                self.error = "scripted startup failure"
                self.ready.set()
                assert release.wait(15)
            else:
                self.ready.set()
                assert self.halt.wait(15)

        def stop(self, *, timeout=5.0):
            return super().stop(timeout=min(timeout, 0.02))

    monkeypatch.setattr(telegram, "Transport", Companion)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    watcher = None
    client = None
    pending = []

    def accept(**kwargs):
        pending.append(kwargs["on_terminal"])
        rpc.submits.append(kwargs["profile"])
        return {"accepted": True}

    monkeypatch.setattr(rpc, "submit", accept)

    async def until(predicate):
        async with asyncio.timeout(10):
            while not predicate():
                await asyncio.sleep(0.01)

    try:
        with pytest.raises(ValueError):
            methods_groups.start_hosted_room_service()
        assert methods_groups._service is None
        if owner == "gateway":
            watcher = asyncio.create_task(runner._hosted_room_worker_watcher(interval=0.02))
        else:
            from fastapi.testclient import TestClient
            from hermes_cli import web_server
            monkeypatch.setattr(web_server, "_warm_gateway_module", lambda: None)
            client = TestClient(web_server.app)
            client.__enter__()
        binding_path.write_text(json.dumps(binding))
        await until(lambda: companions and companions[0].halt.is_set())
        first = companions[0]
        first_worker = first.thread
        assert methods_groups.get_hosted_room_service() is service
        assert first.thread.is_alive()
        assert first._mutex is not None
        # Explicit retry also refuses to release a live publisher's ownership.
        with pytest.raises(RuntimeError, match="did not stop"):
            await asyncio.to_thread(methods_groups.start_hosted_room_service)
        assert companions == [first]
        assert methods_groups._transport is first
        # No successful binding yet: an operator can repair initial config, but
        # the previous worker and mutex must cease before it can be adopted.
        with methods_groups._service_lock:
            binding["chat_id"] = -124
            binding_path.write_text(json.dumps(binding))
            release.set()
        await until(lambda: len(companions) == 2 and companions[1].ready.is_set())
        second = companions[1]
        assert not first_worker.is_alive() and first._mutex is None
        assert second._mutex is not None
        assert second.service is service
        assert second.config == binding
        # Accepted room work still uses the original coordinator after recovery.
        service.send(room_id="room", event_id="accepted", payload={"text": "@hermes inspect"})
        await until(lambda: rpc.submits == ["default"])
        config_path.write_text("gateway: []\n")
        assert await asyncio.to_thread(methods_groups.start_hosted_room_service) is service
        assert methods_groups._transport is second
        second.halt.set()
        await until(lambda: len(companions) == 3 and companions[2].ready.is_set())
        assert companions[2].config == binding
        assert methods_groups.get_hosted_room_service() is service
        assert second._mutex is None
        assert len(rpc.submits) == 1
        pending.pop()({"status": "settled", "text": "accepted work survived"})
        await until(lambda: hosted_room_driver.list_tasks(
            service.db_path, room_id="room", status="settled"))
    finally:
        for terminal in pending:
            terminal({"status": "settled", "text": "test cleanup"})
        release.set()
        runner._running = False
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            await runner._stop_hosted_room_worker()
        if client is not None:
            await asyncio.to_thread(client.__exit__, None, None, None)
        await asyncio.to_thread(methods_groups.stop_hosted_room_service)
    assert methods_groups._transport is None
    assert methods_groups._service is None
    assert all(item.thread is None and item._mutex is None for item in companions)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["gateway", "web"])
async def test_cancelled_native_start_cannot_revive_after_shutdown(tmp_path, monkeypatch, owner):
    from tui_gateway import methods_groups
    from plugins.platforms.telegram import hosted_room_transport as telegram

    entered, release = threading.Event(), threading.Event()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(methods_groups, "_bound_server", _server())
    monkeypatch.setattr(methods_groups, "_service", None)
    monkeypatch.setattr(methods_groups, "_transport", None)
    monkeypatch.setattr(methods_groups, "_binding", None)

    def blocked_binding():
        entered.set()
        assert release.wait(10)
        return None

    monkeypatch.setattr(telegram, "hosted_room_binding_path", blocked_binding)
    cancellation = []
    native_start = methods_groups.start_hosted_room_service

    def start_service(*, cancel_event):
        cancellation.append(cancel_event)
        return native_start(cancel_event=cancel_event)

    monkeypatch.setattr(methods_groups, "start_hosted_room_service", start_service)
    runner = GatewayRunner.__new__(GatewayRunner)
    client = None
    start = None
    stop = None
    try:
        if owner == "gateway":
            start = asyncio.create_task(runner._ensure_hosted_room_worker())
        else:
            from fastapi.testclient import TestClient
            from hermes_cli import web_server
            monkeypatch.setattr(web_server, "_warm_gateway_module", lambda: None)
            client = TestClient(web_server.app)
            client.__enter__()
        assert await asyncio.to_thread(entered.wait, 5)
        if start is not None:
            start.cancel()
            stop = asyncio.create_task(runner._stop_hosted_room_worker())
        else:
            stop = asyncio.create_task(asyncio.to_thread(client.__exit__, None, None, None))
        assert await asyncio.to_thread(cancellation[0].wait, 5)
        release.set()
        await asyncio.wait_for(stop, 10)
        if start is not None:
            with pytest.raises(asyncio.CancelledError):
                await start
        assert methods_groups._service is None
        assert methods_groups._transport is None
        assert native_start(cancel_event=cancellation[0]) is None
        assert methods_groups._service is None
    finally:
        release.set()
        if start is not None:
            await asyncio.gather(start, return_exceptions=True)
            await runner._stop_hosted_room_worker()
        if stop is not None:
            await asyncio.gather(stop, return_exceptions=True)

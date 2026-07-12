from __future__ import annotations

import asyncio
import socket

import pytest

from gateway.config import PlatformConfig
from gateway.config import Platform
from gateway.session import SessionSource
from plugins.platforms.a2a import adapter, server, setup, task_store


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(adapter, "_current_profile_name", lambda: "default", raising=False)
    setup.ensure_a2a_platform_config(public_url="https://agent.example.test/a2a")
    setup.add_principal("alice", profile="default")
    return home


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config(port):
    return PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": port,
            "public_url": "https://agent.example.test/a2a",
            "principals": {
                "alice": {"credential_ref": "inbound:alice", "profile": "default"}
            },
        },
    )


def test_adapter_contract_and_registration_flags():
    instance = adapter.A2AAdapter(_config(8645))
    assert instance.supports_async_delivery is False
    assert instance.SUPPORTS_MESSAGE_EDITING is False
    assert instance.request_dispatch_allows_gateway_commands is False
    assert instance.authorization_is_upstream is True

    calls = []
    skills = []

    class Context:
        def register_platform(self, **kwargs):
            calls.append(kwargs)

        def register_cli_command(self, **kwargs):
            pass

        def register_skill(self, *args):
            skills.append(args)

    adapter.register(Context())
    assert calls[0]["agent_tool_policy"] == "explicit"
    assert calls[0]["inbound_context_references_enabled"] is False
    assert skills[0][0] == "a2a-peer"
    assert skills[0][1].is_file()


@pytest.mark.asyncio
async def test_connect_disconnect_reconnect_and_bind_failure(hermes_home):
    port = _free_port()
    instance = adapter.A2AAdapter(_config(port))
    instance.set_message_handler(lambda event: f"reply:{event.text}")

    assert await instance.connect() is True
    assert instance.is_connected
    await instance.disconnect()
    assert not instance.is_connected
    assert await instance.connect(is_reconnect=True) is True
    instance._uvicorn_server.should_exit = True
    for _ in range(200):
        if instance.has_fatal_error:
            break
        await asyncio.sleep(0.01)
    assert instance.fatal_error_code == "a2a_server_exit"
    await instance.disconnect()
    assert await instance.connect(is_reconnect=True) is True
    await instance.disconnect()

    occupied = socket.socket()
    occupied.bind(("127.0.0.1", port))
    occupied.listen()
    try:
        blocked = adapter.A2AAdapter(_config(port))
        blocked.set_message_handler(lambda event: event.text)
        assert await blocked.connect() is False
    finally:
        occupied.close()


@pytest.mark.asyncio
async def test_non_loopback_bind_is_rejected(hermes_home):
    cfg = _config(8645)
    cfg.extra["host"] = "0.0.0.0"
    instance = adapter.A2AAdapter(cfg)
    assert await instance.connect() is False


@pytest.mark.asyncio
async def test_prepare_disconnect_uses_installed_canonical_interrupt_then_is_idempotent():
    instance = adapter.A2AAdapter(_config(8645))
    calls = []
    source = SessionSource(
        platform=Platform("a2a"),
        chat_id="a2a_chat",
        chat_type="dm",
        user_id="alice",
        message_id="task",
        profile="default",
    )

    async def canonical_interrupt(received, **_kwargs):
        calls.append(("interrupt", received))
        return True

    class App:
        def stop_accepting(self):
            calls.append(("ingress", None))

    class Executor:
        async def shutdown(self):
            calls.append(("shutdown", None))
            assert await instance.request_session_interrupt(source) is True

    instance.set_session_interrupt_handler(canonical_interrupt)
    instance._app = App()
    instance._executor = Executor()

    await instance.prepare_disconnect()
    instance.set_session_interrupt_handler(None)
    await instance.prepare_disconnect()

    assert calls == [
        ("ingress", None),
        ("shutdown", None),
        ("interrupt", source),
    ]


@pytest.mark.asyncio
async def test_timed_prepare_does_not_mark_adapter_prepared_or_suppress_retry():
    instance = adapter.A2AAdapter(_config(8645))
    attempts = 0

    class Executor:
        async def shutdown(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(60)

    instance._executor = Executor()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(instance.prepare_disconnect(), timeout=0.01)
    assert instance._prepared is False

    await instance.prepare_disconnect()

    assert attempts == 2
    assert instance._prepared is True


@pytest.mark.asyncio
async def test_failed_start_cleanup_orders_ingress_executor_transport_store():
    instance = adapter.A2AAdapter(_config(8645))
    calls = []

    class App:
        def stop_accepting(self):
            calls.append("ingress")

    class Executor:
        async def shutdown(self):
            calls.append("executor")

    class Server:
        should_exit = False

    class Store:
        async def close(self):
            calls.append("store")

    async def serve_forever():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            calls.append("transport")
            raise

    instance._app = App()
    instance._executor = Executor()
    instance._uvicorn_server = Server()
    instance._store = Store()
    instance._server_task = asyncio.create_task(serve_forever())
    await asyncio.sleep(0)

    await instance._cleanup_failed_start()

    assert calls == ["ingress", "executor", "transport", "store"]
    assert instance._store is None


@pytest.mark.asyncio
async def test_connect_app_construction_failure_closes_owned_store(hermes_home, monkeypatch):
    instance = adapter.A2AAdapter(_config(8645))
    closed = 0

    class Socket:
        def close(self):
            pass

    class Store:
        async def close(self):
            nonlocal closed
            closed += 1

    owned_store = Store()
    monkeypatch.setattr(instance, "_bind_socket", lambda _host, _port: Socket())
    monkeypatch.setattr(task_store, "create_task_store", lambda: owned_store)
    monkeypatch.setattr(
        server,
        "create_a2a_app",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    assert await instance.connect() is False
    assert closed == 1
    assert instance._store is None


@pytest.mark.asyncio
async def test_connect_cancellation_cleans_partial_lifespan_and_allows_reconnect(
    hermes_home, monkeypatch
):
    import uvicorn

    instance = adapter.A2AAdapter(_config(8645))
    entered = asyncio.Event()
    reconnect_mode = False
    stores = []
    sockets = []

    class Socket:
        closed = False

        def close(self):
            self.closed = True

    class Store:
        closed = False

        async def close(self):
            self.closed = True

    class App:
        def stop_accepting(self):
            pass

    class UvicornServer:
        def __init__(self, _config):
            self.started = reconnect_mode
            self.should_exit = False
            self.force_exit = False

        async def serve(self, sockets):
            del sockets
            entered.set()
            while not self.should_exit:
                await asyncio.sleep(0.001)

    def bind(_host, _port):
        sock = Socket()
        sockets.append(sock)
        return sock

    def make_store():
        store = Store()
        stores.append(store)
        return store

    monkeypatch.setattr(instance, "_bind_socket", bind)
    monkeypatch.setattr(task_store, "create_task_store", make_store)
    monkeypatch.setattr(server, "create_a2a_app", lambda *_a, **_k: App())
    monkeypatch.setattr(uvicorn, "Config", lambda *_a, **_k: object())
    monkeypatch.setattr(uvicorn, "Server", UvicornServer)

    connecting = asyncio.create_task(instance.connect())
    await entered.wait()
    connecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connecting

    assert stores[0].closed and sockets[0].closed
    assert instance._store is None
    assert instance._executor is None
    assert instance._listen_socket is None

    reconnect_mode = True
    entered.clear()
    assert await instance.connect(is_reconnect=True) is True
    await instance.disconnect()
    assert stores[1].closed and sockets[1].closed


@pytest.mark.asyncio
async def test_disconnect_timeout_retains_resistant_children_then_deferred_clears(
    monkeypatch,
):
    monkeypatch.setattr(adapter, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    instance = adapter.A2AAdapter(_config(8645))
    release = asyncio.Event()

    class App:
        def stop_accepting(self):
            pass

    class Executor:
        async def shutdown(self):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

        def active_session_sources(self):
            return ()

    class Store:
        closed = False

        async def close(self):
            self.closed = True

    class Socket:
        closed = False

        def close(self):
            self.closed = True

    async def live_server():
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    store = Store()
    sock = Socket()
    server_task = asyncio.create_task(live_server())
    instance._app = App()
    instance._executor = Executor()
    instance._store = store
    instance._listen_socket = sock
    instance._server_task = server_task

    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(instance.disconnect(), timeout=0.01)
    assert asyncio.get_running_loop().time() - started < 0.08

    assert not server_task.done()
    assert store.closed and sock.closed
    assert instance._server_task is server_task
    assert instance._executor is not None
    assert instance._store is None
    assert instance._listen_socket is None
    assert instance._stopping is False
    assert instance._deferred_cleanup_task is not None
    assert await instance.connect(is_reconnect=True) is False

    release.set()
    await asyncio.wait_for(instance._deferred_cleanup_task, timeout=0.2)

    assert instance._server_task is None
    assert instance._executor is None
    assert instance._deferred_cleanup_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["block", "raise"])
async def test_store_close_failure_or_block_is_retained_and_reconnect_fails_closed(
    monkeypatch, mode
):
    monkeypatch.setattr(adapter, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    instance = adapter.A2AAdapter(_config(8645))
    release = asyncio.Event()

    class Store:
        async def close(self):
            if mode == "raise":
                raise RuntimeError("close failed")
            await release.wait()

    store = Store()
    instance._store = store

    await asyncio.wait_for(instance.disconnect(), timeout=0.1)

    assert instance._store is store
    assert await instance.connect(is_reconnect=True) is False
    if mode == "raise":
        assert instance._cleanup_failed is True
    else:
        assert instance._deferred_cleanup_task is not None
        release.set()
        await asyncio.wait_for(instance._deferred_cleanup_task, timeout=0.2)
        assert instance._store is None

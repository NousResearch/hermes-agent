"""Cross-namespace idempotence for the lark WS isolation shims (issue #135559).

When the Feishu adapter is imported under more than one module namespace (plugin-loader
aliases), each copy used to have its own ``_WS_ISOLATION_INSTALLED`` flag and its own
``threading.local`` state. Every copy therefore installed another wrapper layer on the
shared ``lark_oapi.ws.client`` module, while a worker registered its adapter only into
its *own* copy's thread-local — the outermost wrapper then read ``adapter=None`` and
misclassified a deliberate disconnect (CLOSE 1000) as a dead receive loop.
"""

import asyncio
import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.platforms.feishu import adapter as feishu_adapter
from plugins.platforms.feishu import adapter_ws_isolation as feishu_ws_isolation


def _inject_fake_lark_module(monkeypatch, connect=None):
    """Make ``import lark_oapi.ws.client`` resolve to a module with the SDK's
    global layout (``loop`` + ``websockets.connect``)."""
    if connect is None:
        connect = MagicMock(name="real-connect")
    lark = types.ModuleType("lark_oapi")
    lark_ws = types.ModuleType("lark_oapi.ws")
    client_mod = types.ModuleType("lark_oapi.ws.client")
    client_mod.loop = SimpleNamespace(name="sdk-default-loop")
    client_mod.websockets = SimpleNamespace(connect=connect)

    class Client:  # the SDK class whose receive loop the isolation shim wraps
        async def _receive_message_loop(self):
            await asyncio.sleep(3600)

    client_mod.Client = Client
    lark.ws = lark_ws
    lark_ws.client = client_mod
    monkeypatch.setitem(sys.modules, "lark_oapi", lark)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", lark_ws)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", client_mod)
    monkeypatch.setattr(feishu_ws_isolation, "_WS_ISOLATION_INSTALLED", False)
    return client_mod


def _adapter_stub(**overrides):
    stub = SimpleNamespace(
        _loop=None,
        _ws_thread_loop=None,
        _ws_reconnect_nonce=None,
        _ws_reconnect_interval=None,
        _ws_ping_interval=None,
        _ws_ping_timeout=None,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def test_a_fresh_namespace_flag_does_not_stack_another_wrapper_layer(monkeypatch):
    """A second adapter copy starts with its own module flag unset, but the lark
    module already carries the install marker: the shims must not be re-wrapped."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    feishu_adapter._install_lark_ws_isolation(client_mod)
    loop_after_first = client_mod.loop
    connect_after_first = client_mod.websockets.connect
    receive_after_first = client_mod.Client._receive_message_loop
    # On main each call with an unset flag wrapped the (already wrapped) globals again.
    monkeypatch.setattr(feishu_ws_isolation, "_WS_ISOLATION_INSTALLED", False)
    feishu_adapter._install_lark_ws_isolation(client_mod)
    assert client_mod.loop is loop_after_first
    assert client_mod.websockets.connect is connect_after_first
    assert client_mod.Client._receive_message_loop is receive_after_first
    # The per-thread state is anchored once on the lark module and shared by every copy.
    assert feishu_adapter._lark_ws_state(client_mod) is feishu_adapter._lark_ws_state(
        client_mod
    )


def _load_second_namespace_copy(monkeypatch):
    """Load a second, independent module object for the same adapter file — what a
    plugin loader importing the adapter under an alias name produces in-process."""
    name = "feishu_adapter_namespace_copy"
    spec = importlib.util.spec_from_file_location(name, Path(feishu_adapter.__file__))
    copy = importlib.util.module_from_spec(spec)
    # Register before exec: the module's dataclasses resolve their field types through
    # sys.modules[cls.__module__].__dict__.
    monkeypatch.setitem(sys.modules, name, copy)
    spec.loader.exec_module(copy)
    return copy


def test_worker_of_a_second_namespace_copy_is_not_misclassified_on_disconnect(
    monkeypatch, caplog
):
    """#135559: the wrapper may belong to namespace copy B while the worker registers
    through copy A. The registration must reach the wrapper (shared, lark-anchored
    state) so a deliberate disconnect is logged at DEBUG, never as a died loop."""
    client_mod = _inject_fake_lark_module(monkeypatch)
    copy_b = _load_second_namespace_copy(monkeypatch)

    class InstallsAndExitsClient:
        """Copy B's worker only needs to install the shims from its namespace and
        register; its client exits immediately."""

        def start(self):
            pass

    class DisconnectingSDKClient:
        """Copy A's worker mid-``disconnect()``: the receive loop ends with the SDK's
        normal CLOSE 1000 exception after ``_running`` was already flipped."""

        async def _receive_message_loop(self):
            await asyncio.sleep(0.01)
            raise ConnectionError("sent 1000 (OK); then received 1000 (OK)")

        def start(self):
            loop = client_mod.loop

            async def _select():  # the SDK's forever-parked select loop
                while True:
                    await asyncio.sleep(3600)

            loop.create_task(self._receive_message_loop())
            loop.run_until_complete(_select())

    client_mod.Client = DisconnectingSDKClient  # the class B's install will wrap

    # Copy B's worker starts first and installs the shims from B's namespace.
    stub_b = _adapter_stub()
    thread_b = threading.Thread(
        target=lambda: copy_b._run_official_feishu_ws_client(
            InstallsAndExitsClient(), stub_b
        ),
        daemon=True,
    )
    thread_b.start()
    thread_b.join(timeout=10)
    assert not thread_b.is_alive()

    # Copy A's worker then runs a deliberate-disconnect turn (``disconnect()`` already
    # flipped ``_running``); its registration goes through A's namespace code.
    stub_a = _adapter_stub(_running=False)
    thread_a = threading.Thread(
        target=lambda: feishu_adapter._run_official_feishu_ws_client(
            DisconnectingSDKClient(), stub_a
        ),
        daemon=True,
    )
    thread_a.start()
    thread_a.join(timeout=10)
    assert not thread_a.is_alive()

    # On main, A's install stacked a second wrapper that read B's (empty-for-this-thread)
    # state, saw adapter=None and reported the CLOSE 1000 as a dead receive loop.
    errors = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and "receive loop" in r.getMessage()
    ]
    assert errors == [], [r.getMessage() for r in errors]

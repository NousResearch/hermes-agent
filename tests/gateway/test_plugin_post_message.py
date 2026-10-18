"""``PluginContext.post_message``: one message in the calling session's chat, edited in place."""

import asyncio
import contextvars
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, PlatformConfig, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.session_context import scoped_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
        self.calls = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("send", chat_id, content))
        return SendResult(success=True, message_id=f"m{len(self.calls)}")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _EditingAdapter(_Adapter):
    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.calls.append(("edit", chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)


def _setup(tmp_path, adapter):
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    entry = store.get_or_create_session(SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42", user_name="u"))
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._is_user_authorized = MagicMock(return_value=True)
    ctx = PluginContext(PluginManifest(name="progress", key="progress", source="user"), PluginManager())
    return runner, ctx, entry


async def _in_session(session_key, fn):
    """Run *fn* on a worker thread with the session bound, as a tool handler is."""
    def bound():
        with scoped_session_key(session_key):
            return fn()
    return await asyncio.to_thread(contextvars.copy_context().run, bound)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls, edited", [(_EditingAdapter, True), (_Adapter, False)])
async def test_post_then_edit_targets_the_calling_session(tmp_path, adapter_cls, edited):
    adapter = adapter_cls()
    runner, ctx, entry = _setup(tmp_path, adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: runner), \
            patch.object(GatewayRunner, "_plugin_messaging_session_owned", return_value=True):
        handle = await _in_session(entry.session_key, lambda: ctx.post_message("step 1/2"))
        assert handle.ok and handle.message_id == "m1"
        result = await _in_session("", lambda: handle.edit("step 2/2"))  # the handle carries its route
    assert result["ok"] is True and result["edited"] is edited
    if edited:
        assert adapter.calls == [("send", "42", "step 1/2"), ("edit", "42", "m1", "step 2/2")]
    else:  # no edit support: a fresh message, and the handle follows it
        assert adapter.calls == [("send", "42", "step 1/2"), ("send", "42", "step 2/2")]
        assert result["fallback"] is True and handle.message_id == "m2"


@pytest.mark.asyncio
async def test_no_bound_session_is_refused_without_sending(tmp_path):
    adapter = _EditingAdapter()
    runner, ctx, _entry = _setup(tmp_path, adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        handle = await asyncio.to_thread(ctx.post_message, "from a timer")
        edit = await asyncio.to_thread(handle.edit, "still nothing")
    assert (handle.ok, handle.error, edit["ok"]) == (False, "no_conversation", False)
    assert adapter.calls == []

"""Tests for plugin-registered Slack view-submission (modal) handlers.

Covers:
* ``PluginContext.register_slack_view_handler`` validation + queuing
* ``PluginManager.get_slack_view_handlers`` accessor
* ``SlackAdapter.connect`` wiring those handlers into the AsyncApp via
  ``app.view(callback_id)``
* Defensive wrapping: a plugin view handler that raises does NOT take
  down the gateway and Slack still gets an ack.

Mirrors ``test_slack_plugin_action_handlers.py`` (the Block Kit action
sibling of this API).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure the repo root is importable when this test runs directly
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Mock slack-bolt so SlackAdapter can be imported even without the package
# ---------------------------------------------------------------------------

def _ensure_slack_mock() -> None:
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

from hermes_cli.plugins import (  # noqa: E402
    PluginContext,
    PluginManager,
    PluginManifest,
)


# ---------------------------------------------------------------------------
# PluginContext.register_slack_view_handler — input validation + queuing
# ---------------------------------------------------------------------------

def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    """Build a fresh PluginManager + PluginContext bound to it."""
    mgr = PluginManager()
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        description="test",
    )
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


class TestRegisterSlackViewHandlerAPI:
    """Behaviour of ctx.register_slack_view_handler()."""

    def test_callback_id_is_queued(self):
        mgr, ctx = _make_ctx()

        async def cb(ack, body, view):  # pragma: no cover - never called here
            await ack()

        ctx.register_slack_view_handler("edit_modal", cb)

        handlers = mgr.get_slack_view_handlers()
        assert len(handlers) == 1
        callback_id, callback, plugin_name = handlers[0]
        assert callback_id == "edit_modal"
        assert callback is cb
        assert plugin_name == "test_plugin"

    def test_non_callable_callback_rejected(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_slack_view_handler("edit_modal", "not a function")

    def test_empty_callback_id_rejected(self):
        _mgr, ctx = _make_ctx()

        async def cb(ack, body, view):  # pragma: no cover
            await ack()

        with pytest.raises(ValueError, match="empty callback_id"):
            ctx.register_slack_view_handler("   ", cb)

    def test_accessor_returns_copy(self):
        mgr, ctx = _make_ctx()

        async def cb(ack, body, view):  # pragma: no cover
            await ack()

        ctx.register_slack_view_handler("edit_modal", cb)
        handlers = mgr.get_slack_view_handlers()
        handlers.clear()
        assert len(mgr.get_slack_view_handlers()) == 1

    def test_multiple_plugins_tracked_by_name(self):
        mgr = PluginManager()
        ctx_a = PluginContext(
            manifest=PluginManifest(name="plug_a", version="0.1", description="a"),
            manager=mgr,
        )
        ctx_b = PluginContext(
            manifest=PluginManifest(name="plug_b", version="0.1", description="b"),
            manager=mgr,
        )

        async def cb_a(ack, body, view):  # pragma: no cover
            await ack()

        async def cb_b(ack, body, view):  # pragma: no cover
            await ack()

        ctx_a.register_slack_view_handler("modal_a", cb_a)
        ctx_b.register_slack_view_handler("modal_b", cb_b)

        handlers = mgr.get_slack_view_handlers()
        assert {h[2] for h in handlers} == {"plug_a", "plug_b"}


# ---------------------------------------------------------------------------
# SlackAdapter.connect wires plugin-registered view handlers into AsyncApp
# ---------------------------------------------------------------------------


def _connect_with_recording_app(
    adapter: SlackAdapter,
    *,
    view_handlers: list,
) -> tuple[bool, list]:
    """Run adapter.connect() with mocks and return (result, registered_views).

    Captures every callback_id passed to ``app.view()`` so tests can
    assert plugin-supplied view handlers were wired up.
    """
    registered_views: list = []  # list of (callback_id, callback)

    def mock_view(callback_id):
        def decorator(fn):
            registered_views.append((callback_id, fn))
            return fn
        return decorator

    def mock_action(_action_id):
        def decorator(fn):
            return fn
        return decorator

    def mock_event(_event_type):
        def decorator(fn):
            return fn
        return decorator

    def mock_command(_cmd):
        def decorator(fn):
            return fn
        return decorator

    mock_app = MagicMock()
    mock_app.event = mock_event
    mock_app.command = mock_command
    mock_app.action = mock_action
    mock_app.view = mock_view
    mock_app.client = AsyncMock()

    mock_web_client = AsyncMock()
    mock_web_client.auth_test = AsyncMock(return_value={
        "user_id": "U_BOT",
        "user": "testbot",
        "team_id": "T_FAKE",
        "team": "FakeTeam",
    })

    fake_mgr = MagicMock()
    fake_mgr.get_slack_action_handlers.return_value = []
    fake_mgr.get_slack_view_handlers.return_value = view_handlers

    with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
         patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
         patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
         patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
         patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
         patch("gateway.status.release_scoped_lock"), \
         patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr), \
         patch("asyncio.create_task"):
        result = asyncio.run(adapter.connect())

    return result, registered_views


class TestSlackAdapterPluginViewWiring:
    """connect() must register plugin-supplied view handlers on AsyncApp."""

    def test_plugin_view_handler_wired_into_app(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        async def my_handler(ack, body, view):  # pragma: no cover - not invoked
            await ack()

        result, registered = _connect_with_recording_app(
            adapter, view_handlers=[("edit_modal", my_handler, "jarvis")],
        )

        assert result is True
        assert "edit_modal" in [cid for cid, _cb in registered]

    def test_no_view_handlers_does_not_break_connect(self):
        """An empty view handler list is the common case — must be a no-op."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        result, registered = _connect_with_recording_app(
            adapter, view_handlers=[],
        )
        assert result is True
        assert registered == []

    def test_plugin_exception_does_not_propagate_to_slack(self):
        """A misbehaving view handler must NOT crash slack_bolt's dispatch.

        The wrapper installed by connect() catches exceptions, logs them,
        and best-effort-acks so Slack stops retrying the submission.
        """
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        async def boom(ack, body, view):
            raise RuntimeError("plugin bug")

        _result, registered = _connect_with_recording_app(
            adapter, view_handlers=[("explode_modal", boom, "buggy_plugin")],
        )

        wrapped = next(cb for cid, cb in registered if cid == "explode_modal")
        ack = AsyncMock()

        # Wrapper must swallow the RuntimeError.
        asyncio.run(wrapped(ack, {"foo": "bar"}, {"callback_id": "explode_modal"}))

        # Slack still got an ack — best-effort fallback after exception.
        ack.assert_awaited()

    def test_plugin_view_handler_invoked_with_slack_args(self):
        """Happy path: the plugin's callback receives (ack, body, view)."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        seen: dict = {}

        async def cb(ack, body, view):
            seen["body"] = body
            seen["view"] = view
            await ack()

        _result, registered = _connect_with_recording_app(
            adapter, view_handlers=[("modal_x", cb, "plug_x")],
        )

        wrapped = next(c for cid, c in registered if cid == "modal_x")
        ack = AsyncMock()
        asyncio.run(wrapped(ack, {"b": 1}, {"callback_id": "modal_x"}))

        assert seen["body"] == {"b": 1}
        assert seen["view"] == {"callback_id": "modal_x"}
        ack.assert_awaited()

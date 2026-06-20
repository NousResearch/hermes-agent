"""Tests for plugin-registered Slack Block Kit action handlers.

Covers:
* ``PluginContext.register_slack_action_handler`` validation + queuing
* ``PluginManager.get_slack_action_handlers`` accessor
* ``SlackAdapter.connect`` wiring those handlers into the AsyncApp
* Defensive wrapping: a plugin handler that raises does NOT take down
  the gateway and Slack still gets an ack.
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
# PluginContext.register_slack_action_handler — input validation + queuing
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


class TestRegisterSlackActionHandlerAPI:
    """Behaviour of ctx.register_slack_action_handler()."""

    def test_string_action_id_is_queued(self):
        mgr, ctx = _make_ctx()

        async def cb(ack, body, action):  # pragma: no cover - never called here
            await ack()

        ctx.register_slack_action_handler("inbox_sweep_approve", cb)

        handlers = mgr.get_slack_action_handlers()
        assert len(handlers) == 1
        action_id, callback, plugin_name = handlers[0]
        assert action_id == "inbox_sweep_approve"
        assert callback is cb
        assert plugin_name == "test_plugin"

    def test_regex_action_id_is_accepted(self):
        """slack_bolt accepts re.Pattern matchers — so should the plugin API."""
        import re as _re
        mgr, ctx = _make_ctx()

        async def cb(ack, body, action):  # pragma: no cover
            await ack()

        pat = _re.compile(r"^inbox_sweep_.*$")
        ctx.register_slack_action_handler(pat, cb)
        handlers = mgr.get_slack_action_handlers()
        assert handlers[0][0] is pat

    def test_constraint_dict_action_id_is_accepted(self):
        """slack_bolt also accepts {"action_id": ..., "block_id": ...} dicts."""
        mgr, ctx = _make_ctx()

        async def cb(ack, body, action):  # pragma: no cover
            await ack()

        constraint = {"action_id": "approve", "block_id": "row_3"}
        ctx.register_slack_action_handler(constraint, cb)
        handlers = mgr.get_slack_action_handlers()
        assert handlers[0][0] == constraint

    def test_non_callable_callback_raises(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="non-callable"):
            ctx.register_slack_action_handler("approve", "not a function")  # type: ignore[arg-type]

    def test_empty_string_action_id_raises(self):
        _mgr, ctx = _make_ctx()

        async def cb(ack, body, action):  # pragma: no cover
            await ack()

        with pytest.raises(ValueError, match="empty action_id"):
            ctx.register_slack_action_handler("   ", cb)

    def test_none_action_id_raises(self):
        _mgr, ctx = _make_ctx()

        async def cb(ack, body, action):  # pragma: no cover
            await ack()

        with pytest.raises(ValueError, match="empty action_id"):
            ctx.register_slack_action_handler(None, cb)

    def test_get_slack_action_handlers_returns_copy(self):
        """The accessor should return a copy so callers can't mutate state."""
        mgr, ctx = _make_ctx()

        async def cb(ack, body, action):  # pragma: no cover
            await ack()

        ctx.register_slack_action_handler("a", cb)

        handlers = mgr.get_slack_action_handlers()
        handlers.clear()
        assert len(mgr.get_slack_action_handlers()) == 1

    def test_multiple_plugins_each_recorded(self):
        mgr = PluginManager()
        ctx_a = PluginContext(
            manifest=PluginManifest(name="plug_a", version="0", description=""),
            manager=mgr,
        )
        ctx_b = PluginContext(
            manifest=PluginManifest(name="plug_b", version="0", description=""),
            manager=mgr,
        )

        async def cb_a(ack, body, action):  # pragma: no cover
            await ack()

        async def cb_b(ack, body, action):  # pragma: no cover
            await ack()

        ctx_a.register_slack_action_handler("approve", cb_a)
        ctx_b.register_slack_action_handler("decline", cb_b)

        handlers = mgr.get_slack_action_handlers()
        assert {h[2] for h in handlers} == {"plug_a", "plug_b"}


# ---------------------------------------------------------------------------
# SlackAdapter.connect wires plugin-registered handlers into AsyncApp
# ---------------------------------------------------------------------------


def _connect_with_recording_app(
    adapter: SlackAdapter,
    *,
    plugin_handlers: list,
) -> tuple[bool, list]:
    """Run adapter.connect() with mocks and return (result, registered_actions).

    Captures every action_id passed to ``app.action()`` so tests can
    assert that built-in handlers AND plugin-supplied handlers were
    wired up.
    """
    registered_actions: list = []  # list of (action_id, callback)

    def mock_action(action_id):
        def decorator(fn):
            registered_actions.append((action_id, fn))
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
    mock_app.client = AsyncMock()

    mock_web_client = AsyncMock()
    mock_web_client.auth_test = AsyncMock(return_value={
        "user_id": "U_BOT",
        "user": "testbot",
        "team_id": "T_FAKE",
        "team": "FakeTeam",
    })

    fake_mgr = MagicMock()
    fake_mgr.get_slack_action_handlers.return_value = plugin_handlers

    with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
         patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
         patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
         patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
         patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
         patch("gateway.status.release_scoped_lock"), \
         patch("hermes_cli.plugins.get_plugin_manager", return_value=fake_mgr), \
         patch("asyncio.create_task"):
        result = asyncio.run(adapter.connect())

    return result, registered_actions


class TestSlackAdapterPluginActionWiring:
    """connect() must register plugin-supplied action handlers on AsyncApp."""

    def test_plugin_handler_wired_into_app(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        async def my_handler(ack, body, action):  # pragma: no cover - not invoked
            await ack()

        plugin_handlers = [("inbox_sweep_approve", my_handler, "jarvis")]
        result, registered = _connect_with_recording_app(
            adapter, plugin_handlers=plugin_handlers,
        )

        assert result is True
        action_ids = [aid for aid, _cb in registered]
        # Built-in approval buttons remain registered…
        assert "hermes_approve_once" in action_ids
        assert "hermes_deny" in action_ids
        # …and the plugin's action_id was added.
        assert "inbox_sweep_approve" in action_ids

    def test_no_plugin_handlers_does_not_break_connect(self):
        """An empty plugin handler list is the common case — must be a no-op."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        result, registered = _connect_with_recording_app(
            adapter, plugin_handlers=[],
        )
        assert result is True
        # Built-ins still wired
        action_ids = [aid for aid, _cb in registered]
        assert "hermes_approve_once" in action_ids

    def test_plugin_exception_does_not_propagate_to_slack(self):
        """A misbehaving plugin handler must NOT crash slack_bolt's dispatch.

        The wrapper installed by connect() catches exceptions, logs them,
        and best-effort-acks so Slack stops retrying the click.
        """
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        async def boom(ack, body, action):
            raise RuntimeError("plugin bug")

        plugin_handlers = [("explode", boom, "buggy_plugin")]
        _result, registered = _connect_with_recording_app(
            adapter, plugin_handlers=plugin_handlers,
        )

        wrapped = next(cb for aid, cb in registered if aid == "explode")
        ack = AsyncMock()
        body = {"foo": "bar"}
        action = {"action_id": "explode", "value": "x"}

        # Wrapper must swallow the RuntimeError.
        asyncio.run(wrapped(ack, body, action))

        # Slack still got an ack — best-effort fallback after exception.
        ack.assert_awaited()

    def test_plugin_handler_invoked_with_slack_args(self):
        """Happy path: the plugin's callback receives (ack, body, action)."""
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        seen: dict = {}

        async def cb(ack, body, action):
            seen["body"] = body
            seen["action"] = action
            await ack()

        plugin_handlers = [("approve_x", cb, "plug_x")]
        _result, registered = _connect_with_recording_app(
            adapter, plugin_handlers=plugin_handlers,
        )

        wrapped = next(c for aid, c in registered if aid == "approve_x")
        ack = AsyncMock()
        asyncio.run(wrapped(ack, {"b": 1}, {"action_id": "approve_x"}))

        ack.assert_awaited_once_with()
        assert seen["body"] == {"b": 1}
        assert seen["action"] == {"action_id": "approve_x"}

    def test_wrapper_signature_only_exposes_slack_bolt_args(self):
        """Regression: slack_bolt introspects listener signatures and passes
        ``None`` for any parameter name it doesn't recognise. If the wrapper
        leaks closure variables (e.g. ``_cb``, ``_plugin_name``) into its
        signature via default args, they get clobbered to None at dispatch
        time and the wrapped callback becomes ``NoneType``.

        The wrapper must only expose ``(ack, body, action)``.
        """
        import inspect

        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        async def cb(ack, body, action):  # pragma: no cover
            await ack()

        plugin_handlers = [("approve_x", cb, "plug_x")]
        _result, registered = _connect_with_recording_app(
            adapter, plugin_handlers=plugin_handlers,
        )

        wrapped = next(c for aid, c in registered if aid == "approve_x")
        params = list(inspect.signature(wrapped).parameters)
        assert params == ["ack", "body", "action"], (
            f"wrapper exposes extra params slack_bolt would clobber: {params}"
        )

    def test_plugin_loader_failure_does_not_break_connect(self):
        """If get_plugin_manager() blows up, connect() must still succeed.

        Defensive belt-and-suspenders: the gateway should not refuse to
        start because the plugin layer is unhealthy.
        """
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)

        registered_actions: list = []

        def mock_action(action_id):
            def decorator(fn):
                registered_actions.append((action_id, fn))
                return fn
            return decorator

        def _noop(_):
            def decorator(fn): return fn
            return decorator

        mock_app = MagicMock()
        mock_app.event = _noop
        mock_app.command = _noop
        mock_app.action = mock_action
        mock_app.client = AsyncMock()

        mock_web_client = AsyncMock()
        mock_web_client.auth_test = AsyncMock(return_value={
            "user_id": "U_BOT",
            "user": "testbot",
            "team_id": "T_FAKE",
            "team": "FakeTeam",
        })

        with patch.object(_slack_mod, "AsyncApp", return_value=mock_app), \
             patch.object(_slack_mod, "AsyncWebClient", return_value=mock_web_client), \
             patch.object(_slack_mod, "AsyncSocketModeHandler", return_value=MagicMock()), \
             patch.dict(os.environ, {"SLACK_APP_TOKEN": "xapp-fake"}), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock"), \
             patch("hermes_cli.plugins.get_plugin_manager",
                   side_effect=RuntimeError("plugins broken")), \
             patch("asyncio.create_task"):
            result = asyncio.run(adapter.connect())

        assert result is True
        # Built-ins still wired even when plugin loader failed.
        action_ids = [aid for aid, _cb in registered_actions]
        assert "hermes_approve_once" in action_ids

"""Contract tests for profile-local plugin turns through GatewayRunner."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import threading

import pytest

from gateway import execution_lifecycle
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


def _event(**overrides) -> MessageEvent:
    source = overrides.pop(
        "source",
        SessionSource(
            platform=Platform.LOCAL,
            chat_id="linear-session-1",
            chat_name="Linear Agent Session",
            chat_type="dm",
            user_id="linear-agent",
            user_name="Linear Agent",
            scope_id="workspace-1",
            profile="aggie",
        ),
    )
    values = {
        "text": "untrusted payload",
        "message_type": MessageType.TEXT,
        "source": source,
        "internal": True,
        "allow_gateway_control": False,
        "metadata": {"linear_agent": True},
    }
    values.update(overrides)
    return MessageEvent(**values)


@pytest.mark.asyncio
async def test_dispatch_internal_plugin_event_uses_normal_scoped_handler():
    runner = object.__new__(GatewayRunner)
    handler = AsyncMock(return_value="agent response")
    runner._primary_message_handler = lambda: handler
    event = _event()

    result = await runner.dispatch_internal_plugin_event(event)

    assert result == "agent response"
    handler.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_prepare_internal_plugin_session_persists_session_without_starting_turn():
    runner = object.__new__(GatewayRunner)
    entry = SimpleNamespace(session_id="hermes-session-1")
    store = object()
    get_or_create = AsyncMock(return_value=entry)
    facade = SimpleNamespace(
        _store=store,
        get_or_create_session=get_or_create,
    )
    object.__setattr__(runner, "session_store", store)
    object.__setattr__(runner, "_async_session_store", facade)
    handler = AsyncMock(return_value="must not run")
    runner._primary_message_handler = lambda: handler
    event = _event()

    result = await runner.prepare_internal_plugin_session(event)

    assert result == "hermes-session-1"
    get_or_create.assert_awaited_once_with(
        event.source,
        touch_activity=False,
    )
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_internal_plugin_session_uses_explicit_multiplex_profile_scope(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    profile = root / "profiles" / "aggie"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda source: profile
    seen = []

    async def get_or_create(_source, *, touch_activity):
        seen.append((Path(get_hermes_home()), touch_activity))
        return SimpleNamespace(session_id="hermes-session-1")

    store = object()
    facade = SimpleNamespace(_store=store, get_or_create_session=get_or_create)
    object.__setattr__(runner, "session_store", store)
    object.__setattr__(runner, "_async_session_store", facade)

    result = await runner.prepare_internal_plugin_session(_event())

    assert result == "hermes-session-1"
    assert seen == [(profile, False)]
    assert Path(get_hermes_home()) == root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "error"),
    [
        (object(), TypeError),
        (_event(internal=False), PermissionError),
        (_event(allow_gateway_control=True), PermissionError),
        (_event(message_type=MessageType.PHOTO), ValueError),
        (_event(source=None), ValueError),
        (
            _event(
                source=SessionSource(
                    platform=Platform.SLACK,
                    chat_id="C123",
                    chat_type="dm",
                    profile="aggie",
                )
            ),
            PermissionError,
        ),
        (
            _event(
                source=SessionSource(
                    platform=Platform.LOCAL,
                    chat_id="local-1",
                    chat_type="dm",
                    profile=None,
                )
            ),
            ValueError,
        ),
    ],
)
async def test_dispatch_internal_plugin_event_rejects_unsafe_shapes(event, error):
    runner = object.__new__(GatewayRunner)
    handler = AsyncMock(return_value="must not run")
    runner._primary_message_handler = lambda: handler

    with pytest.raises(error):
        await runner.dispatch_internal_plugin_event(event)

    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_internal_plugin_event_keeps_slash_payload_as_text():
    runner = object.__new__(GatewayRunner)
    handler = AsyncMock(return_value="treated as conversation")
    runner._primary_message_handler = lambda: handler
    event = _event(text="/restart")

    result = await runner.dispatch_internal_plugin_event(event)

    assert result == "treated as conversation"
    assert event.is_command() is False
    handler.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_profile_services_start_for_every_multiplex_profile(tmp_path, monkeypatch):
    from gateway import run as gateway_run
    from hermes_cli import plugins as plugin_module

    active_home = tmp_path / "active"
    secondary_home = tmp_path / "secondary"
    active_home.mkdir()
    secondary_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(active_home))
    monkeypatch.setenv("HERMES_PROFILE", "active")

    seen: list[tuple[str, Path, Path]] = []

    def service(label):
        async def run(runtime):
            seen.append(
                (
                    label,
                    Path(runtime.profile_home),
                    Path(get_hermes_home()),
                )
            )
            await runtime.stop_event.wait()

        return run

    managers = {
        active_home: SimpleNamespace(_profile_services=[("linear", service("active"))]),
        secondary_home: SimpleNamespace(
            _profile_services=[("linear", service("secondary"))]
        ),
    }
    monkeypatch.setattr(
        plugin_module,
        "get_plugin_manager",
        lambda: managers[Path(get_hermes_home())],
    )
    monkeypatch.setattr(
        gateway_run,
        "_multiplex_profile_homes",
        lambda _config: [("active", active_home), ("secondary", secondary_home)],
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._start_plugin_profile_services()
    await asyncio.sleep(0)

    assert seen == [
        ("active", active_home, active_home),
        ("secondary", secondary_home, secondary_home),
    ]
    assert {task.get_name() for task in runner._profile_service_tasks} == {
        "profile-service:active:linear",
        "profile-service:secondary:linear",
    }

    await runner._stop_plugin_profile_services()
    assert runner._profile_service_tasks == []


@pytest.mark.asyncio
async def test_raising_dispatch_leaves_the_session_usable(tmp_path, monkeypatch):
    """A dispatch that raises must not wedge the session_key forever.

    The live record used to leak, so every later execution on that session was rejected with
    "already has a live execution" and nothing swept it.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)

    async def handler(_event):
        raise RuntimeError("handler exploded")

    runner._primary_message_handler = lambda: handler
    event = _event()
    _session_key = runner._session_key_for_source(event.source)

    with pytest.raises(RuntimeError, match="handler exploded"):
        await runner.dispatch_internal_plugin_event(event, execution_id="plugin-exec-001")

    assert "plugin-exec-001" not in runner._internal_plugin_execution_records()
    tombstone = runner._internal_plugin_retired_executions["plugin-exec-001"]
    assert tombstone["state"] == "failed"
    assert tombstone["occupancy"] == "released"

    # The next execution on the same session is accepted.
    runner._primary_message_handler = lambda: AsyncMock(return_value="second run")
    assert await runner.dispatch_internal_plugin_event(
        _event(), execution_id="plugin-exec-002"
    ) == "second run"


@pytest.mark.asyncio
async def test_raising_dispatch_with_live_worker_keeps_its_record(tmp_path, monkeypatch):
    """Respect the worker Event: a physically live worker is still occupying the session."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)
    live = threading.Event()

    async def handler(event):
        runner._track_internal_plugin_execution_worker(
            event._internal_plugin_execution_id, live
        )
        raise RuntimeError("handler exploded")

    runner._primary_message_handler = lambda: handler

    with pytest.raises(RuntimeError):
        await runner.dispatch_internal_plugin_event(_event(), execution_id="plugin-exec-001")
    assert "plugin-exec-001" in runner._internal_plugin_execution_records()

    # Once the worker really finishes, the bounded sweep on the next registration reclaims it.
    live.set()
    runner._primary_message_handler = lambda: AsyncMock(return_value="second run")
    await runner.dispatch_internal_plugin_event(_event(), execution_id="plugin-exec-002")
    assert "plugin-exec-001" not in runner._internal_plugin_execution_records()


@pytest.mark.asyncio
async def test_raising_dispatch_after_a_tool_quarantines_but_not_forever(tmp_path, monkeypatch):
    """Unknown effects stay fail-closed; the quarantine still has a reopen path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)

    async def handler(event):
        runner._observe_internal_plugin_tool_event(
            event._internal_plugin_execution_id, "tool.started"
        )
        raise RuntimeError("handler exploded")

    runner._primary_message_handler = lambda: handler

    with pytest.raises(RuntimeError):
        await runner.dispatch_internal_plugin_event(_event(), execution_id="plugin-exec-001")
    assert runner._internal_plugin_retired_executions["plugin-exec-001"]["occupancy"] == "unknown"

    async def ok_handler(_event):
        return "second run"

    runner._primary_message_handler = lambda: ok_handler
    with pytest.raises(ValueError, match="quarantined"):
        await runner.dispatch_internal_plugin_event(_event(), execution_id="plugin-exec-002")

    monkeypatch.setattr(execution_lifecycle, "QUARANTINE_TTL_SECONDS", 0.0)
    assert await runner.dispatch_internal_plugin_event(
        _event(), execution_id="plugin-exec-002"
    ) == "second run"

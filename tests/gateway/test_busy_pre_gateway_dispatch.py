"""Regression: ``pre_gateway_dispatch`` runs exactly once for a message that arrives mid-turn.

A message that arrives while its session is busy is diverted by the adapter
(``_handle_message_while_active``) before it reaches the runner's
``_handle_message``, so the plugin hook never saw it. The fix runs the hook on
the busy path through a runner-owned handler and marks accepted events so the
later drain through the cold path does not run it a second time.

These tests pin that contract end to end through a real adapter queue and the
real runner entry points:

* busy arrival -> queued -> drained through ``_handle_message``: one hook call,
  and the queued event carries the hook's rewrite;
* a ``skip`` from the hook on the busy path drops the message (nothing queued);
* an inline diversion (``/approve``) reaches the runner's handler without a second hook call;
* under multiplex, a secondary profile's busy arrival runs the hook inside that
  profile's runtime scope, and the drain through the same profile's cold
  handler does not run it again;
* with a real plugin loaded by the plugin manager (no hook-seam patch): the rewrite survives
  queue -> drain with one call, ``/stop`` and ``/approve`` that the plugin allows are still routed
  as commands through the real runner handler, and a plugin installed only in a secondary
  profile fires for that profile's busy arrival.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "FEISHU_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _hook_recorder(monkeypatch, result_for):
    """Patch the async hook seam; ``result_for(event)`` returns the hook results. Returns the call log."""
    calls = []

    async def _hook(name, **kwargs):
        if name != "pre_gateway_dispatch":
            return []
        event = kwargs["event"]
        from hermes_constants import get_hermes_home

        calls.append(SimpleNamespace(text=event.text, home=Path(get_hermes_home())))
        return result_for(event)

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _hook)
    return calls


def _whatsapp_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        message_id="m-busy",
        source=SessionSource(
            platform=Platform.WHATSAPP,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _primary_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.WHATSAPP: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner


def _busy_adapter(platform: Platform, runner, pre_dispatch_handler):
    """A real adapter whose session for the event is mid-turn, wired the way the runner wires it."""
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), platform)
    adapter._send_with_retry = AsyncMock(return_value=None)
    adapter.set_pre_gateway_dispatch_handler(pre_dispatch_handler)
    return adapter


async def _arrive_while_busy(adapter, event) -> str:
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    # Text follow-ups are debounced in queue mode; turn that off so the arrival lands in the queue now.
    adapter._busy_text_debounce_seconds = 0
    await adapter._handle_message_while_active(event, session_key)
    return session_key


@pytest.mark.asyncio
async def test_busy_arrival_runs_hook_once_across_queue_and_drain(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    calls = _hook_recorder(
        monkeypatch, lambda event: [{"action": "rewrite", "text": "rewritten: " + event.text}])

    runner = _primary_runner()
    delivered = []

    async def _capture(event, source, _quick_key, _run_generation):
        delivered.append(event.text)
        return "ok"

    runner._handle_message_with_agent = _capture
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())

    session_key = await _arrive_while_busy(adapter, _whatsapp_event("sent mid-turn"))

    assert [c.text for c in calls] == ["sent mid-turn"]
    queued = adapter._pending_messages.get(session_key)
    assert queued is not None, "an allowed busy arrival must still be queued"
    assert queued.text == "rewritten: sent mid-turn"

    # The current turn ends and the queue drains through the runner's normal entry point.
    adapter._pending_messages.pop(session_key)
    await runner._handle_message(queued)

    assert len(calls) == 1, "the drain must not run pre_gateway_dispatch a second time"
    assert delivered == ["rewritten: sent mid-turn"]


@pytest.mark.asyncio
async def test_busy_arrival_skipped_by_hook_is_never_queued(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    calls = _hook_recorder(monkeypatch, lambda event: [{"action": "skip", "reason": "plugin-handled"}])

    runner = _primary_runner()
    busy_handler = AsyncMock(return_value=True)
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())
    adapter.set_busy_session_handler(busy_handler)

    session_key = await _arrive_while_busy(adapter, _whatsapp_event("handled by a plugin"))

    assert len(calls) == 1
    assert session_key not in adapter._pending_messages
    busy_handler.assert_not_awaited()  # dropped before steer/interrupt/queue logic


@pytest.mark.asyncio
async def test_busy_approval_dispatched_inline_runs_hook_once(monkeypatch):
    """An inline diversion (``/approve`` while the agent waits on approval) reaches the runner's
    handler without re-running the hook it already passed on the busy path."""
    _clear_auth_env(monkeypatch)
    calls = _hook_recorder(monkeypatch, lambda event: [{"action": "allow"}])

    runner = _primary_runner()
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())
    handled = []

    async def _runner_handler(event):
        # The runner's ``_handle_message`` opens with this gate; record what passes it.
        handled.append(await runner._hm_pre_gateway_dispatch_once(event, event.source))

    adapter.set_message_handler(_runner_handler)
    session_key = await _arrive_while_busy(adapter, _whatsapp_event("/approve"))

    assert len(calls) == 1
    assert len(handled) == 1 and handled[0] is not None and handled[0].text == "/approve"
    assert session_key not in adapter._pending_messages, "a bypass command must not be queued"


@pytest.mark.asyncio
async def test_internal_busy_arrival_bypasses_hook(monkeypatch):
    _clear_auth_env(monkeypatch)
    calls = _hook_recorder(monkeypatch, lambda event: [{"action": "skip"}])

    runner = _primary_runner()
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())
    event = _whatsapp_event("[background process finished]")
    event.internal = True

    session_key = await _arrive_while_busy(adapter, event)

    assert calls == []
    assert adapter._pending_messages.get(session_key) is event


# --- multiplex: the secondary profile's busy path ------------------------------------------


@pytest.fixture
def mux_home(tmp_path, monkeypatch):
    from agent import secret_scope

    home = tmp_path / "hh"
    (home / "profiles" / "secondary").mkdir(parents=True)
    (home / ".env").write_text("FEISHU_ALLOWED_USERS=primary-user\n")
    (home / "profiles" / "secondary" / ".env").write_text("FEISHU_ALLOWED_USERS=secondary-user\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    _clear_auth_env(monkeypatch)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield home
    secret_scope.set_multiplex_active(prev)


def _mux_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.FEISHU: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._profile_adapters = {"secondary": {}}
    runner.adapters = {}
    return runner


def _feishu_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        message_id="m-mux",
        source=SessionSource(
            platform=Platform.FEISHU,
            user_id="secondary-user",
            user_name="secondary-user",
            chat_id="oc_dm",
            chat_type="dm",
            profile=None,
        ),
    )


@pytest.mark.asyncio
async def test_secondary_profile_busy_arrival_runs_hook_once_in_its_own_scope(mux_home, monkeypatch):
    calls = _hook_recorder(monkeypatch, lambda event: [{"action": "allow"}])
    runner = _mux_runner()

    reached_cold_path = []

    async def _cold_path(event):
        # Stand-in for the runner's ``_handle_message`` body after its own pre-dispatch step:
        # replay exactly the dedupe gate it runs, then record what would reach the agent.
        event = await runner._hm_pre_gateway_dispatch_once(event, event.source)
        reached_cold_path.append(event)
        return "ok"

    runner._handle_message = _cold_path

    adapter = _busy_adapter(
        Platform.FEISHU, runner, runner._make_profile_pre_gateway_dispatch_handler("secondary"))
    adapter._owner_profile = "secondary"
    session_key = await _arrive_while_busy(adapter, _feishu_event("follow-up while busy"))

    secondary_home = (mux_home / "profiles" / "secondary").resolve()
    assert len(calls) == 1
    assert calls[0].home.resolve() == secondary_home, "hook must run under the owning profile's scope"
    queued = adapter._pending_messages.get(session_key)
    assert queued is not None

    # Drain through the secondary profile's cold handler, as the adapter's message handler would.
    adapter._pending_messages.pop(session_key)
    await runner._make_profile_message_handler("secondary")(queued)

    assert len(reached_cold_path) == 1 and reached_cold_path[0] is not None
    assert len(calls) == 1, "crossing back into the profile's cold path must not re-run the hook"


# --- control traffic and a real plugin, through the real runner ------------------------------
#
# The busy path applies the hook in the same order as the idle path (``_hm_admit_event``): before
# auth and before any slash-command routing. ``allow``/``None`` must therefore leave ``/stop`` and
# ``/approve`` intact so they are still routed as bypass commands; ``skip``/``rewrite`` of control
# traffic is the plugin's explicit choice, exactly as it is when the session is idle.

_REAL_PLUGIN = '''
SEEN = []


def _gate(event, gateway, **kwargs):
    SEEN.append(event.text)
    gateway._busy_hook_seen.append(event.text)
    if event.text.startswith("/"):
        return None  # control traffic passes through untouched
    return {"action": "rewrite", "text": "[plugin] " + event.text}


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", _gate)
'''


def _install_real_plugin(home: Path) -> None:
    """Install and enable a ``pre_gateway_dispatch`` plugin the plugin manager discovers itself."""
    plugin_dir = home / "plugins" / "busy_gate"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text("name: busy_gate\nversion: 0.1.0\n")
    (plugin_dir / "__init__.py").write_text(_REAL_PLUGIN)
    (home / "config.yaml").write_text("plugins:\n  enabled:\n  - busy_gate\n")


@pytest.fixture
def real_plugin_runner(tmp_path, monkeypatch):
    """A runner whose hook seam is the real plugin manager, loading ``busy_gate`` from disk."""
    from hermes_cli import plugins

    home = tmp_path / "home"
    home.mkdir()
    _install_real_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    plugins._reset_plugin_managers_for_tests()
    runner = _primary_runner()
    runner._busy_hook_seen = []
    yield runner
    plugins._reset_plugin_managers_for_tests()


@pytest.mark.asyncio
async def test_real_plugin_rewrites_busy_arrival_once_through_queue_and_drain(real_plugin_runner):
    runner = real_plugin_runner
    delivered = []

    async def _agent(event, source, _quick_key, _run_generation):
        delivered.append(event.text)
        return "ok"

    runner._handle_message_with_agent = _agent
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())
    adapter.set_message_handler(runner._handle_message)

    session_key = await _arrive_while_busy(adapter, _whatsapp_event("sent mid-turn"))

    queued = adapter._pending_messages.pop(session_key)
    assert queued.text == "[plugin] sent mid-turn"
    await runner._handle_message(queued)  # the turn ends; the queue drains through the cold path

    assert runner._busy_hook_seen == ["sent mid-turn"], "the real plugin must run exactly once"
    assert delivered == ["[plugin] sent mid-turn"]


@pytest.mark.asyncio
async def test_stop_mid_turn_survives_the_hook_and_reaches_the_stop_handler(real_plugin_runner):
    runner = real_plugin_runner
    stopped = []

    async def _stop_handler(event):
        stopped.append(event.text)
        return "Stopped."

    runner._handle_stop_command = _stop_handler
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())
    adapter.set_message_handler(runner._handle_message)

    session_key = await _arrive_while_busy(adapter, _whatsapp_event("/stop"))

    assert runner._busy_hook_seen == ["/stop"]
    assert stopped == ["/stop"], "/stop must still be routed as a command, not rewritten or swallowed"
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_approve_mid_turn_survives_the_hook_and_unblocks_the_waiting_agent(real_plugin_runner):
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    runner = real_plugin_runner
    adapter = _busy_adapter(Platform.WHATSAPP, runner, runner._primary_pre_gateway_dispatch_handler())
    adapter.set_message_handler(runner._handle_message)
    event = _whatsapp_event("/approve")
    session_key = runner._session_key_for_source(event.source)
    entry = _ApprovalEntry({"command": "rm -rf build"})
    approval._gateway_queues[session_key] = [entry]
    try:
        await _arrive_while_busy(adapter, event)
    finally:
        approval._gateway_queues.pop(session_key, None)

    assert runner._busy_hook_seen == ["/approve"], "the hook runs once, not again in _handle_message"
    assert entry.event.is_set() and entry.result == "once", "/approve must reach the real handler"
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_real_plugin_installed_only_in_secondary_profile_runs_once_for_its_busy_arrival(
        mux_home):
    """The plugin manager is resolved per profile home: a plugin installed only in the secondary
    profile must fire for that profile's busy arrival, once across queue and drain."""
    from hermes_cli import plugins

    _install_real_plugin(mux_home / "profiles" / "secondary")
    plugins._reset_plugin_managers_for_tests()
    try:
        runner = _mux_runner()
        runner._busy_hook_seen = []
        drained = []

        async def _cold_path(event):
            drained.append(await runner._hm_pre_gateway_dispatch_once(event, event.source))
            return "ok"

        runner._handle_message = _cold_path
        adapter = _busy_adapter(
            Platform.FEISHU, runner, runner._make_profile_pre_gateway_dispatch_handler("secondary"))
        adapter._owner_profile = "secondary"
        session_key = await _arrive_while_busy(adapter, _feishu_event("follow-up while busy"))

        queued = adapter._pending_messages.pop(session_key)
        assert queued.text == "[plugin] follow-up while busy"
        await runner._make_profile_message_handler("secondary")(queued)

        assert runner._busy_hook_seen == ["follow-up while busy"]
        assert len(drained) == 1 and drained[0].text == "[plugin] follow-up while busy"
    finally:
        plugins._reset_plugin_managers_for_tests()

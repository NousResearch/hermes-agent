"""``register_hook("pre_gateway_dispatch", cb, fail_closed=True)``: a required gate (#79152).

An ordinary ``pre_gateway_dispatch`` callback fails open: when it raises, the message is
dispatched as if it had returned ``None``. A callback registered with ``fail_closed=True`` is a
required gate instead: when it raises or returns no valid decision (``skip``, ``rewrite`` with a
text, ``allow``) the message is not processed, and while one is registered a ``skip`` outranks an
earlier plugin's ``allow``/``rewrite``. Without the flag every result keeps today's meaning.
"""

import asyncio
import functools
import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PRIVATE_TEXT = "private words 4711"
DISPATCHED = "dispatched"


@pytest.fixture
def manager(monkeypatch):
    """A real plugin manager the gateway dispatches the hook through."""
    from hermes_cli import plugins

    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: mgr)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return mgr


def _plugin(manager, name):
    return PluginContext(PluginManifest(name=name, source="user"), manager)


def _event(text=PRIVATE_TEXT, platform=Platform.TELEGRAM):
    source = SessionSource(platform=platform, chat_id="12345", chat_type="dm", user_id="777", user_name="tester")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="m1")


def _runner():
    """A runner whose ingress gates pass and whose dispatch, after admission, answers DISPATCHED."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.session_store = None
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._admit_bot_chain = lambda _event: True
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_estop_gate = lambda _event, _source, _internal: DISPATCHED  # first step after admission
    return runner


def _raise(**_kwargs):
    raise RuntimeError("boom")


# ── a required gate that fails refuses the message ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_required_gate_that_raises_drops_the_message(manager, caplog):
    def gate(event, **_kwargs):
        raise RuntimeError("boom")

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    with caplog.at_level(logging.INFO):
        assert await _runner()._handle_message(_event()) is None

    refusals = [r.getMessage() for r in caplog.records if "Required 'pre_gateway_dispatch'" in r.getMessage()]
    assert len(refusals) == 1 and "gate" in refusals[0] and "RuntimeError" in refusals[0]
    assert PRIVATE_TEXT not in caplog.text  # plugin and hook are named, the message is not


@pytest.mark.parametrize("returned", [
    None, "allow", ["skip"], {"action": "maybe"}, {"reason": "no action"}, {"action": "ALLOW"},
    {"action": "rewrite"}, {"action": "rewrite", "text": 5}, {"action": "rewrite", "text": ""},
    {"action": ["skip"]}, {"action": {"a": 1}},  # unhashable: an invalid decision, not a raise
])
@pytest.mark.asyncio
async def test_a_required_gate_without_a_valid_decision_drops_the_message(manager, returned, caplog):
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", lambda **_: returned, fail_closed=True)
    with caplog.at_level(logging.INFO):
        assert await _runner()._handle_message(_event()) is None
    assert "returned no valid decision" in caplog.text and "raised" not in caplog.text


@pytest.mark.asyncio
async def test_an_async_required_gate_that_raises_or_times_out_drops_the_message(manager):
    """``async def`` gates are awaited on the loop; their own ``TimeoutError`` is a failure too
    (the hook has no host deadline, so it is never mistaken for one)."""
    async def raises(**_kwargs):
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def own_timeout(**_kwargs):
        await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)

    async def no_decision(**_kwargs):
        return None

    for gate in (raises, own_timeout, no_decision):
        handle = _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
        assert await _runner()._handle_message(_event()) is None, gate.__name__
        handle.dispose()


@pytest.mark.asyncio
async def test_a_gates_own_timeout_is_not_reported_as_a_host_deadline(manager, caplog):
    def own_timeout(**_kwargs):
        raise TimeoutError("upstream lookup")

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", own_timeout, fail_closed=True)
    with caplog.at_level(logging.INFO):
        assert await _runner()._handle_message(_event()) is None
    assert "raised TimeoutError" in caplog.text and "timed out after" not in caplog.text


@pytest.mark.asyncio
async def test_the_refusal_line_names_the_function_never_its_arguments_or_the_error_text(manager, caplog):
    def secret_gate(event, *, api_key, **_kwargs):
        raise ValueError(f"cannot parse {event.text!r} with {api_key}")

    gate = functools.partial(secret_gate, api_key="SECRET-KEY-123")
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    with caplog.at_level(logging.INFO):
        assert await _runner()._handle_message(_event()) is None

    refusals = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Required ")]
    assert len(refusals) == 1, refusals
    assert "callback secret_gate raised" in caplog.text  # the existing failure line names it too, not "gate"
    assert f"{__name__}.test_the_refusal_line_names_the_function_never_its_arguments_or_the_error_text" \
           ".<locals>.secret_gate raised ValueError" in refusals[0]
    assert "SECRET-KEY-123" not in refusals[0] and PRIVATE_TEXT not in refusals[0] and "functools" not in refusals[0]


@pytest.mark.asyncio
async def test_a_required_gate_keeps_the_meaning_of_valid_decisions(manager):
    decision = {}
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", lambda **_: decision["value"], fail_closed=True)
    runner = _runner()

    decision["value"] = {"action": "skip", "reason": "policy"}
    assert await runner._handle_message(_event()) is None
    decision["value"] = {"action": "allow"}
    assert await runner._handle_message(_event()) == DISPATCHED
    decision["value"] = {"action": "rewrite", "text": "rewritten"}
    rewritten = await runner._hm_pre_gateway_dispatch_hook(_event(), _event().source)
    assert rewritten is not None and rewritten.text == "rewritten"


# ── without the flag nothing changes ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_ordinary_gate_that_raises_still_lets_the_message_through(manager):
    _plugin(manager, "observer").register_hook("pre_gateway_dispatch", _raise)
    assert await _runner()._handle_message(_event()) == DISPATCHED


@pytest.mark.asyncio
async def test_without_a_required_gate_the_first_recognized_action_still_wins(manager):
    _plugin(manager, "first").register_hook("pre_gateway_dispatch", lambda **_: {"action": "allow"})
    _plugin(manager, "second").register_hook("pre_gateway_dispatch", lambda **_: {"action": "skip"})
    assert await _runner()._handle_message(_event()) == DISPATCHED


@pytest.mark.asyncio
async def test_without_a_required_gate_a_failed_invocation_still_dispatches(manager, monkeypatch):
    async def broken(_name, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", broken)
    _plugin(manager, "observer").register_hook("pre_gateway_dispatch", lambda **_: None)
    assert await _runner()._handle_message(_event()) == DISPATCHED


# ── two plugins ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("first, second, expected", [
    # (callback, required?) in registration order
    (({"action": "allow"}, False), ({"action": "skip"}, True), None),
    (({"action": "rewrite", "text": "x"}, False), (RuntimeError, True), None),
    (({"action": "allow"}, True), ({"action": "skip"}, False), None),  # a skip outranks while a gate is required
    (({"action": "allow"}, True), (RuntimeError, False), DISPATCHED),  # the ordinary one still fails open
    ((RuntimeError, False), ({"action": "allow"}, True), DISPATCHED),
])
@pytest.mark.asyncio
async def test_two_plugins_one_required(manager, first, second, expected):
    def returning(result):
        return lambda **_: result

    for name, (result, required) in (("one", first), ("two", second)):
        callback = _raise if result is RuntimeError else returning(result)
        _plugin(manager, name).register_hook("pre_gateway_dispatch", callback, fail_closed=required)
    assert await _runner()._handle_message(_event()) == expected


@pytest.mark.parametrize("registrations, expected", [
    # (result, required?) in registration order; the event arrives as "orig"
    ([({"action": "allow"}, False), ({"action": "rewrite", "text": "redacted"}, True)], "redacted"),
    ([({"action": "rewrite", "text": "other"}, False), ({"action": "rewrite", "text": "redacted"}, True)], "redacted"),
    ([({"action": "allow"}, True), ({"action": "rewrite", "text": "injected"}, False)], "orig"),
    ([({"action": "rewrite", "text": "injected"}, False), ({"action": "allow"}, True)], "orig"),
    ([({"action": "rewrite", "text": "first"}, True), ({"action": "rewrite", "text": "second"}, True)], "first"),
    ([({"action": "skip"}, False), ({"action": "allow"}, True)], None),
    ([({"action": "allow"}, True), ({"action": "skip"}, True)], None),
    ([({"action": "rewrite", "text": "redacted"}, True), ({"action": "skip"}, False)], None),
])
@pytest.mark.asyncio
async def test_a_required_decision_binds_over_ordinary_results(manager, registrations, expected):
    """Processed is only text every required gate saw and allowed: a required rewrite applies,
    an ordinary plugin's rewrite does not (no required gate saw its text), any skip drops."""
    for i, (result, required) in enumerate(registrations):
        _plugin(manager, f"p{i}").register_hook(
            "pre_gateway_dispatch", lambda __r=result, **_: __r, fail_closed=required)
    out = await _runner()._hm_pre_gateway_dispatch_hook(_event("orig"), _event().source)
    assert (None if out is None else out.text) == expected


class _EmptySkip(dict):
    """Reads as a skip but holds no items, so it is falsy."""

    def get(self, key, default=None):
        return "skip" if key == "action" else default


class _Flip(dict):
    """Says skip on the first read of its action and rewrite on every later one."""

    def __init__(self):
        super().__init__(action="skip")
        self.reads = 0

    def get(self, key, default=None):
        if key == "action":
            self.reads += 1
            return "skip" if self.reads == 1 else "rewrite"
        return "FLIPPED" if key == "text" else default


@pytest.mark.parametrize("odd", [_EmptySkip, _Flip])
@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.asyncio
async def test_an_odd_skip_object_cannot_turn_a_refusal_into_dispatch(manager, odd, with_gate):
    _plugin(manager, "plain").register_hook("pre_gateway_dispatch", lambda **_: odd())
    if with_gate:
        _plugin(manager, "policy").register_hook(
            "pre_gateway_dispatch", lambda **_: {"action": "skip", "reason": "no"}, fail_closed=True)
    assert await _runner()._hm_pre_gateway_dispatch_hook(_event("orig"), _event().source) is None


@pytest.mark.parametrize("registrations, expected", [
    # an unflagged plugin returning a RequiredDecision of its own counts as an ordinary result
    ([("forged-rewrite", False), ({"action": "allow"}, True)], "orig"),
    ([({"action": "allow"}, True), ("forged-rewrite", False)], "orig"),
    ([({"action": "rewrite", "text": "x"}, False), ("forged-allow", False)], "x"),  # no gate: first result wins
])
@pytest.mark.asyncio
async def test_a_forged_required_decision_counts_as_an_ordinary_result(manager, registrations, expected):
    from hermes_cli.plugins_dispatch import RequiredDecision

    forged = {"forged-rewrite": lambda **_: RequiredDecision(action="rewrite", text="FORGED"),
              "forged-allow": lambda **_: RequiredDecision(action="allow")}
    for i, (result, required) in enumerate(registrations):
        callback = forged[result] if isinstance(result, str) else (lambda __r=result, **_: __r)
        _plugin(manager, f"p{i}").register_hook("pre_gateway_dispatch", callback, fail_closed=required)
    out = await _runner()._hm_pre_gateway_dispatch_hook(_event("orig"), _event().source)
    assert out is not None and out.text == expected


@pytest.mark.asyncio
async def test_every_callback_judges_the_event_as_it_arrived(manager):
    seen = []
    _plugin(manager, "rewriter").register_hook("pre_gateway_dispatch", lambda **_: {"action": "rewrite", "text": "x"})

    def gate(event, **_kwargs):
        seen.append(event.text)
        return {"action": "allow"}

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    out = await _runner()._hm_pre_gateway_dispatch_hook(_event("orig"), _event().source)
    assert seen == ["orig"] and out.text == "orig"


@pytest.mark.asyncio
async def test_a_failed_invocation_with_a_required_gate_drops_the_message(manager, monkeypatch):
    async def broken(_name, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", broken)
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", lambda **_: {"action": "allow"}, fail_closed=True)
    assert await _runner()._handle_message(_event()) is None


@pytest.mark.asyncio
async def test_an_unreadable_registry_keeps_the_old_behaviour(manager, monkeypatch):
    """Known limit: when the invocation fails and the registry cannot be read either, the gateway
    cannot know a gate was required and dispatches, as without one."""
    async def broken(_name, **_kwargs):
        raise RuntimeError("registry unavailable")

    def unreadable(_hook):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", broken)
    monkeypatch.setattr("hermes_cli.plugins.has_fail_closed_hook", unreadable)
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", lambda **_: {"action": "allow"}, fail_closed=True)
    assert await _runner()._handle_message(_event()) == DISPATCHED


# ── the registration ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_gate_lasts_as_long_as_its_registration(manager):
    handle = _plugin(manager, "policy").register_hook("pre_gateway_dispatch", _raise, fail_closed=True)
    assert manager.has_fail_closed_hook("pre_gateway_dispatch")

    handle.dispose()  # an unloaded or disabled plugin is no required gate
    assert not manager.has_fail_closed_hook("pre_gateway_dispatch")
    assert not manager.has_hook("pre_gateway_dispatch")
    _plugin(manager, "observer").register_hook("pre_gateway_dispatch", _raise)
    assert await _runner()._handle_message(_event()) == DISPATCHED


@pytest.mark.asyncio
async def test_the_mark_belongs_to_the_registration_not_to_the_function(manager):
    def shared(**_kwargs):
        return None  # no decision: refused where required, no opinion where ordinary

    required = _plugin(manager, "a").register_hook("pre_gateway_dispatch", shared, fail_closed=True)
    ordinary = _plugin(manager, "b").register_hook("pre_gateway_dispatch", shared)
    payload = {"event": None, "gateway": None, "session_store": None}
    assert len(manager.invoke_hook("pre_gateway_dispatch", **payload)) == 1  # only the required entry refuses
    assert await _runner()._handle_message(_event()) is None

    required.dispose()  # the other plugin's registration of the same function stays ordinary
    assert not manager.has_fail_closed_hook("pre_gateway_dispatch")
    assert await _runner()._handle_message(_event()) == DISPATCHED
    ordinary.dispose()

    again = _plugin(manager, "a").register_hook("pre_gateway_dispatch", shared)
    _plugin(manager, "a").register_hook("pre_gateway_dispatch", shared, fail_closed=True)
    assert await _runner()._handle_message(_event()) is None
    again.dispose()
    assert manager.has_fail_closed_hook("pre_gateway_dispatch")


@pytest.mark.asyncio
async def test_an_unload_during_dispatch_does_not_skip_the_gate(manager):
    """An earlier plugin unloaded while its callback awaits must not shift the gate out of the loop."""
    handles = {}

    async def earlier(**_kwargs):
        handles["earlier"].dispose()
        await asyncio.sleep(0)

    handles["earlier"] = _plugin(manager, "other").register_hook("pre_gateway_dispatch", earlier)
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", lambda **_: {"action": "skip"}, fail_closed=True)
    assert await _runner()._handle_message(_event()) is None


@pytest.mark.asyncio
async def test_a_gate_registered_during_dispatch_decides_from_the_next_event(manager):
    def late(**_kwargs):
        raise RuntimeError("boom")

    async def first(**_kwargs):
        _plugin(manager, "late").register_hook("pre_gateway_dispatch", late, fail_closed=True)
        await asyncio.sleep(0)
        return {"action": "allow"}

    _plugin(manager, "first").register_hook("pre_gateway_dispatch", first)
    assert await _runner()._handle_message(_event()) == DISPATCHED
    assert await _runner()._handle_message(_event()) is None


def test_the_ledger_fallback_keeps_the_flag():
    """Hooks a memory provider registers go through ``_register_fallback_hook``."""
    manager = PluginManager()
    manager._register_fallback_hook(_plugin(manager, "memory"), None, "pre_gateway_dispatch", _raise, fail_closed=True)
    assert manager.has_fail_closed_hook("pre_gateway_dispatch")


@pytest.mark.parametrize("hook", ["pre_llm_call", "post_gateway_admission"])
def test_fail_closed_on_another_hook_is_ignored_with_a_warning(hook, caplog):
    manager = PluginManager()
    _plugin(manager, "observer").register_hook(hook, lambda **_: None, fail_closed=True)
    assert not manager.has_fail_closed_hook(hook)
    assert "fail_closed" in caplog.text


def test_fail_closed_on_pre_tool_call_is_accepted_silently(caplog):
    """``pre_tool_call`` already fails closed for every callback; the flag changes nothing there."""
    manager = PluginManager()
    _plugin(manager, "guard").register_hook("pre_tool_call", lambda **_: None, fail_closed=True)
    assert "fail_closed" not in caplog.text


def test_the_sync_dispatcher_applies_the_same_contract():
    def gate(**_kwargs):
        raise RuntimeError("boom")

    manager = PluginManager()
    _plugin(manager, "observer").register_hook("pre_gateway_dispatch", _raise)
    assert manager.invoke_hook("pre_gateway_dispatch", event=None, gateway=None, session_store=None) == []

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    results = manager.invoke_hook("pre_gateway_dispatch", event=None, gateway=None, session_store=None)
    assert [r["action"] for r in results] == ["skip"] and "RuntimeError" in results[0]["reason"]


# ── the paths a message takes ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_message_replayed_after_startup_restore_meets_the_gate(manager):
    """A message that arrives during startup restore is buffered before the hook, then replayed
    through ``handle_message``: the replay is where the gate decides."""
    calls = []

    def gate(event, **_kwargs):
        calls.append(event.message_id)
        raise RuntimeError("boom")

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    runner = _runner()
    runner._startup_restore_in_progress = True
    buffered = []
    runner._queue_startup_restore_event = buffered.append

    assert await runner._handle_message(_event()) is None and len(buffered) == 1 and calls == []
    replay = buffered[0]
    replay._hermes_startup_restore_replay = True
    assert await runner._handle_message(replay) is None and calls == ["m1"]


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.parametrize("text", ["/status", "/approve"])
@pytest.mark.parametrize("gate, answered", [(_raise, []), (lambda **_: {"action": "allow"}, [DISPATCHED])])
@pytest.mark.asyncio
async def test_a_command_typed_while_the_session_is_busy_meets_the_gate(manager, text, gate, answered):
    """A slash command that bypasses the busy session's guard is dispatched inline through the
    runner's ``_handle_message``, so the gate decides there too: refused, nothing is answered."""
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    adapter = _Adapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._busy_text_mode = ""
    sent = []

    async def _send_with_retry(chat_id, content, **_kwargs):
        sent.append(content)

    adapter._send_with_retry = _send_with_retry
    adapter._message_handler = _runner()._handle_message
    event = _event(text)
    adapter._active_sessions[build_session_key(event.source)] = asyncio.Event()

    await adapter.handle_message(event)
    assert sent == answered


@pytest.mark.parametrize("fail_closed, committed", [(True, []), (False, ["d-raise"])])
@pytest.mark.asyncio
async def test_a_webhook_delivery_meets_the_gate(manager, fail_closed, committed):
    """Receipt-only ingress (``admit_producer``, webhook and relay) runs the hook through the same
    function: a delivery a required gate refuses is acknowledged and commits no admission, where an
    ordinary callback that raises still lets it through."""
    from aiohttp.test_utils import TestClient, TestServer

    from tests.fixtures.storage_holder import raw_holder
    from tests.gateway.test_pre_gateway_dispatch import _post_webhook, _webhook_app

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", _raise, fail_closed=fail_closed)
    adapter, app = _webhook_app()
    async with TestClient(TestServer(app)) as client:
        authority = adapter._message_handler.__self__.session_authority
        assert (await _post_webhook(client, "hello", "d-raise")).status == 202
        rows = raw_holder(authority.persistence)._read_all("SELECT request_id FROM session_admissions")
    assert [row["request_id"] for row in rows] == committed


# ── the refusal line, the guard around the binding step, a bounded hook ───────────────────────


@pytest.mark.asyncio
async def test_a_refusal_that_repeats_is_warned_once_and_still_refuses_every_message(manager, caplog):
    """A gate whose backend is down fails the same way for every message. The refusal line follows
    the dispatcher's warn-once rule (#111922): WARNING for the first, DEBUG for identical repeats,
    WARNING again for a new cause. Each refused message keeps its own ``skip`` line."""
    causes = iter([RuntimeError, RuntimeError, RuntimeError, ValueError])

    def gate(event, **_kwargs):
        raise next(causes)("backend down")

    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
    runner = _runner()
    with caplog.at_level(logging.DEBUG):
        assert [await runner._handle_message(_event()) for _ in range(4)] == [None] * 4

    refusals = [r.levelname for r in caplog.records if r.getMessage().startswith("Required ")]
    assert refusals == ["WARNING", "DEBUG", "DEBUG", "WARNING"]
    skips = [r.levelname for r in caplog.records if r.getMessage().startswith("pre_gateway_dispatch skip")]
    assert skips == ["INFO"] * 4


@pytest.mark.parametrize("with_gate, expected", [(False, DISPATCHED), (True, None)])
@pytest.mark.asyncio
async def test_a_fault_in_the_binding_step_counts_as_a_failed_invocation(manager, monkeypatch, with_gate, expected):
    """The binding step runs inside the guard around the hook invocation: without a required gate a
    fault there never stops dispatch, with one the message is refused."""
    from gateway import run_inbound_hooks

    def broken(_results):
        raise ImportError("plugin layer unavailable")

    monkeypatch.setattr(run_inbound_hooks, "_bind_required_decisions", broken)
    _plugin(manager, "observer").register_hook("pre_gateway_dispatch", lambda **_: {"action": "allow"})
    if with_gate:
        _plugin(manager, "policy").register_hook(
            "pre_gateway_dispatch", lambda **_: {"action": "allow"}, fail_closed=True)
    assert await _runner()._handle_message(_event()) == expected


@pytest.mark.parametrize("hook_set", ["_HOOK_TIMEOUT_BOUNDED_HOOKS", "_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS"])
@pytest.mark.asyncio
async def test_a_required_gate_refuses_in_its_own_result_type_if_the_hook_were_bounded(manager, monkeypatch, hook_set):
    """``pre_gateway_dispatch`` has no host deadline today. If it gets one (fail-open or with the
    hook-wide veto ``pre_tool_call`` has), a required gate that times out or raises still answers
    with a required ``skip`` in both dispatchers, not with a ``block`` the gateway does not read."""
    import threading

    from hermes_cli import plugins, plugins_dispatch
    from hermes_cli.plugins_dispatch import RequiredDecision

    monkeypatch.setattr(plugins_dispatch, hook_set, getattr(plugins_dispatch, hook_set) | {"pre_gateway_dispatch"})
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.05)

    async def slow(**_kwargs):
        await asyncio.Event().wait()  # never set: only the deadline ends it

    handle = _plugin(manager, "policy").register_hook("pre_gateway_dispatch", slow, fail_closed=True)
    assert await _runner()._handle_message(_event()) is None  # the awaited gate ran into the deadline
    handle.dispose()
    _plugin(manager, "policy").register_hook("pre_gateway_dispatch", _raise, fail_closed=True)
    assert await _runner()._handle_message(_event()) is None

    release = threading.Event()
    try:
        for gate in (lambda **_: release.wait(10), _raise):  # the sync dispatcher: a deadline, then a raise
            sync_manager = PluginManager()
            _plugin(sync_manager, "policy").register_hook("pre_gateway_dispatch", gate, fail_closed=True)
            results = sync_manager.invoke_hook("pre_gateway_dispatch", event=None, gateway=None, session_store=None)
            assert [(type(r), r["action"]) for r in results] == [(RequiredDecision, "skip")]
    finally:
        release.set()  # let the abandoned worker go

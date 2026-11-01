"""Real gateway transport for the auto_decompose-off instruction wake.

When ``kanban.auto_decompose`` is disabled the routing gate never calls the
auxiliary model: it hands the existing decomposition prompt to exactly ONE
eligible subscriber as a wake/instruction. ``deliver_kanban_instruction_wake``
is that transport. These tests drive it for real — real ``deliver_wake`` /
``admit_internal_event`` / ``MessageEvent`` construction, fake runner and
adapters only (the outbound HTTP self-post boundary is stubbed, the established
convention in ``test_kanban_notifier_apiserver_wake``) — pinning the contract
the review demanded:

* a push-capable adapter receives the INTERNAL wake (``handle_message`` with an
  ``internal`` synthetic event) — a passive ``send()`` never reaches the agent,
  and a ``delivery_mode="notify"`` subscription must wake too;
* a stateless adapter (``supports_async_delivery=False``) wakes via the raw
  session self-post, never ``handle_message``;
* several destinations for ONE profile get exactly ONE wake — never a fan-out;
* an unusable first destination falls through to the next, and when no
  destination can accept the wake the result is an honest ``delivered=False``
  naming the reason (never a silent passive "sent").

The transport takes plain subscription dicts and never opens the board DB, so
delivering an instruction can never change the task graph by itself.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.kanban_watchers import deliver_kanban_instruction_wake

INSTRUCTION = "Kanban decomposition requested — apply with kanban_decompose"


# --------------------------------------------------------------------------- fakes


class _Runner:
    """Minimal GatewayRunner surface the transport actually touches.

    ``multiplex_profiles=False`` keeps ``served`` unset, so destinations wake in
    this process without a profile-home scope.
    """

    def __init__(self, adapters, home):
        self.config = SimpleNamespace(multiplex_profiles=False, profile_routes=[])
        self.adapters = dict(adapters)   # {Platform: adapter}
        self._adapters = dict(adapters)
        self._home = home
        self.authz_calls = []

    def _authorization_adapter(self, platform, profile):
        self.authz_calls.append((platform, profile))
        return self._adapters.get(platform)

    def _resolve_profile_home_for_source(self, source):
        return self._home


class _PushAdapter:
    """Push-capable (default) adapter: wakes arrive as internal handle_message events."""

    def __init__(self, *, accept=True):
        self.events = []
        self.sent = []          # a passive send must NEVER be the delivery
        self._accept = accept

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text})

    async def handle_message(self, event):
        self.events.append(event)
        if self._accept:
            event._gateway_accepted = True


class _StatelessAdapter:
    """API-server-like adapter: no push, wakes go out as a raw-session self-post."""

    supports_async_delivery = False

    def __init__(self):
        self.events = []
        self.sent = []
        self._host = "127.0.0.1"
        self._port = 8642
        self._api_key = "k"
        self._model_name = "hermes"

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text})

    async def handle_message(self, event):
        self.events.append(event)


def _sub(platform, chat_id, **extra):
    sub = {"platform": platform, "chat_id": chat_id, "chat_type": "dm",
           "notifier_profile": "worker", "delivery_mode": "notify",
           "delivery_metadata": {}}
    sub.update(extra)
    return sub


def _run(runner, **kw):
    return asyncio.run(deliver_kanban_instruction_wake(
        runner, profile="worker", task_id="t_wake", text=INSTRUCTION, **kw))


@pytest.fixture
def profile_scope(monkeypatch):
    """Hermetic profile-home scope: the wake itself stays real."""
    import gateway.run as run_mod
    monkeypatch.setattr(run_mod, "_async_profile_runtime_scope",
                        lambda home: contextlib.nullcontext())


@pytest.fixture
def self_posts(monkeypatch):
    """Stub only the outbound HTTP boundary of the raw-session self-post."""
    posts: list[dict] = []

    async def fake_self_post(adapter, *, text, session_id, **kw):
        posts.append({"text": text, "session_id": session_id, **kw})

    import gateway.wake as wake_mod
    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    return posts


# --------------------------------------------------------------------------- push adapters


def test_push_adapter_gets_one_internal_wake_even_for_notify_mode(tmp_path, profile_scope):
    """A push-capable adapter must receive the INTERNAL wake — the old bug gave
    it only a passive ``send()`` that never reaches the agent — and a
    notify-only subscription must wake too."""
    adapter = _PushAdapter()
    runner = _Runner({Platform.TELEGRAM: adapter}, tmp_path)

    delivered, detail = _run(runner, subs=[_sub("telegram", "111")])

    assert delivered is True
    assert len(adapter.events) == 1, "exactly one internal wake"
    event = adapter.events[0]
    assert event.text == INSTRUCTION
    assert event.internal is True
    assert event.source.chat_id == "111"
    assert event.source.platform == Platform.TELEGRAM
    assert event.metadata.get("notification_category") == "diagnostic"
    assert adapter.sent == [], "a passive send is never the delivery"
    assert "single push wake" in detail
    assert "telegram/111" in detail


def test_stateless_adapter_wakes_via_the_raw_session_self_post(
        tmp_path, self_posts):
    """``supports_async_delivery=False`` adapters take the raw-session self-post
    leg — keyed to the subscription's destination, never handle_message."""
    adapter = _StatelessAdapter()
    runner = _Runner({Platform.API_SERVER: adapter}, tmp_path)

    delivered, detail = _run(runner, subs=[_sub("api_server", "sess-1")])

    assert delivered is True
    assert adapter.events == [], "stateless adapters must not take handle_message"
    assert adapter.sent == [], "a passive send is never the delivery"
    assert len(self_posts) == 1
    assert self_posts[0]["session_id"] == "sess-1"
    assert self_posts[0]["text"] == INSTRUCTION
    assert self_posts[0]["notification_category"] == "diagnostic"
    assert "session self-post wake" in detail
    assert "api_server/sess-1" in detail


# --------------------------------------------------------------------------- no fan-out


def test_two_destinations_for_one_profile_get_exactly_one_wake(tmp_path, profile_scope):
    """The routing gate collapses several chats of ONE profile into one route;
    the transport must stop at the first accepted wake, never broadcast."""
    adapter = _PushAdapter()
    runner = _Runner({Platform.TELEGRAM: adapter}, tmp_path)

    delivered, detail = _run(runner, subs=[
        _sub("telegram", "1"), _sub("telegram", "2"),
    ])

    assert delivered is True
    assert len(adapter.events) == 1, "one profile receives ONE wake, not a fan-out"
    assert adapter.events[0].source.chat_id == "1", "stops at the first accepted destination"
    assert adapter.sent == []
    assert "deliberately not fanned out" in detail
    assert "1 other destination" in detail


def test_unusable_first_destination_falls_through_to_the_next(tmp_path, profile_scope):
    """A destination with no live adapter is skipped; the next one delivers."""
    adapter = _PushAdapter()
    runner = _Runner({Platform.TELEGRAM: adapter}, tmp_path)  # no discord adapter

    delivered, detail = _run(runner, subs=[
        _sub("discord", "9"), _sub("telegram", "3"),
    ])

    assert delivered is True
    assert len(adapter.events) == 1
    assert adapter.events[0].source.chat_id == "3"
    assert "telegram/3" in detail


def test_all_destinations_unwakable_reports_undelivered_never_a_passive_send(
        tmp_path, profile_scope):
    """When no adapter accepts the internal wake the answer is an honest
    ``delivered=False`` naming the failure — not a silent send()."""
    adapter = _PushAdapter(accept=False)   # never admits the internal event
    runner = _Runner({Platform.TELEGRAM: adapter}, tmp_path)

    delivered, detail = _run(runner, subs=[_sub("telegram", "4")])

    assert delivered is False
    assert "could not wake" in detail
    assert "WakeNotAccepted" in detail
    assert adapter.sent == [], "must never fall back to a passive send"


def test_no_subscription_destination_is_not_delivered(tmp_path, profile_scope):
    runner = _Runner({}, tmp_path)
    delivered, detail = _run(runner, subs=[])
    assert delivered is False
    assert "no live subscription destination" in detail


# ------------------------------------------------- gateway-side delivery surfaces
#
# Two process-level surfaces make the wake REACHABLE, not merely implemented:
# ``install_kanban_instruction_transport`` (called at gateway start from
# ``GatewayStartupMixin._start_spawn_background_watchers``) binds the in-process
# transport, and ``decompose_instruction_control_verb`` is the
# ``deliver-decompose-instruction`` control-socket handler the shared
# CLI/dashboard backend calls when it holds no adapters. Both marshal the wake
# onto a RUNNING gateway loop from a caller thread and report ``delivered``
# only for the confirmed result.


@pytest.fixture
def gateway_loop():
    """A real running event loop on a background thread, as the gateway has."""
    import threading
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_installed_transport_confirms_a_wake_from_another_thread(
        tmp_path, profile_scope, gateway_loop, monkeypatch):
    """The installed transport marshals onto the gateway loop and confirms a
    REAL wake; from ON that loop it refuses honestly instead of deadlocking."""
    from gateway.kanban_watchers import install_kanban_instruction_transport
    from hermes_cli import kanban_decompose as decomp

    # Save/restore the process-global transport around this test.
    monkeypatch.setattr(decomp, "_INSTRUCTION_TRANSPORT",
                        decomp._INSTRUCTION_TRANSPORT)

    adapter = _PushAdapter()
    runner = _Runner({Platform.TELEGRAM: adapter}, tmp_path)

    async def _install():
        install_kanban_instruction_transport(runner)

    asyncio.run_coroutine_threadsafe(_install(), gateway_loop).result(timeout=10)
    transport = decomp._INSTRUCTION_TRANSPORT
    assert transport is not None, "gateway start must install the transport"

    # Caller OUTSIDE the gateway loop (a worker/CLI context): confirmed wake.
    delivered, detail = transport(
        profile="worker", task_id="t_inst", text=INSTRUCTION,
        subs=[_sub("telegram", "77")])
    assert delivered is True, detail
    assert len(adapter.events) == 1
    assert adapter.events[0].text == INSTRUCTION
    assert adapter.sent == [], "a passive send is never the delivery"

    # Caller ON the gateway loop: cannot block for confirmation, so it must
    # say so (delivered=False) rather than queue-and-claim delivery.
    async def _on_loop():
        return decomp._INSTRUCTION_TRANSPORT(
            profile="worker", task_id="t_inst", text=INSTRUCTION,
            subs=[_sub("telegram", "77")])
    refused, why = asyncio.run_coroutine_threadsafe(
        _on_loop(), gateway_loop).result(timeout=10)
    assert refused is False
    assert "own event loop" in why
    assert len(adapter.events) == 1, "the refusal must not fire a second wake"


def test_control_verb_confirms_delivery_and_validates_params(
        tmp_path, profile_scope, gateway_loop):
    """``deliver-decompose-instruction`` — the reachable path for a process
    that holds no adapters — confirms a real wake and rejects bad params
    without touching any adapter."""
    from gateway.kanban_watchers import decompose_instruction_control_verb

    adapter = _PushAdapter()
    runner = _Runner({Platform.TELEGRAM: adapter}, tmp_path)
    handler = decompose_instruction_control_verb(runner, gateway_loop)

    # Malformed params: refused BEFORE any delivery attempt.
    bad = handler({})
    assert bad["delivered"] is False
    assert "required" in bad["detail"]
    assert adapter.events == [], "validation failure must not deliver"

    # Valid params: the coroutine is marshalled onto the gateway loop and the
    # answer waits for the CONFIRMED wake.
    ok = handler({"profile": "worker", "task_id": "t_ctrl",
                  "text": INSTRUCTION, "subs": [_sub("telegram", "88")]})
    assert ok["delivered"] is True, ok["detail"]
    assert len(adapter.events) == 1
    assert adapter.events[0].source.chat_id == "88"
    assert adapter.sent == []

    # delivered=False stays False when the gateway cannot wake the profile —
    # the detail must name the failure, never claim a handoff.
    runner2 = _Runner({Platform.TELEGRAM: _PushAdapter(accept=False)}, tmp_path)
    handler2 = decompose_instruction_control_verb(runner2, gateway_loop)
    refused = handler2({"profile": "worker", "task_id": "t_ctrl",
                        "text": INSTRUCTION, "subs": [_sub("telegram", "99")]})
    assert refused["delivered"] is False
    assert "could not wake" in refused["detail"]

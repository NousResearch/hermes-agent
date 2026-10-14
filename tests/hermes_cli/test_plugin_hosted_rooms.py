"""Behaviour contract for ``ctx.hosted_rooms`` — the plugin facade that admits an authenticated
Telegram group message into a hosted Group Chat room and reads back attributed turn outcomes.

Every test drives the real ``HostedRoomService`` over a temp SQLite store; only the per-profile
agent RPC is faked (it settles, or never settles, a member turn).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from hermes_cli import plugin_hosted_rooms as facade_mod
from hermes_cli.plugin_hosted_rooms import (
    FACADE_VERSION,
    HostedRoomFacade,
    TelegramRoomBinding,
)
from hermes_constants import get_hermes_home
from tui_gateway import methods_groups
from tui_gateway.hosted_room_service import HostedRoomService

PLUGIN_ID = "group-relay"
OWNER = 1384262162
CHAT = -1001234567890
TOPIC = 42


class _SettlingRPC:
    """Per-profile session backend that settles every member turn immediately."""

    def __init__(self, *, settle: bool = True) -> None:
        self.sessions: dict = {}
        self.settle = settle
        self.prompts: list[str] = []

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(self, *, profile, prompt, on_terminal, **_kwargs):
        self.prompts.append(prompt)
        if self.settle:
            on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def history(self, **_kwargs):
        return []

    def info(self, **_kwargs):
        return {"active": False, "task_id": None}

    def interrupt(self, **_kwargs):
        return {"interrupted": True}


def _grant(*capabilities: str) -> None:
    path = get_hermes_home() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"granted_capabilities": list(capabilities)}
    path.write_text(json.dumps({"plugins": {"entries": {PLUGIN_ID: entry}}}), encoding="utf-8")


def _make_service(tmp_path: Path, *, settle: bool = True) -> HostedRoomService:
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=None)
    service = HostedRoomService(server, db_path=tmp_path / "shared-state.db")
    rpc = _SettlingRPC(settle=settle)
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "agent-tester")
    service.create_room(
        room_id="relay-room",
        name="Group relay",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "agent-tester", "profile": "agent-tester", "handle": "tester"},
        ],
    )
    return service


@pytest.fixture
def running_service(tmp_path, monkeypatch):
    """A started process-owned service, published where the gateway publishes it."""
    service = _make_service(tmp_path)
    monkeypatch.setattr(methods_groups, "_service", service)
    service.start()
    yield service
    service.stop(timeout=5.0)


@pytest.fixture
def idle_service(tmp_path, monkeypatch):
    """A service whose worker is not driving turns, so tests control every transition."""
    service = _make_service(tmp_path, settle=False)
    monkeypatch.setattr(methods_groups, "get_hosted_room_service", lambda: service)
    return service


def _binding(**overrides) -> TelegramRoomBinding:
    fields = {"room_id": "relay-room", "chat_id": CHAT, "thread_id": TOPIC,
              "sender_ids": frozenset({OWNER})}
    fields.update(overrides)
    return TelegramRoomBinding(**fields)


def _update(*, update_id=9001, message_id=77, text="@tester summarize the plan", chat_id=CHAT,
            chat_type="supergroup", thread_id=TOPIC, is_topic=True, sender_id=OWNER,
            is_bot=False, edited=False, **message_extra):
    message = SimpleNamespace(
        message_id=message_id, text=text,
        chat=SimpleNamespace(id=chat_id, type=chat_type),
        from_user=SimpleNamespace(id=sender_id, is_bot=is_bot),
        message_thread_id=thread_id, is_topic_message=is_topic,
        sender_chat=None, via_bot=None, forward_origin=None, forward_date=None,
    )
    for key, value in message_extra.items():
        setattr(message, key, value)
    if edited:
        return SimpleNamespace(update_id=update_id, message=None, edited_message=message)
    return SimpleNamespace(update_id=update_id, message=message, edited_message=None)


def _events(service: HostedRoomService) -> list[dict]:
    return service._events("relay-room")


def _user_events(service: HostedRoomService) -> list[dict]:
    return [event for event in _events(service) if event["kind"] == "message.user"]


def _submit(facade: HostedRoomFacade, binding, update):
    return asyncio.run(facade.submit_telegram_update(binding, update))


def _wait_for(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition did not settle before timeout")


# -- version / capability discovery and fail-closed disablement ------------------------------


def test_describe_reports_facade_version_and_enabled_when_granted_and_compatible(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    status = HostedRoomFacade(PLUGIN_ID).describe()
    assert status.facade_version == FACADE_VERSION
    assert status.enabled is True
    assert status.reason == ""
    assert status.room_protocol == hosted_rooms.PROTOCOL_VERSION


def test_ungranted_plugin_is_disabled_and_appends_nothing(running_service):
    _grant()  # plugin entry exists, capability not granted
    facade = HostedRoomFacade(PLUGIN_ID)
    assert facade.describe().reason == "capability_not_granted"

    result = _submit(facade, _binding(), _update())

    assert (result.status, result.reason, result.ok) == ("disabled", "capability_not_granted", False)
    assert _user_events(running_service) == []


def test_no_running_hosted_room_worker_disables_the_facade(monkeypatch):
    _grant(facade_mod.CAPABILITY_ID)
    monkeypatch.setattr(methods_groups, "_service", None)
    facade = HostedRoomFacade(PLUGIN_ID)

    assert facade.describe().enabled is False
    result = _submit(facade, _binding(), _update())
    assert (result.status, result.reason) == ("disabled", "hosted_rooms_unavailable")


@pytest.mark.parametrize("protocol", [99, None, "2", True], ids=["future", "missing", "str", "bool"])
def test_unknown_or_incompatible_room_protocol_fails_closed(running_service, monkeypatch, protocol):
    _grant(facade_mod.CAPABILITY_ID)
    if protocol is None:
        monkeypatch.delattr(hosted_rooms, "PROTOCOL_VERSION")
    else:
        monkeypatch.setattr(hosted_rooms, "PROTOCOL_VERSION", protocol)
    facade = HostedRoomFacade(PLUGIN_ID)

    status = facade.describe()
    assert (status.enabled, status.reason, status.room_protocol) == (False, "incompatible_gateway", None)
    result = _submit(facade, _binding(), _update())
    assert (result.status, result.reason) == ("disabled", "incompatible_gateway")
    assert facade.outcomes(_binding()).reason == "incompatible_gateway"
    assert _user_events(running_service) == []


def test_capability_is_read_from_the_plugins_own_profile_home(running_service, tmp_path):
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    _grant(facade_mod.CAPABILITY_ID)  # granted only in the ambient (launch) home
    facade = HostedRoomFacade(PLUGIN_ID, home=other_home)
    assert facade.describe().reason == "capability_not_granted"


# -- ingress ---------------------------------------------------------------------------------


def test_valid_owner_update_is_admitted_with_telegram_attribution_and_settles(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)

    result = _submit(facade, _binding(), _update())

    assert (result.status, result.ok) == ("accepted", True)
    assert result.event_id == f"tg:{CHAT}:77"
    assert result.event.sender_id == OWNER
    assert (result.event.chat_id, result.event.thread_id, result.event.message_id) == (CHAT, TOPIC, 77)
    [event] = _user_events(running_service)
    assert event["actor"] == {"kind": "user", "id": f"telegram:{OWNER}"}
    assert event["payload"] == {"text": "@tester summarize the plan", "thread_id": f"tg:{CHAT}:{TOPIC}"}

    page = _wait_for(lambda: facade.outcomes(_binding()).outcomes and facade.outcomes(_binding()))
    [outcome] = page.outcomes
    assert outcome.status == "settled"
    assert (outcome.member_id, outcome.profile) == ("agent-tester", "agent-tester")
    assert outcome.text == "reply from agent-tester"
    assert outcome.source_event_id == result.event_id
    assert page.cursor >= outcome.seq


def test_redelivered_update_is_a_transactional_noop(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)

    first = _submit(facade, _binding(), _update())
    again = _submit(facade, _binding(), _update(update_id=9002))

    assert (first.status, again.status, again.ok) == ("accepted", "duplicate", True)
    assert again.event_id == first.event_id
    assert len(_user_events(running_service)) == 1


def test_reused_message_id_with_different_text_is_rejected_not_overwritten(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)
    _submit(facade, _binding(), _update())

    result = _submit(facade, _binding(), _update(text="@tester something else"))

    assert (result.status, result.reason) == ("rejected", "event_conflict")
    [event] = _user_events(running_service)
    assert event["payload"]["text"] == "@tester summarize the plan"


def test_main_chat_binding_admits_only_non_topic_messages(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)
    binding = _binding(thread_id=None)

    topic = _submit(facade, binding, _update())
    main = _submit(facade, binding, _update(message_id=78, thread_id=None, is_topic=False))

    assert (topic.status, topic.reason) == ("rejected", "topic_mismatch")
    assert main.status == "accepted"
    assert _user_events(running_service)[0]["payload"]["thread_id"] == f"tg:{CHAT}:main"


def test_outcomes_are_topic_local(running_service):
    """Replies to another topic routed into the same room never surface for this binding."""
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)
    other = _binding(thread_id=7)
    assert _submit(facade, other, _update(thread_id=7)).status == "accepted"
    _wait_for(lambda: facade.outcomes(other).outcomes)

    assert facade.outcomes(_binding()).outcomes == ()


def test_unknown_room_is_rejected(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    result = _submit(HostedRoomFacade(PLUGIN_ID), _binding(room_id="no-such-room"), _update())
    assert (result.status, result.reason) == ("rejected", "room_not_found")


# -- spoofed coordinates / untrusted origins -------------------------------------------------


@pytest.mark.parametrize(
    ("update_kwargs", "reason"),
    [
        ({"chat_id": -1009999999999}, "chat_mismatch"),
        ({"thread_id": 7}, "topic_mismatch"),
        ({"is_topic": False}, "topic_mismatch"),
        ({"sender_id": 555}, "sender_not_allowed"),
        ({"chat_type": "private", "chat_id": CHAT}, "not_group_chat"),
        ({"edited": True}, "edit_not_admitted"),
        ({"forward_origin": object()}, "forwarded"),
        ({"forward_date": 1700000000}, "forwarded"),
        ({"sender_chat": SimpleNamespace(id=CHAT)}, "sender_chat"),
        ({"via_bot": SimpleNamespace(id=1)}, "via_bot"),
        ({"text": None}, "unsupported_message"),
        ({"text": "   "}, "unsupported_message"),
        ({"sender_id": True}, "invalid_coordinates"),
        ({"chat_id": str(CHAT)}, "invalid_coordinates"),
        ({"message_id": 0}, "invalid_coordinates"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_spoofed_or_untrusted_coordinates_are_rejected_before_any_write(
    running_service, update_kwargs, reason,
):
    _grant(facade_mod.CAPABILITY_ID)

    result = _submit(HostedRoomFacade(PLUGIN_ID), _binding(), _update(**update_kwargs))

    assert (result.status, result.reason, result.ok) == ("rejected", reason, False)
    assert result.event_id is None
    assert _user_events(running_service) == []


@pytest.mark.parametrize("update", [None, object(), SimpleNamespace(update_id=1)],
                         ids=["none", "object", "no-message"])
def test_malformed_update_objects_never_raise(running_service, update):
    _grant(facade_mod.CAPABILITY_ID)
    result = _submit(HostedRoomFacade(PLUGIN_ID), _binding(), update)
    assert result.status == "rejected"
    assert _user_events(running_service) == []


@pytest.mark.parametrize("binding", [None, object(), SimpleNamespace(room_id="relay-room")],
                         ids=["none", "object", "lookalike"])
def test_malformed_bindings_never_raise_or_write(running_service, binding):
    _grant(facade_mod.CAPABILITY_ID)

    result = _submit(HostedRoomFacade(PLUGIN_ID), binding, _update())

    assert (result.status, result.reason) == ("rejected", "invalid_binding")
    assert _user_events(running_service) == []


def test_binding_with_no_authorized_senders_is_refused():
    with pytest.raises(ValueError, match="sender"):
        _binding(sender_ids=frozenset())
    with pytest.raises(ValueError, match="sender"):
        _binding(sender_ids=frozenset({True}))


# -- publication refusal ---------------------------------------------------------------------


def test_bot_authored_publication_echo_is_refused(running_service):
    """The coordinator's own labeled post (or any bot) never re-enters the room as human input."""
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)

    result = _submit(facade, _binding(sender_ids=frozenset({OWNER, 4242})),
                     _update(sender_id=4242, is_bot=True, text="[agent-tester] reply"))

    assert (result.status, result.reason) == ("rejected", "bot_sender")
    assert _user_events(running_service) == []


def test_facade_public_verbs_are_exactly_the_reviewed_set():
    """New surface (another post/send path) needs its own review; publication is the one gated verb."""
    public = {name for name in dir(HostedRoomFacade) if not name.startswith("_")}
    assert public == {"describe", "submit_telegram_update", "outcomes", "publish_outcome"}


def test_unsettled_member_reply_is_not_surfaced_as_an_outcome(running_service, monkeypatch):
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)
    _submit(facade, _binding(), _update())
    _wait_for(lambda: facade.outcomes(_binding()).outcomes)
    settled = next(e for e in _events(running_service) if e["kind"] == "turn.settled")

    # A page boundary that ends on the member message must not advance past it.
    original = hosted_rooms.read_events

    def truncated(db_path, **kwargs):
        page = original(db_path, **kwargs)
        rows = [row for row in page["events"] if row["seq"] < settled["seq"]]
        return {**page, "events": rows, "cursor": settled["seq"] - 1, "has_more": True}

    monkeypatch.setattr(hosted_rooms, "read_events", truncated)
    page = facade.outcomes(_binding())
    assert page.outcomes == ()
    assert page.cursor < settled["seq"] - 1


def test_caller_cancellation_mid_submit_leaves_at_most_one_event(idle_service, monkeypatch):
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)
    original_send = idle_service.send
    entered = asyncio.Event()
    release = threading.Event()

    def slow_send(**kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=5), "test did not release the blocked send"
        return original_send(**kwargs)

    monkeypatch.setattr(idle_service, "send", slow_send)

    async def scenario():
        task = asyncio.create_task(facade.submit_telegram_update(_binding(), _update()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()
    _wait_for(lambda: _user_events(idle_service))
    monkeypatch.setattr(idle_service, "send", original_send)

    retry = _submit(facade, _binding(), _update())

    assert retry.status == "duplicate"
    assert len(_user_events(idle_service)) == 1


# -- baseline: no relay consumer -------------------------------------------------------------


def test_plugin_context_facade_is_lazy_and_touches_no_room_state(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    calls = []
    monkeypatch.setattr(methods_groups, "get_hosted_room_service", lambda: calls.append("svc"))
    monkeypatch.setattr(hosted_rooms, "append_event", lambda *a, **k: calls.append("append"))

    ctx = PluginContext(PluginManifest(name=PLUGIN_ID, source="user"), PluginManager())
    facade = ctx.hosted_rooms

    assert isinstance(facade, HostedRoomFacade)
    assert ctx.hosted_rooms is facade
    assert calls == []


def test_disabled_facade_returns_results_instead_of_raising_into_telegram_dispatch(monkeypatch):
    """A relay handler sits in PTB group -1; a raising facade would break every later handler."""
    monkeypatch.setattr(methods_groups, "_service", None)
    facade = HostedRoomFacade(PLUGIN_ID)

    assert _submit(facade, _binding(), _update()).status == "disabled"
    assert facade.outcomes(_binding()).outcomes == ()

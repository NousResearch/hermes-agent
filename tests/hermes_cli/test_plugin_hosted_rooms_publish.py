"""Behaviour contract for ``ctx.hosted_rooms.publish_outcome`` — the coordinator-only verb that posts
a settled Group Chat reply back to the binding's Telegram chat/topic as the gateway bot.

Every test drives the real ``HostedRoomService`` over a temp SQLite store; only the per-profile
agent RPC, the gateway runner and the Telegram adapter are faked.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

# Imported at collection: its import-time bootstrap probes the checkout's git dir, which the per-test
# home-I/O guard refuses when the checkout itself lives under the Hermes home.
import gateway.run
from gateway import hosted_rooms
from gateway.config import Platform
from gateway.platforms.base import SendResult
from hermes_cli import plugin_hosted_rooms as facade_mod
from hermes_cli.plugin_hosted_rooms import FACADE_VERSION, HostedRoomFacade
from tests.hermes_cli.test_plugin_hosted_rooms import (
    CHAT,
    OWNER,
    PLUGIN_ID,
    TOPIC,
    _binding,
    _events,
    _grant,
    _make_service,
    _submit,
    _update,
    _wait_for,
)
from tui_gateway import methods_groups

BOTH = (facade_mod.CAPABILITY_ID, facade_mod.PUBLISH_CAPABILITY_ID)


class _Adapter:
    def __init__(self, *, result=None, raises=False, connected=True) -> None:
        self.result = result or SendResult(success=True, message_id="555")
        self.raises = raises
        self.is_connected = connected
        self.calls: list[dict] = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        if self.raises:
            raise RuntimeError("telegram down")
        return self.result


def _install(monkeypatch, runner) -> None:
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)


def _runner(adapter):
    return SimpleNamespace(
        _gateway_loop=None,
        _authorization_adapter=lambda platform, profile: adapter if platform is Platform.TELEGRAM else None)


@pytest.fixture
def running_service(tmp_path, monkeypatch):
    """A started process-owned service, published where the gateway publishes it."""
    service = _make_service(tmp_path)
    monkeypatch.setattr(methods_groups, "_service", service)
    service.start()
    yield service
    service.stop(timeout=5.0)


@pytest.fixture(autouse=True)
def _fresh_ledger(monkeypatch):
    monkeypatch.setattr(facade_mod, "_ledger", OrderedDict())


@pytest.fixture
def adapter(monkeypatch):
    adapter = _Adapter()
    _install(monkeypatch, _runner(adapter))
    return adapter


def _publish(facade, binding, outcome, *, version=FACADE_VERSION, label=None):
    return asyncio.run(facade.publish_outcome(binding, outcome, facade_version=version, label=label))


def _settled(facade, binding=None, update=None):
    binding = binding or _binding()
    assert _submit(facade, binding, update or _update()).status == "accepted"
    page = _wait_for(lambda: facade.outcomes(binding).outcomes and facade.outcomes(binding))
    [outcome] = page.outcomes
    assert outcome.status == "settled"
    return outcome


def _ready(*grants):
    _grant(*BOTH)
    facade = HostedRoomFacade(PLUGIN_ID)
    outcome = _settled(facade)
    if grants != BOTH:
        _grant(*grants)
    return facade, outcome


# -- the happy path ---------------------------------------------------------------------------


def test_settled_reply_is_published_once_to_the_bound_topic_and_writes_nothing(running_service, adapter):
    facade, outcome = _ready(*BOTH)
    before = _events(running_service)

    result = _publish(facade, _binding(), outcome)

    assert (result.status, result.reason, result.ok, result.message_id) == ("published", "", True, "555")
    assert adapter.calls == [{
        "chat_id": str(CHAT), "content": "[agent-tester] reply from agent-tester",
        "metadata": {"thread_id": str(TOPIC), "telegram_dm_topic_created_for_send": True}}]
    assert _events(running_service) == before


def test_verified_reply_accepts_a_bounded_display_label_without_changing_attribution(running_service, adapter):
    facade, outcome = _ready(*BOTH)

    result = _publish(facade, _binding(), outcome, label="🧪 Agent Tester")

    assert result.status == "published"
    assert adapter.calls[0]["content"] == "[🧪 Agent Tester] reply from agent-tester"


@pytest.mark.parametrize("label", ["", " agent-tester", "agent-tester ", "bad[label]", "bad\nlabel", "x" * 65])
def test_invalid_display_label_is_rejected_before_send(running_service, adapter, label):
    facade, outcome = _ready(*BOTH)

    result = _publish(facade, _binding(), outcome, label=label)

    assert (result.status, result.reason) == ("rejected", "invalid_label")
    assert adapter.calls == []


def test_main_chat_binding_sends_without_topic_metadata(running_service, adapter):
    _grant(*BOTH)
    facade = HostedRoomFacade(PLUGIN_ID)
    binding = _binding(thread_id=None)
    outcome = _settled(facade, binding, _update(thread_id=None, is_topic=False))

    assert _publish(facade, binding, outcome).status == "published"
    [call] = adapter.calls
    assert call["metadata"] is None


def test_republishing_is_a_duplicate_with_the_first_message_id(running_service, adapter):
    facade, outcome = _ready(*BOTH)
    _publish(facade, _binding(), outcome)

    again = _publish(facade, _binding(), outcome)

    assert (again.status, again.message_id, again.ok) == ("duplicate", "555", True)
    assert len(adapter.calls) == 1


# -- gates ------------------------------------------------------------------------------------


def test_describe_reports_publication_only_with_both_grants(running_service):
    _grant(facade_mod.CAPABILITY_ID)
    status = HostedRoomFacade(PLUGIN_ID).describe()
    assert status.publication is False and status.publication_labels is True
    _grant(facade_mod.PUBLISH_CAPABILITY_ID)
    assert HostedRoomFacade(PLUGIN_ID).describe().publication is False
    _grant(*BOTH)
    assert HostedRoomFacade(PLUGIN_ID).describe().publication is True


@pytest.mark.parametrize("grants", [(facade_mod.CAPABILITY_ID,), (facade_mod.PUBLISH_CAPABILITY_ID,)],
                         ids=["publish-missing", "ingress-missing"])
def test_missing_capability_disables_publication(running_service, adapter, grants):
    facade, outcome = _ready(*grants)
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason) == ("disabled", "capability_not_granted")
    assert adapter.calls == []


def test_no_running_service_disables_publication(running_service, adapter, monkeypatch):
    facade, outcome = _ready(*BOTH)
    monkeypatch.setattr(methods_groups, "_service", None)
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason) == ("disabled", "hosted_rooms_unavailable")
    assert adapter.calls == []


@pytest.mark.parametrize("version", [2, "1", True], ids=["future", "str", "bool"])
def test_incompatible_facade_version_is_disabled(running_service, adapter, version):
    facade, outcome = _ready(*BOTH)
    result = _publish(facade, _binding(), outcome, version=version)
    assert (result.status, result.reason) == ("disabled", "incompatible_facade")
    assert adapter.calls == []


def test_unsupported_room_protocol_is_disabled(running_service, adapter, monkeypatch):
    facade, outcome = _ready(*BOTH)
    monkeypatch.setattr(hosted_rooms, "PROTOCOL_VERSION", 99)
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason) == ("disabled", "incompatible_gateway")
    assert adapter.calls == []


@pytest.mark.parametrize("make_runner", [
    lambda a: None,
    lambda a: _runner(None),
    lambda a: _runner(_Adapter(connected=False)),
    lambda a: SimpleNamespace(_gateway_loop=None, adapters={Platform.TELEGRAM: a}),
], ids=["no-runner", "no-adapter", "disconnected", "no-profile-aware-lookup"])
def test_unavailable_adapter_fails_closed(running_service, monkeypatch, make_runner):
    facade, outcome = _ready(*BOTH)
    fallback = _Adapter()
    _install(monkeypatch, make_runner(fallback))
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason) == ("disabled", "adapter_unavailable")
    assert fallback.calls == []


# -- target and claim verification ------------------------------------------------------------


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"thread_id": 7}, "topic_mismatch"),
    ({"chat_id": -1009999999999}, None),
    ({"sender_ids": frozenset({555})}, "source_not_admitted"),
], ids=["other-topic", "other-chat", "sender-not-allowlisted"])
def test_outcome_cannot_be_published_to_another_target(running_service, adapter, overrides, reason):
    facade, outcome = _ready(*BOTH)
    result = _publish(facade, _binding(**overrides), outcome)
    assert result.status == "rejected"
    if reason is not None:
        assert result.reason == reason
    assert adapter.calls == []


@pytest.mark.parametrize(("change", "reasons"), [
    ({"text": "forged"}, {"outcome_mismatch"}),
    ({"profile": "default"}, {"roster_mismatch", "outcome_mismatch"}),
    ({"source_event_id": f"tg:{CHAT}:1"}, {"outcome_mismatch"}),
    ({"seq": -1}, None),
    ({"status": "cancelled", "text": None}, {"not_settled"}),
    ({"status": "passed", "text": None}, {"not_settled"}),
], ids=["text", "profile", "source", "seq", "cancelled", "passed"])
def test_forged_outcome_claims_are_rejected(running_service, adapter, change, reasons):
    facade, outcome = _ready(*BOTH)
    if "seq" in change:
        change = {"seq": outcome.seq - 1}
    result = _publish(facade, _binding(), dataclasses.replace(outcome, **change))
    assert result.status == "rejected"
    if reasons is not None:
        assert result.reason in reasons
    assert adapter.calls == []


def test_desktop_started_round_is_never_published(running_service, adapter):
    _grant(*BOTH)
    facade = HostedRoomFacade(PLUGIN_ID)
    running_service.send(room_id="relay-room", event_id=f"tg:{CHAT}:999",
                         payload={"text": "@tester go", "thread_id": f"tg:{CHAT}:{TOPIC}"})
    [outcome] = _wait_for(lambda: facade.outcomes(_binding()).outcomes)

    result = _publish(facade, _binding(), outcome)

    assert (result.status, result.reason) == ("rejected", "source_not_admitted")
    assert adapter.calls == []


def test_only_the_room_coordinator_publishes(running_service, adapter, monkeypatch):
    facade, outcome = _ready(*BOTH)
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "install:someone-else")
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason) == ("rejected", "not_room_authority")
    assert adapter.calls == []


def test_reply_over_the_telegram_bound_is_rejected(running_service, adapter, monkeypatch):
    rpc = running_service.rpc

    def long_submit(*, profile, prompt, on_terminal, **_kwargs):
        on_terminal({"status": "settled", "text": "x" * facade_mod.MAX_PUBLICATION_CHARS})
        return {"accepted": True}

    monkeypatch.setattr(rpc, "submit", long_submit)
    facade, outcome = _ready(*BOTH)

    result = _publish(facade, _binding(), outcome)

    assert (result.status, result.reason) == ("rejected", "text_too_long")
    assert adapter.calls == []


@pytest.mark.parametrize(("binding", "outcome"), [
    (None, "real"), ("real", None), ("real", object()), (object(), "real"),
], ids=["no-binding", "no-outcome", "object-outcome", "object-binding"])
def test_malformed_arguments_never_raise(running_service, adapter, binding, outcome):
    facade, real = _ready(*BOTH)
    binding = _binding() if binding == "real" else binding
    outcome = real if outcome == "real" else outcome
    result = _publish(facade, binding, outcome)
    assert (result.status, result.reason) == ("rejected", "invalid_outcome")
    assert adapter.calls == []


# -- send failures ----------------------------------------------------------------------------


def test_raising_adapter_fails_retryably_and_a_retry_publishes(running_service, monkeypatch):
    facade, outcome = _ready(*BOTH)
    broken = _Adapter(raises=True)
    _install(monkeypatch, _runner(broken))

    failed = _publish(facade, _binding(), outcome)

    assert (failed.status, failed.reason, failed.retryable, failed.ok) == ("failed", "adapter_error", True, False)
    healthy = _Adapter()
    _install(monkeypatch, _runner(healthy))
    assert _publish(facade, _binding(), outcome).status == "published"
    assert len(healthy.calls) == 1


def test_unsuccessful_send_result_is_failed_with_its_retryability(running_service, monkeypatch):
    facade, outcome = _ready(*BOTH)
    _install(monkeypatch, _runner(_Adapter(result=SendResult(success=False, error="flood", retryable=True))))
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason, result.retryable) == ("failed", "send_failed", True)


class _BlockingAdapter(_Adapter):
    """Holds every send open until released, recording the loop it ran on."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = asyncio.Event()  # binds to the loop of its first wait (3.10+)
        self.loops: list = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.loops.append(asyncio.get_running_loop())
        self.entered.set()
        await self.release.wait()
        return await super().send(chat_id, content, reply_to, metadata)


def test_send_runs_on_the_gateway_loop_and_a_concurrent_publish_is_in_flight(running_service, monkeypatch):
    """The adapter is bound to the gateway loop: a caller on another loop hops onto it, and a second
    publish of the same outcome while the first is in flight never sends twice."""
    facade, outcome = _ready(*BOTH)
    gateway_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=gateway_loop.run_forever, daemon=True)
    thread.start()
    blocking = _BlockingAdapter()
    runner = _runner(blocking)
    runner._gateway_loop = gateway_loop
    _install(monkeypatch, runner)

    async def scenario():
        first = asyncio.create_task(facade.publish_outcome(_binding(), outcome, facade_version=FACADE_VERSION))
        assert await asyncio.to_thread(blocking.entered.wait, 5)
        second = await facade.publish_outcome(_binding(), outcome, facade_version=FACADE_VERSION)
        blocking.loops[0].call_soon_threadsafe(blocking.release.set)  # on the loop that owns the event
        return await asyncio.wait_for(first, 10), second

    try:
        first, second = asyncio.run(scenario())
    finally:
        gateway_loop.call_soon_threadsafe(gateway_loop.stop)
        thread.join(5)
        gateway_loop.close()

    assert blocking.loops == [gateway_loop]
    assert (first.status, first.message_id) == ("published", "555")
    assert (second.status, second.reason) == ("duplicate", "in_flight")
    assert len(blocking.calls) == 1


def test_caller_cancelled_mid_send_never_reposts(running_service, monkeypatch):
    """A cancelled send may already have reached Telegram, so the claim is kept, not released."""
    facade, outcome = _ready(*BOTH)
    blocking = _BlockingAdapter()
    _install(monkeypatch, _runner(blocking))

    async def scenario():
        task = asyncio.create_task(facade.publish_outcome(_binding(), outcome, facade_version=FACADE_VERSION))
        assert await asyncio.to_thread(blocking.entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    healthy = _Adapter()
    _install(monkeypatch, _runner(healthy))

    retry = _publish(facade, _binding(), outcome)

    assert (retry.status, retry.reason) == ("duplicate", "delivery_uncertain")
    assert healthy.calls == []


def test_uncertain_delivery_is_never_evicted_within_the_process(monkeypatch):
    monkeypatch.setattr(facade_mod, "_LEDGER_MAX", 1)
    uncertain = ("relay-room", 1, "agent-tester")
    assert facade_mod._ledger_claim(uncertain) is None
    facade_mod._ledger_settle(uncertain, None)

    for seq in range(2, 6):
        key = ("relay-room", seq, "agent-tester")
        assert facade_mod._ledger_claim(key) is None
        facade_mod._ledger_settle(key, facade_mod.PublicationResult("published", message_id=str(seq)))

    duplicate = facade_mod._ledger_claim(uncertain)
    assert duplicate is not None
    assert (duplicate.status, duplicate.reason) == ("duplicate", "delivery_uncertain")


def test_unexpected_internal_error_is_a_result_not_a_raise(running_service, adapter, monkeypatch):
    facade, outcome = _ready(*BOTH)

    def boom(*_args):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(facade_mod, "_verify_publication", boom)
    result = _publish(facade, _binding(), outcome)
    assert (result.status, result.reason) == ("failed", "internal_error")
    assert adapter.calls == []


# -- baseline: ordinary relay without the publish grant ---------------------------------------


def test_ingress_only_relay_is_unchanged_and_never_sends(running_service, monkeypatch):
    broken = _Adapter(raises=True)
    _install(monkeypatch, _runner(broken))
    _grant(facade_mod.CAPABILITY_ID)
    facade = HostedRoomFacade(PLUGIN_ID)

    status = facade.describe()
    assert (status.enabled, status.reason, status.publication) == (True, "", False)
    outcome = _settled(facade)
    assert outcome.text == "reply from agent-tester"
    assert [e["actor"]["id"] for e in _events(running_service) if e["kind"] == "message.user"] == [
        f"telegram:{OWNER}"]

    result = _publish(facade, _binding(), outcome)

    assert (result.status, result.reason) == ("disabled", "capability_not_granted")
    assert broken.calls == []

"""Signal owner detection.

The owner is the sender listed by explicit id in ``SIGNAL_ALLOWED_USERS`` (``*`` is open access, never
an owner), also when a group shows them only by UUID. The durable inbound envelope rebuilds the
source without the verdict, so the turn settles it before anything renders it: an ``**Owner:**`` line
in a single-user session, an owner-only ``[SYSTEM: sender NAME is the owner]`` marker in a shared one.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionContext, SessionSource, build_session_context_prompt, label_shared_sender
from gateway.session_envelope import restore_native_source

OWNER, GUEST = "+15550001111", "+15550002222"
OWNER_UUID = "0b8e2c1a-1111-4222-8333-444455556666"


@pytest.fixture(autouse=True)
def owner_allowlist(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", OWNER)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")


def _adapter():
    from plugins.platforms.signal.adapter import SignalAdapter
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={
        "http_url": "http://localhost:8080", "account": "+15559999999"}))
    adapter.set_authorization_check(lambda *args, **kwargs: True)  # the runner wires one at startup
    return adapter


def _runner(adapter=None):
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.SIGNAL: adapter} if adapter else {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _source(user_id, *, chat_type="dm", platform=Platform.SIGNAL, **fields):
    chat_id = user_id if chat_type == "dm" else "group:g1"
    return SessionSource(platform=platform, chat_id=chat_id, chat_type=chat_type, user_id=user_id,
                         user_name="Alice", **fields)


def _through_envelope(source):
    """The source a turn runs on: encoded into the durable inbound envelope and restored."""
    return restore_native_source({"text": "hi", "native_text_v1": {"source": {**source.to_dict(), "is_bot": False}}})


async def _intake(adapter, **envelope):
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    await adapter._handle_envelope({"envelope": {
        "timestamp": 1700000000000, "sourceName": "Alice",
        "dataMessage": {"message": "hi", "groupInfo": {"groupId": "g1"}}, **envelope}})
    return events[0].source


@pytest.mark.asyncio
async def test_the_turn_settles_owner_status_before_building_the_context_prompt(monkeypatch):
    """The envelope drops the intake verdict; ``_hmwa_prepare_turn`` must restore it first."""
    import gateway.run_turn_prepare as turn_prepare
    from tools import approval_yolo

    runner = _runner()
    runner._hmwa_open_session = AsyncMock(return_value=(False, False))
    monkeypatch.setattr(approval_yolo, "restore_gateway_yolo", lambda *args, **kwargs: None)
    seen = []

    class _ContextBuilt(Exception):
        pass

    def capture(source, *args, **kwargs):
        seen.append(source.is_owner)
        raise _ContextBuilt

    monkeypatch.setattr(turn_prepare, "build_session_context", capture)
    restored = _through_envelope(_source(OWNER, is_owner=True))
    assert restored.is_owner is False
    with pytest.raises(_ContextBuilt):
        await runner._hmwa_prepare_turn(MessageEvent(text="hi", source=restored), restored,
                                        SimpleNamespace(yolo=None), "key", "key", 1)
    assert seen == [True]


@pytest.mark.asyncio
async def test_a_group_sender_seen_only_by_uuid_is_the_owner():
    adapter = _adapter()
    await _intake(adapter, sourceNumber=OWNER, sourceUuid=OWNER_UUID)  # the adapter learns the pair
    restored = _through_envelope(await _intake(adapter, sourceUuid=OWNER_UUID))
    _runner(adapter)._resolve_owner_status(restored)
    assert restored.is_owner is True


@pytest.mark.asyncio
async def test_a_uuid_only_envelope_keeps_the_learned_phone():
    adapter = _adapter()
    await _intake(adapter, sourceNumber=OWNER, sourceUuid=OWNER_UUID)
    await _intake(adapter, sourceUuid=OWNER_UUID)
    assert adapter._recipient_number_by_uuid[OWNER_UUID] == OWNER


@pytest.mark.parametrize("allowlist, user_id, expected", [
    (OWNER, OWNER, True), (OWNER, GUEST, False), ("*", OWNER, False)])
def test_only_an_explicitly_listed_sender_is_the_owner(monkeypatch, allowlist, user_id, expected):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", allowlist)
    source = _source(user_id)
    _runner(_adapter())._resolve_owner_status(source)
    assert source.is_owner is expected


@pytest.mark.parametrize("is_owner, line", [(True, "**Owner:** yes"), (False, "**Owner:** no")])
def test_a_single_user_prompt_states_owner_status(is_owner, line):
    context = SessionContext(source=_source(OWNER, is_owner=is_owner), connected_platforms=[Platform.SIGNAL],
                             home_channels={})
    assert line in build_session_context_prompt(context)


def test_a_shared_group_marks_only_the_owner():
    def label(source):
        return label_shared_sender(source, "hi", group_sessions_per_user=False)

    assert label(_source(OWNER, chat_type="group", is_owner=True)) == "[SYSTEM: sender Alice is the owner] [Alice] hi"
    assert label(_source(GUEST, chat_type="group")) == "[Alice] hi"
    other_platform = _source("1", chat_type="group", platform=Platform.TELEGRAM, is_owner=True)
    assert label(other_platform) == "[Alice] hi"


def test_an_owner_flip_rekeys_only_the_single_user_prompt_pin():
    from gateway.run import GatewayRunner

    def key(is_owner, shared):
        source = _source(OWNER, chat_type="group" if shared else "dm", is_owner=is_owner)
        context = SessionContext(source=source, connected_platforms=[Platform.SIGNAL], home_channels={},
                                 shared_multi_user_session=shared)
        return GatewayRunner._ephemeral_change_key(context, False)

    assert key(True, False) != key(False, False)
    assert key(True, True) == key(False, True)

"""WhatsApp owner detection.

The owner is the sender listed by explicit id in ``WHATSAPP_ALLOWED_USERS`` (``*`` is open access, never
an owner), matched across phone and LID through the bridge's lid-mapping files, so a group that shows
the owner only by LID still recognizes them. The durable inbound envelope rebuilds the source without
the verdict, so the turn settles it before anything renders it: an ``**Owner:**`` line in a
single-user session, an owner-only ``[SYSTEM: sender NAME is the owner]`` marker in a shared one.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionContext, SessionSource, build_session_context_prompt, label_shared_sender
from gateway.session_envelope import restore_native_source
from hermes_constants import get_hermes_home

OWNER_PHONE, OWNER_LID = "15550001111", "900000000000001"
OWNER_JID = f"{OWNER_PHONE}@s.whatsapp.net"
GROUP = "120363001234567890@g.us"


@pytest.fixture(autouse=True)
def owner_allowlist(monkeypatch):
    """The owner's phone is allowlisted; the bridge has mapped it to the owner's LID."""
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OWNER_PHONE)
    session_dir = get_hermes_home() / "platforms" / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"lid-mapping-{OWNER_PHONE}.json").write_text(json.dumps(OWNER_LID), encoding="utf-8")
    (session_dir / f"lid-mapping-{OWNER_LID}_reverse.json").write_text(json.dumps(OWNER_PHONE), encoding="utf-8")


def _runner():
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _source(user_id, *, chat_type="dm", platform=Platform.WHATSAPP, **fields):
    chat_id = user_id if chat_type == "dm" else GROUP
    return SessionSource(platform=platform, chat_id=chat_id, chat_type=chat_type, user_id=user_id,
                         user_name="Alice", **fields)


def _through_envelope(source):
    """The source a turn runs on: encoded into the durable inbound envelope and restored."""
    return restore_native_source({"text": "hi", "native_text_v1": {"source": {**source.to_dict(), "is_bot": False}}})


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
    restored = _through_envelope(_source(OWNER_JID, is_owner=True))
    assert restored.is_owner is False
    with pytest.raises(_ContextBuilt):
        await runner._hmwa_prepare_turn(MessageEvent(text="hi", source=restored), restored,
                                        SimpleNamespace(yolo=None), "key", "key", 1)
    assert seen == [True]


@pytest.mark.parametrize("allowlist, user_id, expected", [
    (OWNER_PHONE, OWNER_JID, True),
    (OWNER_PHONE, f"{OWNER_LID}@lid", True),
    (f"{OWNER_LID}@lid", OWNER_JID, True),
    (OWNER_PHONE, "15550002222@s.whatsapp.net", False),
    (OWNER_PHONE, "900000000000999@lid", False),
    ("*", OWNER_JID, False),
])
def test_only_an_explicitly_listed_sender_is_the_owner(monkeypatch, allowlist, user_id, expected):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", allowlist)
    source = _through_envelope(_source(user_id, chat_type="group"))
    _runner()._resolve_owner_status(source)
    assert source.is_owner is expected


@pytest.mark.parametrize("is_owner, line", [(True, "**Owner:** yes"), (False, "**Owner:** no")])
def test_a_single_user_prompt_states_owner_status(is_owner, line):
    context = SessionContext(source=_source(OWNER_JID, is_owner=is_owner), connected_platforms=[Platform.WHATSAPP],
                             home_channels={})
    assert line in build_session_context_prompt(context)


def test_a_shared_group_marks_only_the_owner():
    def label(source):
        return label_shared_sender(source, "hi", group_sessions_per_user=False)

    assert label(_source(OWNER_JID, chat_type="group", is_owner=True)) == "[SYSTEM: sender Alice is the owner] [Alice] hi"
    assert label(_source("15550002222@s.whatsapp.net", chat_type="group")) == "[Alice] hi"
    other_platform = _source("1", chat_type="group", platform=Platform.TELEGRAM, is_owner=True)
    assert label(other_platform) == "[Alice] hi"


def test_an_owner_flip_rekeys_only_the_single_user_prompt_pin():
    from gateway.run import GatewayRunner

    def key(is_owner, shared):
        source = _source(OWNER_JID, chat_type="group" if shared else "dm", is_owner=is_owner)
        context = SessionContext(source=source, connected_platforms=[Platform.WHATSAPP], home_channels={},
                                 shared_multi_user_session=shared)
        return GatewayRunner._ephemeral_change_key(context, False)

    assert key(True, False) != key(False, False)
    assert key(True, True) == key(False, True)

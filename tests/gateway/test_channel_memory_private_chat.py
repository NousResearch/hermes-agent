"""Owner memory needs proof of a one-to-one chat: a "dm" label alone is not proof.

``chat_type`` is "dm" by default, the relay fills it in when the connector omits it, Slack
labels group DMs (mpim) "dm", and a home-channel handoff without a thread is built as "dm".
Each of those is a chat with several people in it, so the owner's MEMORY.md / USER.md must
stay out of it."""

import contextvars

import pytest

from gateway.channel_memory import channel_memory_binding
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.private_chat import is_confirmed_private_chat
from gateway.relay.ws_transport import _event_from_wire
from gateway.session import SessionSource
from gateway.session_envelope import restore_native
from hermes_state_runtime import list_session_admissions
from tools import memory_tool as mt
from tools.memory_tool_channel import CHANNEL_MEMORY


class _Runner:
    """Lets the owner DM the agent and nobody else."""

    def _under_authorization_profile(self, source, check):
        return check()

    def _principal_authorized(self, source, *, allow_adapter_delegation):
        return source.chat_type == "dm" and source.user_id == "owner"


def _relay_source_without_type():
    raw = {"type": "message", "text": "hi", "message_id": "m1",
           "source": {"platform": "slack", "chat_id": "G123", "user_id": "owner", "scope_id": "T1"}}
    return _event_from_wire(raw).source


MULTI_PERSON_DMS = {
    "default chat_type": lambda: SessionSource(platform=Platform.DISCORD, chat_id="c1", user_id="owner"),
    "slack group DM (mpim)": lambda: SessionSource(
        platform=Platform.SLACK, chat_id="G123", chat_type="dm", user_id="owner", scope_id="T1"),
    "relay without a chat type": _relay_source_without_type,
    "handoff into a guild home channel": lambda: SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="dm", user_id="system:handoff"),
}


@pytest.mark.parametrize("make_source", MULTI_PERSON_DMS.values(), ids=MULTI_PERSON_DMS.keys())
def test_unproven_dm_gets_channel_memory_and_no_owner_writes(make_source):
    binding = channel_memory_binding(_Runner(), make_source(), {})
    assert binding is not None
    # A "dm" session key holds every member of the chat, so even the owner's turn is shared.
    assert binding.owner_writes is False


PROVEN_ONE_TO_ONE = {
    "telegram private chat (chat id is the user id)": SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42"),
    "slack IM (D channel)": SessionSource(platform=Platform.SLACK, chat_id="D123", chat_type="dm", user_id="owner"),
    "adapter-confirmed (Discord DMChannel)": SessionSource(
        platform=Platform.DISCORD, chat_id="9", chat_type="dm", user_id="owner", private_chat=True),
}


@pytest.mark.parametrize("source", PROVEN_ONE_TO_ONE.values(), ids=PROVEN_ONE_TO_ONE.keys())
def test_proven_one_to_one_chat_keeps_owner_memory(source):
    assert channel_memory_binding(_Runner(), source, {}) is None


def test_adapter_confirmation_survives_build_source():
    class _Adapter:
        platform, gateway_runner = Platform.DISCORD, None

    def built(**extra):
        return BasePlatformAdapter.build_source(_Adapter(), chat_id="9", chat_type="dm", user_id="owner", **extra)

    assert is_confirmed_private_chat(built(private_chat=True))
    assert not is_confirmed_private_chat(built())


def test_owner_note_in_a_slack_group_dm_lands_in_the_chat_file_not_memory_md(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path / "memories")
    mpim = SessionSource(platform=Platform.SLACK, chat_id="G123", chat_type="dm", user_id="owner", scope_id="T1")

    def turn():
        CHANNEL_MEMORY.set(channel_memory_binding(_Runner(), mpim, {}))
        store = mt.MemoryStore()
        store.load_from_disk()
        return mt.memory_tool(action="add", target="memory", content="Team offsite is in May", store=store)

    result = contextvars.copy_context().run(turn)
    assert '"routed_to": "channel"' in result
    assert "offsite" in (tmp_path / "memories" / "channels" / "slack" / "T1_G123.md").read_text(encoding="utf-8-sig")
    memory_md = tmp_path / "memories" / "MEMORY.md"
    assert not memory_md.exists() or "offsite" not in memory_md.read_text(encoding="utf-8-sig")


@pytest.mark.asyncio
async def test_adapter_confirmation_survives_native_admission(tmp_path, monkeypatch):
    """The queued row is replayed through from_dict, which drops the wire-invisible flag."""
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    chat = "15551234567@s.whatsapp.net"
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15551234567")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_path": str(tmp_path / "wa")}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="pc")
    runner._wire_adapter_handlers(adapter)
    try:
        replays = []
        for message_id, proof in (("p1", True), ("p2", False)):
            source = adapter.build_source(chat_id=chat, user_id=chat, chat_type="dm", private_chat=proof)
            receipt = await authority.admit_native(MessageEvent(text="hi", source=source, message_id=message_id))
            rows = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)
            row = next(r for r in rows if r["request_id"] == message_id)
            replays.append(is_confirmed_private_chat(restore_native(row["payload"], runner).source))
        assert replays == [True, False]
    finally:
        authority.db.close()

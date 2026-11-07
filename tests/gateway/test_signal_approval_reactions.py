"""A 👍/👎 reaction on a Signal approval prompt resolves the approval.

Envelopes use signal-cli's real shape: a reaction is ``dataMessage.reaction`` (``JsonReaction``:
``emoji``, ``targetAuthor``, ``targetSentTimestamp``, ``isRemove``) with no message text; a
Note-to-Self reaction arrives as ``syncMessage.sentMessage.reaction``.
"""
import pytest

from gateway.config import PlatformConfig

ACCOUNT = "+15551234567"
USER = "+15550001111"
PROMPT_TS = 1712345678000


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": ACCOUNT, **extra}
    adapter = SignalAdapter(config)

    async def no_typing(chat_id):
        return None

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        return {"timestamp": PROMPT_TS}

    adapter._stop_typing_indicator = no_typing
    adapter._rpc = mock_rpc
    dispatched = []

    async def fake_handle(event):
        dispatched.append(event)

    adapter.handle_message = fake_handle
    return adapter, dispatched


def _reaction(emoji, target_ts=PROMPT_TS, is_remove=False):
    return {"emoji": emoji, "targetAuthor": ACCOUNT, "targetAuthorNumber": ACCOUNT,
            "targetSentTimestamp": target_ts, "isRemove": is_remove}


def _dm_reaction_envelope(emoji, **kw):
    return {"envelope": {"source": USER, "sourceNumber": USER, "sourceUuid": "uuid-user",
                         "timestamp": PROMPT_TS + 5000,
                         "dataMessage": {"timestamp": PROMPT_TS + 5000, "message": None,
                                         "reaction": _reaction(emoji, **kw)}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji,word", [("👍", "👍"), ("✅", "👍"), ("👎", "👎"), ("❌", "👎")])
async def test_dm_reaction_on_approval_prompt_dispatches_approval_word(monkeypatch, emoji, word):
    adapter, dispatched = _make_adapter(monkeypatch)
    await adapter.send(USER, "approve?", metadata={"is_approval_prompt": True})

    await adapter._handle_envelope(_dm_reaction_envelope(emoji))

    assert [e.text for e in dispatched] == [word]
    assert dispatched[0].source.chat_id == USER
    assert dispatched[0].source.user_id == USER


@pytest.mark.asyncio
async def test_reaction_consumed_once_and_ignored_on_other_messages(monkeypatch):
    adapter, dispatched = _make_adapter(monkeypatch)
    await adapter.send(USER, "approve?", metadata={"is_approval_prompt": True})

    await adapter._handle_envelope(_dm_reaction_envelope("👍", target_ts=PROMPT_TS + 1))
    await adapter._handle_envelope(_dm_reaction_envelope("👍", is_remove=True))
    await adapter._handle_envelope(_dm_reaction_envelope("❤️"))
    assert dispatched == []

    await adapter._handle_envelope(_dm_reaction_envelope("👍"))
    await adapter._handle_envelope(_dm_reaction_envelope("👍"))
    assert [e.text for e in dispatched] == ["👍"]


@pytest.mark.asyncio
async def test_reaction_on_non_approval_send_is_not_routed(monkeypatch):
    adapter, dispatched = _make_adapter(monkeypatch)
    await adapter.send(USER, "just chatting")

    await adapter._handle_envelope(_dm_reaction_envelope("👍"))

    assert dispatched == []


@pytest.mark.asyncio
async def test_group_reaction_bypasses_mention_gate(monkeypatch):
    adapter, dispatched = _make_adapter(monkeypatch, group_allowed="*", require_mention=True)
    await adapter.send("group:grp1", "approve?", metadata={"is_approval_prompt": True})
    envelope = _dm_reaction_envelope("👍")
    envelope["envelope"]["dataMessage"]["groupInfo"] = {"groupId": "grp1", "type": "DELIVER"}

    await adapter._handle_envelope(envelope)

    assert [e.text for e in dispatched] == ["👍"]
    assert dispatched[0].source.chat_id == "group:grp1"


@pytest.mark.asyncio
async def test_note_to_self_reaction_dispatches_approval_word(monkeypatch):
    adapter, dispatched = _make_adapter(monkeypatch)
    await adapter.send(ACCOUNT, "approve?", metadata={"is_approval_prompt": True})

    await adapter._handle_envelope({"envelope": {
        "source": ACCOUNT, "sourceNumber": ACCOUNT, "sourceUuid": "uuid-self",
        "timestamp": PROMPT_TS + 5000,
        "syncMessage": {"sentMessage": {"destination": ACCOUNT, "destinationNumber": ACCOUNT,
                                        "timestamp": PROMPT_TS + 5000, "message": None,
                                        "reaction": _reaction("👍")}}}})

    assert [e.text for e in dispatched] == ["👍"]
    assert dispatched[0].source.chat_id == ACCOUNT

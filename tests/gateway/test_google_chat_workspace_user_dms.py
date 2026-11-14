"""Google Chat: person-to-person DMs answered by the Workspace user, as that user (``answer_dms``).

Covers: the scopes and allowlist requirement, DM discovery (joined vs invitation, allowed people
only, DMs that go away), which DM messages are admitted, and that everything outbound in those
DMs goes through the delegated user while app spaces keep using the app.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_google_chat import _FakeHttpError, _gc_mod
from tests.gateway.test_google_chat_workspace_user import (
    ALICE, BOB, USER, _Call, _FakeEvents, _make, _ws_pubsub,
)

from plugins.platforms.google_chat import workspace_user_dms as ud
from plugins.platforms.google_chat.workspace_user import (
    MEMBERSHIPS_READONLY_SCOPE, MESSAGES_SCOPE, SPACES_READONLY_SCOPE, WorkspaceUserSettings,
)

ALICE_DM, BOB_DM = "spaces/DM_ALICE", "spaces/DM_BOB"
APP_SPACE = "spaces/APP"


@pytest.fixture(autouse=True)
def _http_error_is_fake(monkeypatch):
    monkeypatch.setattr(_gc_mod, "HttpError", _FakeHttpError)


class _UserChat:
    """The delegated user's Chat client: DMs with Alice (joined) and Bob (an invitation)."""

    def __init__(self, dms=None, invited=(BOB_DM,)):
        self.dms = dict(dms if dms is not None else {ALICE: ALICE_DM, BOB: BOB_DM})
        self.invited = set(invited)
        self.api = MagicMock()
        spaces = self.api.spaces.return_value
        spaces.findDirectMessage.side_effect = self._find
        spaces.get.side_effect = self._get
        spaces.members.return_value.get.side_effect = self._member
        self.messages = spaces.messages.return_value
        self.messages.create.return_value.execute.return_value = {"name": f"{ALICE_DM}/messages/OUT"}
        self.messages.patch.return_value.execute.return_value = {"name": f"{ALICE_DM}/messages/OUT"}
        self.media = self.api.media.return_value

    def _find(self, name):
        space = self.dms.get(name.split("/", 1)[1])
        return _Call({"name": space}) if space else _Call(exc=_FakeHttpError(404))

    def _get(self, name):
        return _Call(exc=_FakeHttpError(403)) if name in self.invited else _Call({"name": name})

    def _member(self, name):
        return _Call({"member": {"name": "users/" + name.rsplit("/", 1)[-1].split("@")[0], "type": "HUMAN"}})


async def _dm_adapter(monkeypatch, user_chat=None, **more):
    a = _make(monkeypatch, {"email": USER, "answer_dms": True, **more})
    user_chat = user_chat or _UserChat()
    a._workspace_user._clients["chat"] = user_chat.api
    await a._workspace_user.start(pubsub=True)
    await a._workspace_user.dms.stop()  # the first pass ran; keep the loop out of the way
    return a, user_chat


# ===========================================================================
# Settings
# ===========================================================================


class TestSettings:
    def test_answering_dms_asks_for_message_space_and_membership_scopes(self):
        settings = WorkspaceUserSettings.from_extra(
            {"workspace_user": {"email": USER, "answer_dms": True, "thinking_reaction": "🤔"}}, ALICE)
        assert settings.scopes() == [MESSAGES_SCOPE, SPACES_READONLY_SCOPE, MEMBERSHIPS_READONLY_SCOPE]

    def test_answering_dms_needs_an_allowlist(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert WorkspaceUserSettings.from_extra({"workspace_user": {"email": USER, "answer_dms": True}}, "") is None
        assert "GOOGLE_CHAT_ALLOWED_USERS" in caplog.text

    @pytest.mark.parametrize("value", [False, "no", None])
    def test_off_unless_set_true(self, value):
        assert WorkspaceUserSettings.from_extra(
            {"workspace_user": {"email": USER, "answer_dms": value}}, ALICE) is None


# ===========================================================================
# Discovery and subscriptions
# ===========================================================================


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_joined_dms_with_allowed_people_are_subscribed(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            a, _ = await _dm_adapter(monkeypatch)
        dms = a._workspace_user.dms
        assert dms.spaces[ALICE_DM] == {"email": ALICE, "peer": "users/alice", "joined": True}
        assert dms.spaces[BOB_DM]["joined"] is False and "still an invitation" in caplog.text
        assert dms.user_id == "users/assistant"
        events: _FakeEvents = a._workspace_user.events()
        assert [s["targetResource"] for s in events.subscriptions_.subs.values()] == [
            f"//chat.googleapis.com/{ALICE_DM}"]

    @pytest.mark.asyncio
    async def test_only_allowed_people_are_looked_up(self, monkeypatch):
        _, user_chat = await _dm_adapter(monkeypatch)
        looked_up = {c.kwargs["name"] for c in user_chat.api.spaces.return_value.findDirectMessage.call_args_list}
        assert looked_up == {f"users/{ALICE}", f"users/{BOB}"}

    @pytest.mark.asyncio
    async def test_a_dm_that_goes_away_loses_its_subscription(self, monkeypatch):
        a, user_chat = await _dm_adapter(monkeypatch)
        user_chat.dms.pop(ALICE)
        await asyncio.to_thread(a._workspace_user.dms.upkeep)
        assert not a._workspace_user.dms.owns(ALICE_DM)
        assert a._workspace_user.events().subscriptions_.subs == {}

    @pytest.mark.asyncio
    async def test_an_accepted_invitation_is_picked_up(self, monkeypatch):
        a, user_chat = await _dm_adapter(monkeypatch)
        user_chat.invited.clear()
        await asyncio.to_thread(a._workspace_user.dms.upkeep)
        assert a._workspace_user.dms.spaces[BOB_DM]["joined"] is True
        assert len(a._workspace_user.events().subscriptions_.subs) == 2


# ===========================================================================
# Inbound
# ===========================================================================


def _dm_message(sender="users/alice", space=ALICE_DM, **more):
    return {"name": f"{space}/messages/D1", "text": "are you there?", "sender": {"name": sender, "type": "HUMAN"},
            "space": {"name": space}, **more}


class TestInbound:
    @pytest.mark.asyncio
    async def test_the_peers_message_is_dispatched_as_a_dm(self, monkeypatch):
        a, _ = await _dm_adapter(monkeypatch)
        a._dispatch_message = AsyncMock()
        submitted = []
        a._submit_on_loop = lambda coro: (submitted.append(coro), coro.close())
        a._handle_pubsub_message(_ws_pubsub(_dm_message()))
        assert len(submitted) == 1
        msg, envelope = a._dispatch_message.call_args.args
        assert msg["sender"]["email"] == ALICE and envelope["space"]["spaceType"] == "DIRECT_MESSAGE"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message", [
        _dm_message(sender="users/assistant"),                                # the user's own reply
        _dm_message(sender="users/bob"),                                      # not this DM's peer
        {**_dm_message(), "sender": {"name": "users/x", "type": "BOT"}},
        _dm_message(space=BOB_DM, sender="users/bob"),                        # an unaccepted invitation
    ])
    async def test_other_messages_are_not_dispatched(self, monkeypatch, message):
        a, _ = await _dm_adapter(monkeypatch)
        submitted = []
        a._submit_on_loop = lambda coro: (submitted.append(coro), coro.close())
        a._handle_pubsub_message(_ws_pubsub(message))
        assert submitted == []


# ===========================================================================
# Outbound goes through the user in its DMs, through the app elsewhere
# ===========================================================================


class TestOutbound:
    @pytest.mark.asyncio
    async def test_replies_edits_and_deletes_in_a_dm_are_made_as_the_user(self, monkeypatch):
        a, user_chat = await _dm_adapter(monkeypatch)
        assert (await a.send(ALICE_DM, "hello")).success
        assert (await a.edit_message(ALICE_DM, f"{ALICE_DM}/messages/OUT", "hello again")).success
        assert await a.delete_message(ALICE_DM, f"{ALICE_DM}/messages/OUT")
        assert user_chat.messages.create.call_args.kwargs["parent"] == ALICE_DM
        user_chat.messages.patch.assert_called_once()
        user_chat.messages.delete.assert_called_once()
        a._chat_api.spaces.return_value.messages.return_value.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_app_spaces_still_use_the_app(self, monkeypatch):
        a, user_chat = await _dm_adapter(monkeypatch)
        app_messages = a._chat_api.spaces.return_value.messages.return_value
        app_messages.create.return_value.execute.return_value = {"name": f"{APP_SPACE}/messages/1"}
        assert (await a.send(APP_SPACE, "hello")).success
        assert app_messages.create.call_args.kwargs["parent"] == APP_SPACE
        user_chat.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_cards_and_clarify_buttons_become_text(self, monkeypatch):
        a, user_chat = await _dm_adapter(monkeypatch)
        card = {"cardId": "c", "card": {"header": {"title": "Pick one"},
                                        "sections": [{"widgets": [{"textParagraph": {"text": "<b>Lights</b>"}}]}]}}
        assert (await a.send_card(ALICE_DM, card)).success
        await a.send_clarify(ALICE_DM, "Which room?", ["Kitchen", "Office"], "cl1", "session")
        bodies = [c.kwargs["body"] for c in user_chat.messages.create.call_args_list]
        assert all("cardsV2" not in b for b in bodies)
        assert bodies[0]["text"] == "Pick one\nLights"
        assert "Kitchen" in bodies[1]["text"]

    @pytest.mark.asyncio
    async def test_a_refused_send_in_a_dm_is_not_fatal_for_the_adapter(self, monkeypatch):
        a, user_chat = await _dm_adapter(monkeypatch)
        user_chat.messages.create.return_value.execute.side_effect = _FakeHttpError(403)
        a._workspace_user.dms.wake = MagicMock()
        result = await a.send(ALICE_DM, "hello")
        assert not result.success and a.fatal_error_code is None
        a._workspace_user.dms.wake.assert_called_once()

    @pytest.mark.asyncio
    async def test_files_in_a_dm_are_uploaded_as_the_user(self, monkeypatch, tmp_path):
        a, user_chat = await _dm_adapter(monkeypatch)
        monkeypatch.setattr(_gc_mod, "MediaFileUpload", MagicMock())
        user_chat.media.upload.return_value.execute.return_value = {"attachmentDataRef": {"resourceName": "r"}}
        a._acquire_user_chat_api = AsyncMock(side_effect=AssertionError("no per-user OAuth in a user DM"))
        path = tmp_path / "report.txt"
        path.write_text("ok")
        assert (await a.send_document(ALICE_DM, str(path))).success
        assert user_chat.media.upload.call_args.kwargs["parent"] == ALICE_DM
        assert user_chat.messages.create.call_args.kwargs["body"]["attachment"] == [
            {"attachmentDataRef": {"resourceName": "r"}}]


def test_card_text_keeps_titles_and_text_without_markup():
    card = {"card": {"header": {"title": "T", "subtitle": "S"},
                     "sections": [{"widgets": [{"decoratedText": {"topLabel": "L", "text": "<i>x</i>"}},
                                               {"buttonList": {"buttons": [{"text": "Go", "onClick": {"x": 1}}]}}]}]}}
    assert ud.card_text(card) == "T\nS\nL\nx\nGo"

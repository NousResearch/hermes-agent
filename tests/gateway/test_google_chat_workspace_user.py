"""Google Chat: the opt-in Workspace user reached by domain-wide delegation.

Covers: off unless configured, per-feature DWD scopes, the thinking reaction in place of the
marker message (and its fallback), Workspace Events deliveries told apart from app events, which
listened messages are admitted, and the subscription upkeep (create, adopt, renew, reactivate).
"""

import asyncio
import datetime as dt
import logging
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from tests.gateway.test_google_chat import (
    GoogleChatAdapter, _FakeHttpError, _base_config, _gc_mod, _make_pubsub_message,
)

from plugins.platforms.google_chat import workspace_events as we
from plugins.platforms.google_chat.workspace_user import (
    MESSAGES_READONLY_SCOPE, REACTIONS_SCOPE, WorkspaceUserSettings,
)

USER = "assistant@example.com"
ALICE, BOB = "alice@example.com", "bob@example.com"
SPACE = "spaces/HOME"
TOPIC = "projects/p/topics/chat"


class _Call:
    """``.execute(http=...)`` wrapper around a canned result or exception."""

    def __init__(self, result=None, exc=None, hook=None):
        self._result, self._exc, self._hook = result, exc, hook

    def execute(self, http=None):
        if self._hook:
            self._hook()
        if self._exc:
            raise self._exc
        return self._result


class _FakeReactions:
    def __init__(self):
        self.created, self.deleted = [], []
        self.fail_with = None

    def create(self, parent, body):
        if self.fail_with is not None:
            return _Call(exc=self.fail_with)
        self.created.append((parent, body["emoji"]["unicode"]))
        return _Call({"name": f"{parent}/reactions/R{len(self.created)}"})

    def delete(self, name):
        self.deleted.append(name)
        return _Call({})


class _FakeUserChat:
    """``spaces().messages().reactions()`` of the delegated user's Chat client."""

    def __init__(self):
        self.fake_reactions = _FakeReactions()

    def spaces(self):
        return self

    def messages(self):
        return self

    def reactions(self):
        return self.fake_reactions


class _FakeSubscriptions:
    """In-memory Workspace Events subscriptions (the behaviour the keeper relies on)."""

    def __init__(self):
        self.subs: dict[str, dict] = {}
        self.calls: list[str] = []
        self.create_conflict = False
        self.list_hides = 0  # list() returns nothing this many times (an eventually consistent list)

    def _op(self, sub):
        return _Call({"done": True, "response": {"@type": "x", **sub}})

    def get(self, name):
        self.calls.append("get")
        if name not in self.subs:
            return _Call(exc=_FakeHttpError(404))
        return _Call(dict(self.subs[name]))

    def list(self, filter):
        self.calls.append("list")
        if self.list_hides:
            self.list_hides -= 1
            return _Call({"subscriptions": []})
        target = filter.split('target_resource="', 1)[1].rstrip('"')
        return _Call({"subscriptions": [dict(s) for s in self.subs.values() if s["targetResource"] == target]})

    def create(self, body):
        self.calls.append("create")
        if self.create_conflict:
            return _Call(exc=_FakeHttpError(409))
        name = f"subscriptions/S{len(self.subs) + 1}"
        self.subs[name] = {**body, "name": name, "state": "ACTIVE", "authority": "users/111",
                           "expireTime": _in(hours=4)}
        return self._op(self.subs[name])

    def patch(self, name, updateMask, body):
        self.calls.append("patch")
        self.subs[name]["expireTime"] = _in(hours=4)
        return self._op(self.subs[name])

    def reactivate(self, name, body):
        self.calls.append("reactivate")
        self.subs[name]["state"] = "ACTIVE"
        return self._op(self.subs[name])

    def delete(self, name):
        self.calls.append("delete")
        self.subs.pop(name, None)
        return _Call({})


class _FakeEvents:
    def __init__(self):
        self.subscriptions_ = _FakeSubscriptions()

    def subscriptions(self):
        return self.subscriptions_


def _in(**delta):
    return (dt.datetime.now(dt.UTC) + dt.timedelta(**delta)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sa_credentials():
    """A service-account credential that records the delegation it was asked for."""
    creds = MagicMock()
    creds.with_scopes.return_value = creds
    creds.with_subject.return_value = creds
    return creds


@pytest.fixture(autouse=True)
def _http_error_is_fake(monkeypatch):
    # The real googleapiclient may be installed; the adapter's ``except HttpError`` must catch ours.
    monkeypatch.setattr(_gc_mod, "HttpError", _FakeHttpError)


def _make(monkeypatch, workspace_user=None, allowed=f"{ALICE},{BOB}", **extra):
    monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", allowed)
    cfg = _base_config(**extra) if workspace_user is None else _base_config(workspace_user=workspace_user, **extra)
    a = GoogleChatAdapter(cfg)
    a._loop = asyncio.get_running_loop()
    a._chat_api = MagicMock()
    a._credentials = _sa_credentials()
    a._new_authed_http = MagicMock(return_value=MagicMock())
    a._subscription_path = "projects/test-project/subscriptions/test-sub"
    a._subscriber = MagicMock()
    a._subscriber.get_subscription.return_value = MagicMock(topic=TOPIC)
    a.handle_message = AsyncMock()
    if a._workspace_user is not None:
        a._workspace_user._clients.update(chat=_FakeUserChat(), workspaceevents=_FakeEvents())
        a._workspace_user.http = MagicMock(return_value=MagicMock())
    return a


def _event(message_id="spaces/HOME/messages/M1", chat_id=SPACE):
    source = MagicMock(chat_id=chat_id)
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id=message_id)


def _listen(**more):
    return {"email": USER, "listen_spaces": [SPACE], "listen_pattern": r"(?i)\bhermes\b", **more}


# ===========================================================================
# Configuration: off by default, explicit features, minimal scopes
# ===========================================================================


class TestSettings:
    @pytest.mark.asyncio
    async def test_absent_section_leaves_the_adapter_unchanged(self, monkeypatch):
        assert _make(monkeypatch)._workspace_user is None

    def test_email_alone_enables_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert WorkspaceUserSettings.from_extra({"workspace_user": {"email": USER}}, ALICE) is None
        assert "no feature is enabled" in caplog.text

    def test_scopes_follow_the_enabled_features(self):
        react = WorkspaceUserSettings.from_extra({"workspace_user": {"email": USER, "thinking_reaction": "🤔"}}, ALICE)
        assert react.scopes() == [REACTIONS_SCOPE]
        listen = WorkspaceUserSettings.from_extra({"workspace_user": _listen()}, ALICE)
        assert listen.scopes() == [MESSAGES_READONLY_SCOPE]
        both = WorkspaceUserSettings.from_extra({"workspace_user": _listen(thinking_reaction="👀")}, ALICE)
        assert set(both.scopes()) == {REACTIONS_SCOPE, MESSAGES_READONLY_SCOPE}

    @pytest.mark.parametrize("section", [
        {"email": "not-an-email", "thinking_reaction": "🤔"},
        "assistant@example.com",
    ])
    def test_unusable_section_is_refused(self, section):
        assert WorkspaceUserSettings.from_extra({"workspace_user": section}, ALICE) is None

    def test_listening_needs_an_allowlist(self, caplog):
        with caplog.at_level(logging.WARNING):
            settings = WorkspaceUserSettings.from_extra({"workspace_user": _listen(thinking_reaction="🤔")}, "")
        assert not settings.listening and settings.scopes() == [REACTIONS_SCOPE]
        assert "GOOGLE_CHAT_ALLOWED_USERS" in caplog.text

    @pytest.mark.parametrize("override", [{"listen_pattern": ""}, {"listen_pattern": "("},
                                          {"listen_spaces": ["general"]}])
    def test_listening_needs_a_pattern_and_space_names(self, override):
        settings = WorkspaceUserSettings.from_extra({"workspace_user": _listen(thinking_reaction="🤔", **override)}, ALICE)
        assert not settings.listening

    @pytest.mark.asyncio
    async def test_delegation_asks_for_exactly_the_enabled_scopes(self, monkeypatch):
        a = _make(monkeypatch, {"email": USER, "thinking_reaction": "🤔"})
        a._workspace_user.credentials()
        a._credentials.with_scopes.assert_called_once_with([REACTIONS_SCOPE])
        a._credentials.with_subject.assert_called_once_with(USER)

    @pytest.mark.asyncio
    async def test_credentials_that_cannot_impersonate_turn_it_off(self, monkeypatch, caplog):
        a = _make(monkeypatch, {"email": USER, "thinking_reaction": "🤔"})
        a._credentials = object()  # e.g. Application Default Credentials on GCE
        with caplog.at_level(logging.WARNING):
            await a._workspace_user.start(pubsub=True)
        assert "cannot impersonate" in caplog.text
        await a.on_processing_start(_event())
        assert not a._workspace_user._reaction_tasks


# ===========================================================================
# Thinking reaction instead of the marker message
# ===========================================================================


class TestThinkingReaction:
    @pytest.mark.asyncio
    async def test_reaction_replaces_the_marker_and_is_removed_after_the_turn(self, monkeypatch):
        a = _make(monkeypatch, {"email": USER, "thinking_reaction": "🤔"})
        reactions = a._workspace_user.chat().fake_reactions
        a._create_message = AsyncMock()
        await a.on_processing_start(_event())
        await a.send_typing(SPACE)
        assert reactions.created == [("spaces/HOME/messages/M1", "🤔")]
        a._create_message.assert_not_called()  # no "thinking…" marker message
        await a.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        assert reactions.deleted == ["spaces/HOME/messages/M1/reactions/R1"]

    @pytest.mark.asyncio
    async def test_refused_reaction_falls_back_to_the_marker_and_backs_off(self, monkeypatch):
        a = _make(monkeypatch, {"email": USER, "thinking_reaction": "🤔"})
        reactions = a._workspace_user.chat().fake_reactions
        reactions.fail_with = _FakeHttpError(403)  # the user is not a member (e.g. the app's DM)
        a._create_message = AsyncMock(return_value=MagicMock(success=True, message_id="spaces/HOME/messages/CARD"))
        await a.on_processing_start(_event())
        await a.send_typing(SPACE)
        a._create_message.assert_awaited_once()
        await a.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        reactions.fail_with = None
        await a.on_processing_start(_event("spaces/HOME/messages/M2"))
        assert not a._workspace_user._reaction_tasks and reactions.created == []

    @pytest.mark.asyncio
    async def test_reaction_created_after_the_turn_ended_is_removed(self, monkeypatch):
        a = _make(monkeypatch, {"email": USER, "thinking_reaction": "🤔"})
        reactions = a._workspace_user.chat().fake_reactions
        release, create = threading.Event(), reactions.create
        reactions.create = lambda parent, body: _Call(hook=lambda: release.wait(10), result=create(parent, body).execute())
        await a.on_processing_start(_event())
        task = a._workspace_user._reaction_tasks["spaces/HOME/messages/M1"]
        await a.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        release.set()
        await asyncio.wait_for(task, timeout=10)
        assert reactions.deleted == ["spaces/HOME/messages/M1/reactions/R1"]

    @pytest.mark.asyncio
    async def test_no_reaction_without_thinking_reaction(self, monkeypatch):
        a = _make(monkeypatch, _listen())
        await a.on_processing_start(_event())
        assert not a._workspace_user._reaction_tasks


# ===========================================================================
# Workspace Events deliveries on the app's subscription
# ===========================================================================


def _ws_message(text="hermes, are the lights off?", sender="users/alice", name="spaces/HOME/messages/W1", **more):
    return {"name": name, "text": text, "sender": {"name": sender, "type": "HUMAN"},
            "space": {"name": SPACE}, "thread": {"name": "spaces/HOME/threads/T1"}, **more}


def _ws_pubsub(message=None, ce_type=we.MESSAGE_CREATED):
    data = {"message": message or _ws_message()}
    return _make_pubsub_message(data, attributes={
        "ce-type": ce_type, "ce-source": "//workspaceevents.googleapis.com/subscriptions/S1"})


async def _listening_adapter(monkeypatch):
    a = _make(monkeypatch, _listen())
    # Upkeep is exercised separately; here only the inbound path, so no background task.
    a._workspace_user.listener = listener = we.SpaceListener(a._workspace_user, a)
    listener.members = {"users/alice": ALICE, "users/bob": BOB}
    listener.user_id = "users/self"
    listener._members_at = 10**12
    a._dispatch_message = AsyncMock()
    submitted = []
    a._submit_on_loop = lambda coro: (submitted.append(coro), coro.close())
    return a, submitted


class TestListenedMessages:
    @pytest.mark.asyncio
    async def test_matching_message_from_an_allowed_user_is_dispatched_with_its_email(self, monkeypatch):
        a, submitted = await _listening_adapter(monkeypatch)
        msg = _ws_pubsub()
        a._handle_pubsub_message(msg)
        msg.ack.assert_called_once()
        assert len(submitted) == 1
        dispatched, envelope = a._dispatch_message.call_args.args
        assert dispatched["sender"]["email"] == ALICE
        assert dispatched["argumentText"] == "hermes, are the lights off?"
        assert envelope["space"]["name"] == SPACE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message", [
        _ws_message(text="lunch at noon?"),                                   # does not address the agent
        _ws_message(sender="users/mallory"),                                  # not on the allowlist
        _ws_message(sender="users/self"),                                     # the Workspace user itself
        {**_ws_message(), "sender": {"name": "users/b", "type": "BOT"}},     # a bot
        _ws_message(slashCommand={"commandId": "1"}),                         # the app gets slash commands
        _ws_message(annotations=[{"type": "USER_MENTION", "userMention": {"user": {"type": "BOT"}}}]),
        {**_ws_message(name="spaces/OTHER/messages/X"), "space": {"name": "spaces/OTHER"}},
    ])
    async def test_other_messages_are_not_dispatched(self, monkeypatch, message):
        a, submitted = await _listening_adapter(monkeypatch)
        a._handle_pubsub_message(_ws_pubsub(message))
        assert submitted == []

    @pytest.mark.asyncio
    async def test_a_message_the_app_already_received_is_not_dispatched_twice(self, monkeypatch):
        a, submitted = await _listening_adapter(monkeypatch)
        a._dedup.is_duplicate("spaces/HOME/messages/W1")  # the app's own delivery came first
        a._handle_pubsub_message(_ws_pubsub())
        assert submitted == []

    @pytest.mark.asyncio
    async def test_workspace_events_never_reach_the_app_message_path(self, monkeypatch):
        a = _make(monkeypatch, {"email": USER, "thinking_reaction": "🤔"})  # not listening
        a._dispatch_message = AsyncMock()
        a._handle_pubsub_message(_ws_pubsub())
        a._dispatch_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifecycle_event_wakes_the_upkeep(self, monkeypatch):
        a, _ = await _listening_adapter(monkeypatch)
        a._workspace_user.listener.wake = MagicMock()
        a._handle_pubsub_message(_ws_pubsub(ce_type="google.workspace.events.subscription.v1.expirationReminder"))
        a._workspace_user.listener.wake.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_callback_mode_does_not_listen(self, monkeypatch, caplog):
        a = _make(monkeypatch, _listen())
        with caplog.at_level(logging.WARNING):
            await a._workspace_user.start(pubsub=False)
        assert a._workspace_user.listener is None and "Pub/Sub inbound" in caplog.text

    def test_batch_events_carry_every_message(self):
        attrs = {"ce-type": "google.workspace.chat.message.v1.batchCreated"}
        envelope = {"messages": [{"message": {"name": "a"}}, {"message": {"name": "b"}}, {}]}
        assert [m["name"] for m in we.event_messages(attrs, envelope)] == ["a", "b"]


# ===========================================================================
# Subscription upkeep
# ===========================================================================


def _keeper(events):
    return we.SubscriptionKeeper(lambda: events, lambda: None, TOPIC)


class TestSubscriptionKeeper:
    def test_creates_one_and_learns_the_user_id(self):
        events = _FakeEvents()
        keeper = _keeper(events)
        assert keeper.ensure(SPACE) == "created"
        sub = events.subscriptions_.subs[keeper.names[SPACE]]
        assert sub["notificationEndpoint"] == {"pubsubTopic": TOPIC}
        assert sub["payloadOptions"] == {"includeResource": True}
        assert keeper.authority == "users/111"
        assert keeper.ensure(SPACE) is None  # healthy: nothing to do

    def test_adopts_an_existing_subscription(self):
        events = _FakeEvents()
        _keeper(events).ensure(SPACE)
        fresh = _keeper(events)  # e.g. after a restart
        assert fresh.ensure(SPACE) is None and "create" not in events.subscriptions_.calls[2:]

    def test_already_exists_on_create_is_adopted(self):
        events = _FakeEvents()
        _keeper(events).ensure(SPACE)
        events.subscriptions_.list_hides, events.subscriptions_.create_conflict = 1, True
        assert _keeper(events).ensure(SPACE) == "adopted"

    def test_renews_before_expiry_and_reactivates_when_suspended(self):
        events = _FakeEvents()
        keeper = _keeper(events)
        keeper.ensure(SPACE)
        sub = events.subscriptions_.subs[keeper.names[SPACE]]
        sub["expireTime"] = _in(minutes=30)
        assert keeper.ensure(SPACE) == "renewed"
        sub["state"], sub["suspensionReason"] = "SUSPENDED", "USER_SCOPE_REVOKED"
        assert keeper.ensure(SPACE).startswith("reactivated")

    def test_a_subscription_on_another_topic_is_replaced(self):
        events = _FakeEvents()
        _keeper(events).ensure(SPACE)
        moved = we.SubscriptionKeeper(lambda: events, lambda: None, "projects/p/topics/other")
        assert moved.ensure(SPACE) == "replaced (other topic), created"
        assert [s["notificationEndpoint"]["pubsubTopic"] for s in events.subscriptions_.subs.values()] == [
            "projects/p/topics/other"]


@pytest.mark.asyncio
async def test_upkeep_subscribes_every_listen_space_and_maps_senders(monkeypatch):
    a = _make(monkeypatch, _listen(listen_spaces=[SPACE, "spaces/TEAM"]))
    events = a._workspace_user.events()

    def member(name):  # members.get by email alias -> users/<local part>
        return _Call({"member": {"name": "users/" + name.rsplit("/", 1)[-1].split("@")[0]}})
    a._chat_api.spaces.return_value.members.return_value.get.side_effect = member
    listener = we.SpaceListener(a._workspace_user, a)
    await asyncio.to_thread(listener.upkeep)
    targets = sorted(s["targetResource"] for s in events.subscriptions_.subs.values())
    assert targets == ["//chat.googleapis.com/spaces/HOME", "//chat.googleapis.com/spaces/TEAM"]
    assert listener.members == {"users/alice": ALICE, "users/bob": BOB}
    assert listener.user_id == "users/assistant"


def test_unconfigured_adapter_builds_no_workspace_user():
    assert GoogleChatAdapter(PlatformConfig(enabled=True))._workspace_user is None

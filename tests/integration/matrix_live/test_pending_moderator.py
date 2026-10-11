"""Actual homeserver authority decides pending moderator withdrawal."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from nio import JoinResponse, RoomInviteResponse, RoomRedactError, RoomRedactResponse

from tests.integration.matrix_live.conftest import _register, _wait_for
from tests.integration.matrix_live.context_client import (
    group_member as group_member,
    _send,
    hand_off,
)


@pytest.fixture
def gateway_home_setup(group_member):
    def setup(home):
        with (home / ".env").open("a") as stream:
            stream.write(
                "\nGATEWAY_ALLOW_ALL_USERS=true\nMATRIX_ALLOW_ALL_USERS=true\n"
            )
        with (home / "plugins/matrix-live-context/__init__.py").open("a") as stream:
            stream.write(OBSERVE)

    return setup


@pytest.mark.parametrize("gateway", ["pause-context"], indirect=True)
def test_actual_moderator_withdrawal_preserves_input_after_refused_redaction(
    gateway, live_room, group_member
):
    async def prepare():
        alice = live_room.observer.client(live_room.homeserver)
        bob = group_member.client(live_room.homeserver)
        charlie_account = await _register(
            live_room.homeserver, f"charlie-{uuid.uuid4().hex[:8]}"
        )
        charlie = charlie_account.client(live_room.homeserver)
        try:
            invited = await alice.room_invite(
                live_room.room_id, charlie_account.user_id
            )
            assert isinstance(invited, RoomInviteResponse), invited
            assert isinstance(await charlie.join(live_room.room_id), JoinResponse)
            root = await _send(alice, live_room.room_id, "Moderator control root")
            await _send(
                bob,
                live_room.room_id,
                f"{live_room.bot.user_id} Wait @matrix-live:pause",
                root=root,
            )
            return root, charlie_account
        finally:
            await alice.close()
            await bob.close()
            await charlie.close()

    root, charlie_account = asyncio.run(prepare())
    _wait_for(
        lambda: (gateway.home / "context-started").exists(),
        "actual author turn blocked",
        timeout=15,
    )

    async def queue():
        bob = group_member.client(live_room.homeserver)
        try:
            first = await _send(
                bob,
                live_room.room_id,
                f"{live_room.bot.user_id} Retained moderator queue",
                root=root,
            )
            second = await _send(
                bob,
                live_room.room_id,
                f"{live_room.bot.user_id} Withdrawn moderator queue",
                root=root,
            )
            return first, second
        finally:
            await bob.close()

    retained, target = asyncio.run(queue())
    _wait_for(
        lambda: (
            (gateway.home / "moderator-queued").exists()
            and "Withdrawn moderator queue"
            in (gateway.home / "moderator-queued").read_text()
        ),
        "actual pending admission",
        timeout=15,
    )

    async def denied_then_allowed():
        ordinary = charlie_account.client(live_room.homeserver)
        moderator = live_room.observer.client(live_room.homeserver)
        try:
            denied = await ordinary.room_redact(live_room.room_id, target)
            assert (
                isinstance(denied, RoomRedactError)
                and denied.status_code == "M_FORBIDDEN"
            ), denied
            assert not (gateway.home / "moderator-withdrawn").exists()
            allowed = await moderator.room_redact(live_room.room_id, target)
            assert isinstance(allowed, RoomRedactResponse), allowed
            return allowed.event_id
        finally:
            await ordinary.close()
            await moderator.close()

    redaction_id = asyncio.run(denied_then_allowed())
    _wait_for(
        lambda: (gateway.home / "moderator-withdrawn").exists(),
        "authoritative moderator withdrawal",
        timeout=15,
    )
    actual = json.loads((gateway.home / "moderator-withdrawn").read_text())
    assert actual == {
        "room": live_room.room_id,
        "author": group_member.user_id,
        "target": target,
    }
    _wait_for(
        lambda: (gateway.home / "moderator-wire").exists(),
        "actual redacted native target",
        timeout=15,
    )
    wire = json.loads((gateway.home / "moderator-wire").read_text())
    because = wire["unsigned"]["redacted_because"]
    assert (
        wire["sender"],
        wire["event_id"],
        because["sender"],
        because["event_id"],
        because.get("redacts") or because["content"]["redacts"],
    ) == (
        group_member.user_id,
        target,
        live_room.observer.user_id,
        redaction_id,
        target,
    )
    hand_off(gateway.home / "context-release", "release")
    _wait_for(
        lambda: len(gateway.model.main_requests()) == 2,
        "remaining native pending turn",
        timeout=30,
    )
    current = json.dumps(gateway.model.main_requests()[1]["messages"][-1]["content"])
    assert (
        "Retained moderator queue" in current
        and "Withdrawn moderator queue" not in current
    )


@pytest.mark.parametrize("withdrawn", [False, True])
def test_moderator_observer_signals_only_completed_withdrawal(withdrawn):
    import ast
    from types import FunctionType

    events = []

    async def withdraw(self, room, author, target):
        events.append(("completed", withdrawn))
        return withdrawn

    def signal(kind, payload):
        events.append((kind, json.loads(payload)))

    observer = next(
        node
        for node in ast.parse(OBSERVE).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "observed_withdraw"
    )
    namespace: dict[str, object] = {
        "original_withdraw": withdraw,
        "signal": signal,
        "json": json,
    }
    exec(
        compile(ast.Module(body=[observer], type_ignores=[]), __file__, "exec"),
        namespace,
    )

    observed_withdraw = namespace["observed_withdraw"]
    assert isinstance(observed_withdraw, FunctionType)

    async def exercise():
        result = observed_withdraw(None, "!room", "@author", "$target")
        before = list(events)
        completed = await result
        return {"before": before, "withdrawn": completed, "events": events}

    expected_events = [("completed", withdrawn)]
    if withdrawn:
        expected_events.append((
            "moderator-withdrawn",
            {
                "room": "!room",
                "author": "@author",
                "target": "$target",
            },
        ))
    assert asyncio.run(exercise()) == {
        "before": [],
        "withdrawn": withdrawn,
        "events": expected_events,
    }


OBSERVE = r"""
import json
from urllib.parse import quote
from plugins.platforms.matrix.intake_mixin import MatrixIntakeMixin
from plugins.platforms.matrix.redaction_mixin import MatrixRedactionMixin, _redacted_event_id
from plugins.platforms.matrix.client_events import Method
original_batch = MatrixIntakeMixin._dispatch_text_batch
async def observed_batch(self,event):
    result = await original_batch(self,event)
    if 'moderator queue' in event.text:
        with (get_hermes_home()/'moderator-queued').open('a') as stream:
            stream.write(event.text+'\n')
    return result
MatrixIntakeMixin._dispatch_text_batch = observed_batch
original_withdraw = MatrixRedactionMixin._withdraw_redacted_message
original_redact = MatrixRedactionMixin._on_redaction
async def observed_withdraw(self,room,author,target):
    result = await original_withdraw(self,room,author,target)
    if result:
        signal('moderator-withdrawn',json.dumps({'room':room,'author':author,'target':target}))
    return result
async def observed_redact(self,event):
    await original_redact(self,event)
    room = str(event.room_id)
    target = _redacted_event_id(event)
    if target:
        path = '/_matrix/client/v3/rooms/'+quote(room,safe='')+'/event/'+quote(target,safe='')
        current = await self._client.api.request(Method.GET,path)
        signal('moderator-wire',json.dumps(current))
MatrixRedactionMixin._withdraw_redacted_message = observed_withdraw
MatrixRedactionMixin._on_redaction = observed_redact
"""

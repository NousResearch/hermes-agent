"""The session door (``session:control`` → ``session_bot.relay_operation``) serves the same
install-wide mailbox as the tui_gateway door: roster/outbox/reply state is install-wide and the
Desktop drains the install root only, so a profile-scoped write is a mailbox nothing reads
(#146035)."""

import json
from types import SimpleNamespace

from gateway import session_bot
from gateway.session_contract import SessionActor
from tests.gateway.test_profile_retire_stops_turns import _Agent, _authority
from tools import bot_relay

ACTOR = SessionActor(
    "bot-relay",
    "owned",
    frozenset({"session:submit", "session:control"}),
    "relay-client",
)


def _connection(authority):
    return SimpleNamespace(authority=authority, actor=ACTOR)


def test_session_door_serves_the_install_mailbox(tmp_path, monkeypatch):
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(install))
    session_home = tmp_path / "served"
    session_home.mkdir()
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    from gateway.session_authority import LiveSession, SessionAuthority

    db = SessionDB(db_path=session_home / "state.db")
    db.create_session("s", source="test")
    epoch = begin_runtime_epoch(db, instance_id="owner")
    runner = SimpleNamespace(
        _draining=False,
        config=SimpleNamespace(multiplex_profiles=False),
        _adapter_for_source=lambda source: None,
        _running_agents={},
    )
    authority = SessionAuthority(
        runner, profile_id="owned", instance_id="owner", db=db, epoch=epoch
    )
    authority.sessions["s"] = LiveSession(
        SimpleNamespace(platform=None, user_id="human"), "route"
    )
    connection = _connection(authority)
    row = {
        "profile": "researcher",
        "handle": "researcher",
        "connection_id": "cloud-1",
        "connection_label": "Hermes Cloud",
    }

    # roster.sync through the session door lands in the install root, not the session home.
    assert session_bot.relay_operation(connection, "roster", {"agents": [row]}) == {
        "count": 1
    }
    assert [r["connection_id"] for r in bot_relay.read_remote_roster(install)] == [
        "cloud-1"
    ]

    # outbox.drain claims the install-wide outbox: an envelope queued there is served.
    target = {
        "profile": "researcher",
        "handle": "researcher",
        "connection_id": "cloud-1",
        "connection_label": "Hermes Cloud",
        "title": "",
        "description": "",
    }
    envelope = bot_relay.enqueue_envelope(
        install,
        target=target,
        message="ping",
        sender_profile="coder",
        sender_handle="coder",
    )
    drained = session_bot.relay_operation(connection, "outbox", {})
    assert [e["id"] for e in drained["envelopes"]] == [envelope["id"]]

    # reply writes the waiter's file under the install root, where the sender's watcher looks.
    assert session_bot.relay_operation(
        connection, "reply", {"id": envelope["id"], "reply": "pong"}
    ) == {"ok": True}
    replies = list(
        (bot_relay.relay_root(install) / "replies").glob(f"{envelope['id']}*")
    )
    assert replies and json.loads(replies[0].read_text())["reply"] == "pong"

    # None of the relay state split off under the session's own home.
    assert bot_relay.relay_root(session_home).exists() is False
    db.close()

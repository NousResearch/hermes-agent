"""A shutdown record queries the same persistent owner as its prepared input."""

from dataclasses import replace

import pytest
import hermes_state

from gateway.config import GatewayConfig, Platform
from gateway.input_owner import gateway_input_owner
from gateway.platforms.base_pending import (
    merge_recorded,
    reserve_pending_dispatch,
    withdraw_from_event,
)
from gateway.platforms.base_pending_merge import _absorb_pending_text
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.mark.parametrize("message_id", [None, "first"])
@pytest.mark.parametrize(
    "boundary", ["before-preparation", "after-preparation", "withdrawal", "alias"]
)
def test_prepared_input_ownership_survives_shutdown_capture(
    tmp_path, monkeypatch, message_id, boundary
):
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    homes = {profile: tmp_path / profile for profile in ("A", "B")}
    observations = []
    for index, profile in enumerate(("A", "B", "A")):
        homes[profile].mkdir(exist_ok=True)
        token = set_hermes_home_override(str(homes[profile]))
        try:
            source = SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=f"chat-{index}",
                user_id="sender",
                profile=profile,
            )
            event = MessageEvent(text="first", source=source, message_id=message_id)
            if boundary == "withdrawal":
                merge_recorded(
                    event,
                    MessageEvent(text="second", source=source, message_id="second"),
                    _absorb_pending_text,
                )
            store = SessionStore(get_hermes_home() / "sessions", GatewayConfig())
            entry = store.get_or_create_session(source)
            if boundary == "before-preparation":
                (record,) = PendingQueueSnapshot.capture(
                    entry.session_key, [event]
                ).events
            owner = gateway_input_owner(event, source)
            if boundary != "before-preparation":
                (record,) = PendingQueueSnapshot.capture(
                    entry.session_key, [event]
                ).events
            restored = decode_pending_event(record)
            if boundary == "withdrawal":
                changed, restored = withdraw_from_event(
                    restored, lambda part: part.message_id == message_id
                )
                assert changed is True
                assert restored is not None
            if boundary == "alias":

                class Adapter:
                    _pending_dispatch_reservations = {}

                reservation = reserve_pending_dispatch(
                    Adapter(), entry.session_key, restored
                )
                alias = replace(restored)
                reservation.bind(alias)
                restored = alias
            replay_owner = gateway_input_owner(restored, restored.source)
            db = store._db_for_session_id(entry.session_id)
            assert store.has_input_owner(entry.session_id, replay_owner) is False
            db.append_message(
                entry.session_id,
                "user",
                "prepared input",
                display_metadata={"gateway_input_owner": owner},
            )
            observations.append((
                profile,
                db.db_path.parent == homes[profile],
                replay_owner == owner,
                store.has_input_owner(entry.session_id, replay_owner),
            ))
        finally:
            reset_hermes_home_override(token)
    assert observations == [
        ("A", True, True, True),
        ("B", True, True, True),
        ("A", True, True, True),
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "B"),
        ("scope", "other-workspace"),
        ("pending_uid", "another-input"),
        ("owner", "another-receipt"),
        ("identifier", "pending:another-input"),
    ],
)
def test_restored_owner_rejects_another_pending_scope(field, value):
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="room",
        user_id="sender",
        profile="A",
        scope_id="workspace",
    )
    event = MessageEvent(text="input", source=source, message_id="input")
    (record,) = PendingQueueSnapshot.capture("session", [event]).events
    owner = gateway_input_owner(event, source)
    state = {
        "namespace": ["slack", "A", "workspace", "room", None],
        "identifier": f"pending:{record['uid']}",
        "pending_uid": record["uid"],
        "owner": owner,
    }
    if field in {"profile", "scope"}:
        state["namespace"][{"profile": 1, "scope": 2}[field]] = value
    else:
        state[field] = value
    record["input_owner"] = state
    with pytest.raises(ValueError, match="owner"):
        decode_pending_event(record)

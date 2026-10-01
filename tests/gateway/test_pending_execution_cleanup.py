"""Only canonical user execution receipts consume shutdown input records."""

import json
from types import SimpleNamespace

import pytest
import hermes_state

from gateway.config import GatewayConfig, Platform
from gateway.input_owner import gateway_input_owner
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from gateway.session_transcript import TranscriptReadError
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_flush import recover_gateway_pending
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from utils import atomic_json_write


@pytest.mark.parametrize(
    "receipt", ["user", "assistant", "none", "unreadable", "all", "foreign"]
)
def test_restart_consumes_only_persisted_user_ownership(tmp_path, monkeypatch, receipt):
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    observations = []
    for index, profile in enumerate(("A", "B", "A")):
        home = tmp_path / profile
        home.mkdir(exist_ok=True)
        token = set_hermes_home_override(str(home))
        try:
            storage_home = (
                home / "foreign" if receipt == "foreign" else get_hermes_home()
            )
            store = SessionStore(storage_home / "sessions", GatewayConfig())
            if receipt == "foreign":
                store._db = store._open_session_db_for_active_scope(
                    db_path=storage_home / "state.db"
                )
            source = SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=f"chat-{index}",
                user_id="sender",
                profile=None if receipt == "foreign" else profile,
            )
            entry = store.get_or_create_session(source)
            events = [
                MessageEvent(text=text, source=source, message_id=text)
                for text in ("first", "second")
            ]
            owner = gateway_input_owner(events[0], source)
            setattr(events[0], "_gateway_accepted", True)
            snapshot = PendingQueueSnapshot.capture(entry.session_key, events)
            directory = home / "pending_messages"
            directory.mkdir(exist_ok=True)
            path = directory / f"pending-{index}.json"
            atomic_json_write(path, snapshot.to_payload(), mode=0o600)
            if receipt in {"user", "assistant", "all", "foreign"}:
                store._db_for_session_id(entry.session_id).append_message(
                    entry.session_id,
                    "user" if receipt in {"all", "foreign"} else receipt,
                    "first",
                    display_metadata={"gateway_input_owner": owner},
                )
            if receipt == "all":
                store._db_for_session_id(entry.session_id).append_message(
                    entry.session_id,
                    "user",
                    "second",
                    display_metadata={
                        "gateway_input_owner": gateway_input_owner(events[1], source)
                    },
                )
            if receipt == "unreadable":

                def unavailable(session_id, owner):
                    raise TranscriptReadError(session_id)

                monkeypatch.setattr(store, "has_input_owner", unavailable)
            recovered = recover_gateway_pending(
                SimpleNamespace(config=store.config, session_store=store)
            )
            remaining = (
                PendingQueueSnapshot.from_payload(json.loads(path.read_text())).events
                if path.exists()
                else []
            )
            observations.append((
                profile,
                recovered,
                [record["uid"] for record in remaining],
                [record["event"]["text"] for record in remaining],
            ))
            expected = (
                []
                if receipt == "all"
                else snapshot.events[1:]
                if receipt == "user"
                else snapshot.events
            )
            assert observations[-1] == (
                profile,
                len(snapshot.events) - len(expected),
                [record["uid"] for record in expected],
                [record["event"]["text"] for record in expected],
            )
            before = path.read_bytes() if path.exists() else None
            assert (
                recover_gateway_pending(
                    SimpleNamespace(config=store.config, session_store=store)
                )
                == 0
            )
            assert (path.read_bytes() if path.exists() else None) == before
        finally:
            reset_hermes_home_override(token)
    assert [item[0] for item in observations] == ["A", "B", "A"]

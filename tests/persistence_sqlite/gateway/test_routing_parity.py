"""The SQLite ``RoutingRepository``'s latest-peer read: the recovered session record carries the stored row's lineage markers, its decoded origin and model-config JSON and its title source.

The seed runs through the backend's own session, message and SQL writes under a fixed clock.
"""

from __future__ import annotations

import json
import time
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from persistence.conversation.sessions import SessionLineageKind
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session


NOW = 1_800_000_000.0


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> Clock:
    fixed = Clock()
    monkeypatch.setattr(time, "time", fixed)
    return fixed


def _set(db, sql: str, params: Any = ()) -> None:
    db._write_sql(sql, params)


def _keyed(db, sid: str, key: str | None, *, source: str = "telegram", started: float = NOW - 100,
           **cols: Any) -> None:
    db.create_session(sid, source)
    names = ("session_key", "started_at", *cols)
    _set(db, f"UPDATE sessions SET {', '.join(f'{n} = ?' for n in names)} WHERE id = ?",
         (key, started, *cols.values(), sid))


def _backend_seeder(store) -> SimpleNamespace:
    """The backend's own session, message and SQL writes under the ``SessionDB`` names the seeds call."""
    holder = store.holder()
    return SimpleNamespace(create_session=partial(create_session, holder), append_message=partial(append_message, holder),
                           _write_sql=holder._write_sql)


def _seed_peers(db) -> None:
    """Keyed rows of one exact key (with and without messages), a fenced key, compression and fork lineages and
    tuple rows of this profile, an unowned legacy row and a sibling profile."""
    _keyed(db, "k-msg", "K", last_activity_at=NOW - 50, chat_id="c", chat_type="dm", user_id="u")
    db.append_message("k-msg", "user", "hi", timestamp=NOW - 50)
    _keyed(db, "k-empty", "K", last_activity_at=NOW - 10, origin_json='{"chat": "c"}', display_name="D",
           model_config=json.dumps({"_branched_from": " k-msg ", "x": 1}), title="T", title_source="user")
    _keyed(db, "r-old", "R", last_activity_at=NOW - 90)
    db.append_message("r-old", "user", "x", timestamp=NOW - 90)
    _keyed(db, "r-reset", "R", ended_at=NOW - 40, end_reason="session_reset")
    _keyed(db, "t-mine", "T1", chat_id="tc", chat_type="dm", user_id="tu", profile_name="profile-a",
           last_activity_at=NOW - 30)
    db.append_message("t-mine", "user", "m", timestamp=NOW - 30)
    _keyed(db, "t-null", "T2", chat_id="tc", chat_type="dm", user_id="tu", last_activity_at=NOW - 60)
    db.append_message("t-null", "user", "m", timestamp=NOW - 60)
    _keyed(db, "t-other", "T3", chat_id="tc", chat_type="dm", user_id="tu", profile_name="other",
           last_activity_at=NOW - 5)
    db.append_message("t-other", "user", "m", timestamp=NOW - 5)
    _keyed(db, "root", "L", ended_at=NOW - 80, end_reason="compression")
    _keyed(db, "tip", "L", parent_session_id="root")
    _keyed(db, "fork", "F", parent_session_id="tip", model_config=json.dumps({"_branched_from": "tip"}))


def test_recovered_session_record_projects_markers_and_json(backend, scope_a):
    store = backend.open(scope_a)
    _seed_peers(_backend_seeder(store))
    _set(store.holder(), "UPDATE sessions SET message_count = 1 WHERE id = 'k-empty'")
    found = store.routing.latest_session_for_peer(source="telegram", session_key="K")
    record = found.session
    assert (record.session_id, found.has_messages) == ("k-empty", True)
    assert record.lineage_kind is SessionLineageKind.BRANCH and record.lineage_origin_id == " k-msg "
    assert (record.branched_from, record.reset_from) == ("k-msg", None)
    assert record.origin == {"chat": "c"} and record.model_config == {"_branched_from": " k-msg ", "x": 1}
    assert record.title_source.value == "user" and record.profile_name == "profile-a"

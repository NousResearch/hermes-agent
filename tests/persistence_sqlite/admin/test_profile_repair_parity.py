"""``SQLiteProfileRepairAdmin`` on a read-only handle: on a seeded store every write method refuses and leaves every non-FTS table as it was, and on a profile with no ``state.db`` the reads answer as the profile-repair command's empty stand-in without creating the file."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli.sessions_repair_profiles import _EMPTY_STORE
from persistence import PermanentStoreError, ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteProfileRepairAdmin
from tests.fixtures.storage_seed import seed_messages, seed_session

ROOT_KEY = "agent:main:telegram:dm:100"
ACME_KEY = "agent:acme:telegram:dm:200"
STRAY_KEY = "agent:acme:telegram:dm:202"
SCOPE = "/routing-scope"


def _store_session(store, sid: str, key: str, *, profile: str, parent=None, messages=2, title=None) -> None:
    """One telegram DM session with ``messages`` alternating user and assistant rows, written through the backend's store."""
    seed_session(store, sid, "telegram", session_key=key, chat_id=key.rsplit(":", 1)[-1], chat_type="dm",
                 profile_name=profile, parent_session_id=parent, system_prompt="sys")
    seed_messages(store, sid, *({"role": "user" if i % 2 == 0 else "assistant", "content": f"{sid} m{i}"}
                                for i in range(messages)))
    if title is not None:
        store.sessions.set_title(sid, title)


def _admit(path: Path, sid: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,lineage_json,"
            "payload_json,payload_digest,intent,status,outcome,owner_epoch,generation) "
            "VALUES(?,?,'p',?,'[]','{}','d','queue','terminal','completed',1,1)",
            (f"adm-{sid}", f"req-{sid}", sid))
        conn.commit()


def _store_seed(path: Path) -> Path:
    """One store holding every crossing the profile-repair finders name, the rows the fixers settle and a ledgered session."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scope = ProfileScope(profile_dir=path.parent, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: path})
    try:
        store = backend.open(scope)
        _store_session(store, "ok-root", ROOT_KEY, profile="default")
        _store_session(store, "mislabelled", ACME_KEY, profile="default")
        _store_session(store, "stray-parent", STRAY_KEY, profile="acme", messages=3, title="Stray parent")
        _store_session(store, "stray-child", STRAY_KEY, profile="acme", parent="stray-parent", title="Shared title")
        _store_session(store, "crossed-child", "agent:main:telegram:dm:203", profile="default", parent="stray-parent")
        _store_session(store, "legacy-main", "agent:main:telegram:dm:204", profile="default")
        _store_session(store, "ledgered", "agent:acme:telegram:dm:205", profile="acme")
        store.routing.upsert_routing_entry(ROOT_KEY, json.dumps({"session_key": ROOT_KEY, "session_id": "ok-root"}),
                                           routing_scope=SCOPE)
        store.routing.upsert_routing_entry(
            ACME_KEY, json.dumps({"session_key": ACME_KEY, "session_id": "mislabelled"}), routing_scope=SCOPE)
        store.routing.enable_topic_mode(chat_id="200", user_id="200", profile_name="default")
        store.routing.bind_topic(chat_id="200", thread_id="7", user_id="200", session_key=ACME_KEY,
                                 session_id="mislabelled", profile_name="default")
        store.routing.bind_topic(chat_id="200", thread_id="8", user_id="200", session_key=STRAY_KEY,
                                 session_id="stray-parent", profile_name="default")
        store.routing.bind_topic(chat_id="200", thread_id="8", user_id="200", session_key=STRAY_KEY,
                                 session_id="stray-child", profile_name="acme")
    finally:
        backend.close()
    return _raw_rows(path)


def _raw_rows(path: Path) -> Path:
    """The rows written past the store's verbs: ``ledgered``'s terminal admission and a model-usage row."""
    _admit(path, "ledgered")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("INSERT INTO session_model_usage (session_id, model, input_tokens) VALUES ('stray-parent', 'm', 7)")
        conn.commit()
    return path


def _clone(src: Path, *destinations: Path) -> None:
    for dst in destinations:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(src)) as source, closing(sqlite3.connect(dst)) as target:
            source.backup(target)


def _dump(path: Path) -> dict:
    """Every user table but the FTS ones as sorted row reprs."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE '%_fts%' ORDER BY name")]
        return {t: sorted(map(repr, conn.execute(f"SELECT * FROM {t}").fetchall())) for t in tables}


@pytest.fixture
def store_pair(tmp_path):
    """``(legacy path, admin path)``: two identical copies of one seeded store."""
    seed = _store_seed(tmp_path / "seed" / "state.db")
    legacy, admin = tmp_path / "legacy" / "state.db", tmp_path / "admin" / "state.db"
    _clone(seed, legacy, admin)
    return legacy, admin


WRITES = (
    ("relabel_sessions_to_key_profile", (["mislabelled"],), {}),
    ("sever_crossed_parents", (["crossed-child"],), {}),
    ("import_moved_session", ({"session": {"id": "new"}},), {"profile_name": "acme"}),
    ("delete_moved_session", ("stray-parent",), {}),
    ("rekey_legacy_main_sessions", (["legacy-main"], "acme"), {}),
    ("delete_gateway_routing_rows", ([(SCOPE, ROOT_KEY)],), {}),
    ("insert_gateway_routing_rows_if_absent", ([(SCOPE, "k", "{}", 1.0)],), {}),
    ("relabel_telegram_topic_rows", ([{"session_key": ACME_KEY, "chat_id": "200", "thread_id": "7"}],), {}),
)


@pytest.mark.parametrize(("method", "args", "kwargs"), WRITES)
def test_a_read_only_handle_refuses_every_write(store_pair, method, args, kwargs):
    admin = store_pair[1]
    before = _dump(admin)
    with pytest.raises(PermanentStoreError):
        getattr(SQLiteProfileRepairAdmin(admin, read_only=True), method)(*args, **kwargs)
    assert _dump(admin) == before


def test_a_read_only_handle_on_a_missing_store_reads_empty_and_creates_nothing(tmp_path):
    """The legacy run reads a profile with no ``state.db`` through its empty stand-in; the admin answers the same
    without opening, and the two moves' reads answer empty."""
    path = tmp_path / "profiles" / "acme" / "state.db"
    path.parent.mkdir(parents=True)
    admin = SQLiteProfileRepairAdmin(path, read_only=True)
    assert admin.find_crossed_profile_sessions("acme") == _EMPTY_STORE.find_crossed_profile_sessions("acme")
    assert admin.find_profile_less_telegram_topic_rows() == _EMPTY_STORE.find_profile_less_telegram_topic_rows()
    assert admin.list_gateway_routing_rows() == _EMPTY_STORE.list_gateway_routing_rows()
    assert admin.key_profiles_for_chat("telegram", "1") == _EMPTY_STORE.key_profiles_for_chat("telegram", "1")
    assert admin.count_messages_all("x") == 0
    assert admin.export_session_for_move("x") is None
    assert list(path.parent.iterdir()) == []

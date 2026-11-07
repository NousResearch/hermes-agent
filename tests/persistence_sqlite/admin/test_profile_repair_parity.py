"""``SQLiteProfileRepairAdmin`` against the legacy store's profile-repair methods: every case clones one seeded store
into two identical files, runs the legacy method on one and the admin method on the other, and compares the results
and every non-FTS table the two files hold afterwards."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli.sessions_repair_profiles import _EMPTY_STORE
from persistence import PermanentStoreError, ProfileScope, SessionLedgered
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteProfileRepairAdmin
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite._legacy_store import legacy_store

__all__ = ["legacy_store"]

ROOT_KEY = "agent:main:telegram:dm:100"
ACME_KEY = "agent:acme:telegram:dm:200"
STRAY_KEY = "agent:acme:telegram:dm:202"
SCOPE = "/routing-scope"


def _session(db, sid: str, key: str, *, profile: str, parent=None, messages=2, title=None) -> None:
    db.create_session(sid, "telegram", session_key=key, chat_id=key.rsplit(":", 1)[-1], chat_type="dm",
                      profile_name=profile, parent_session_id=parent, system_prompt="sys")
    for i in range(messages):
        db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"{sid} m{i}")
    if title is not None:
        db.set_session_title(sid, title)


def _store_session(store, sid: str, key: str, *, profile: str, parent=None, messages=2, title=None) -> None:
    """``_session`` through the backend's store."""
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


def _seed(path: Path) -> Path:
    """One store holding every crossing the finders name, the rows the fixers settle and a ledgered session."""
    from hermes_state import SessionDB

    path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=path)
    try:
        _session(db, "ok-root", ROOT_KEY, profile="default")
        _session(db, "mislabelled", ACME_KEY, profile="default")
        _session(db, "stray-parent", STRAY_KEY, profile="acme", messages=3, title="Stray parent")
        _session(db, "stray-child", STRAY_KEY, profile="acme", parent="stray-parent", title="Shared title")
        _session(db, "crossed-child", "agent:main:telegram:dm:203", profile="default", parent="stray-parent")
        _session(db, "legacy-main", "agent:main:telegram:dm:204", profile="default")
        _session(db, "ledgered", "agent:acme:telegram:dm:205", profile="acme")
        db.save_gateway_routing_entry(ROOT_KEY, json.dumps({"session_key": ROOT_KEY, "session_id": "ok-root"}),
                                      scope=SCOPE)
        db.save_gateway_routing_entry(ACME_KEY, json.dumps({"session_key": ACME_KEY, "session_id": "mislabelled"}),
                                      scope=SCOPE)
        db.enable_telegram_topic_mode(chat_id="200", user_id="200", profile_name="default")
        db.bind_telegram_topic(chat_id="200", thread_id="7", user_id="200", session_key=ACME_KEY,
                               session_id="mislabelled", profile_name="default")
        db.bind_telegram_topic(chat_id="200", thread_id="8", user_id="200", session_key=STRAY_KEY,
                               session_id="stray-parent", profile_name="default")
        db.bind_telegram_topic(chat_id="200", thread_id="8", user_id="200", session_key=STRAY_KEY,
                               session_id="stray-child", profile_name="acme")
    finally:
        db.close()
    return _raw_rows(path)


def _store_seed(path: Path) -> Path:
    """``_seed``'s rows written through the backend's store."""
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
def pair(tmp_path, legacy_store):
    """``(legacy path, admin path)``: two identical copies of one seeded store."""
    seed = _seed(tmp_path / "seed" / "state.db")
    legacy, admin = tmp_path / "legacy" / "state.db", tmp_path / "admin" / "state.db"
    _clone(seed, legacy, admin)
    return legacy, admin


@pytest.fixture
def store_pair(tmp_path):
    """``pair`` seeded through the backend's store, for the cases that compare with no main code."""
    seed = _store_seed(tmp_path / "seed" / "state.db")
    legacy, admin = tmp_path / "legacy" / "state.db", tmp_path / "admin" / "state.db"
    _clone(seed, legacy, admin)
    return legacy, admin


def _legacy(path: Path, method: str, *args, **kwargs):
    from hermes_state import SessionDB

    db = SessionDB(db_path=path)
    try:
        return getattr(db, method)(*args, **kwargs)
    finally:
        db.close()


def _both(pair, method: str, *args, **kwargs):
    """The legacy and the admin result of one call, each on its own copy."""
    legacy, admin = pair
    expected = _legacy(legacy, method, *args, **kwargs)
    actual = getattr(SQLiteProfileRepairAdmin(admin, read_only=False), method)(*args, **kwargs)
    return expected, actual


READS = (
    ("find_crossed_profile_sessions", ("default",)),
    ("find_crossed_profile_sessions", ("acme",)),
    ("list_gateway_routing_rows", ()),
    ("find_profile_less_telegram_topic_rows", ()),
    ("count_messages_all", ("stray-parent",)),
    ("count_messages_all", ("missing",)),
    ("key_profiles_for_chat", ("telegram", "202")),
    ("key_profiles_for_chat", ("telegram", 999)),
)


@pytest.mark.parametrize(("method", "args"), READS)
def test_reads_match_on_a_writable_handle(pair, method, args):
    expected, actual = _both(pair, method, *args)
    assert actual == expected
    assert _dump(pair[1]) == _dump(pair[0])


@pytest.mark.parametrize(("method", "args"), READS)
def test_reads_match_on_a_read_only_handle(pair, method, args):
    from hermes_state import SessionDB

    legacy, admin = pair
    db = SessionDB(legacy, read_only=True)
    try:
        expected = getattr(db, method)(*args)
    finally:
        db.close()
    before = _dump(admin)
    assert getattr(SQLiteProfileRepairAdmin(admin, read_only=True), method)(*args) == expected
    assert _dump(admin) == before


def test_the_seed_carries_every_finding(pair):
    found = _both(pair, "find_crossed_profile_sessions", "default")[1]
    assert {row["id"] for row in found["mislabelled"]} == {"mislabelled"}
    assert {row["id"] for row in found["foreign"]} == {"mislabelled", "stray-parent", "stray-child", "ledgered"}
    assert [row["ledgered"] for row in found["foreign"] if row["id"] == "ledgered"] == [True]
    assert {row["id"] for row in found["crossed_parents"]} == {"crossed-child"}
    assert len(_both(pair, "find_profile_less_telegram_topic_rows")[1]) == 2


def test_relabel_sessions_to_key_profile(pair):
    ids = ["mislabelled", "ok-root", "missing"]
    expected, actual = _both(pair, "relabel_sessions_to_key_profile", ids)
    assert actual == expected == 1
    assert _dump(pair[1]) == _dump(pair[0])
    assert _both(pair, "relabel_sessions_to_key_profile", []) == (0, 0)


def test_sever_crossed_parents(pair):
    expected, actual = _both(pair, "sever_crossed_parents", ["crossed-child", "stray-child"])
    assert actual == expected == 1
    assert _dump(pair[1]) == _dump(pair[0])


def test_export_session_for_move(pair):
    expected, actual = _both(pair, "export_session_for_move", "stray-parent")
    assert actual == expected
    assert len(actual["messages"]) == 3 and len(actual["usage"]) == 1 and actual["system_prompt"] == "sys"
    assert _both(pair, "export_session_for_move", "missing") == (None, None)


def test_export_of_a_ledgered_session_refuses(pair):
    from hermes_state_profile_repair import SessionLedgeredError

    legacy, admin = pair
    with pytest.raises(SessionLedgeredError) as legacy_error:
        _legacy(legacy, "export_session_for_move", "ledgered")
    with pytest.raises(SessionLedgered) as admin_error:
        SQLiteProfileRepairAdmin(admin, read_only=False).export_session_for_move("ledgered")
    assert str(admin_error.value) == str(legacy_error.value)


def test_import_moved_session(tmp_path, pair):
    """A target holding one unrelated session with the moved row's title: the parent link is dropped, the title
    takes the id tail, and a second import reports ``present``."""
    from hermes_state import SessionDB

    payload = _legacy(pair[0], "export_session_for_move", "stray-child")
    target_seed = tmp_path / "target-seed" / "state.db"
    target_seed.parent.mkdir(parents=True)
    db = SessionDB(db_path=target_seed)
    try:
        _session(db, "resident", "agent:acme:telegram:dm:900", profile="acme", title="Shared title")
    finally:
        db.close()
    targets = (tmp_path / "target-legacy" / "state.db", tmp_path / "target-admin" / "state.db")
    _clone(target_seed, *targets)
    for _ in range(2):
        expected, actual = _both(targets, "import_moved_session", payload, profile_name="acme")
        assert actual == expected
        assert _dump(targets[1]) == _dump(targets[0])
    assert actual == "present"
    assert _dump(targets[1]) != _dump(target_seed)


def test_delete_moved_session(pair, monkeypatch):
    hooks: list[tuple[str, dict]] = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: hooks.append((name, kw)) or [])
    expected, actual = _both(pair, "delete_moved_session", "stray-parent")
    assert actual is expected is True
    assert _dump(pair[1]) == _dump(pair[0])
    assert len(hooks) == 2 and hooks[0] == hooks[1]
    assert hooks[0][0] == "on_session_delete" and hooks[0][1]["reason"] == "profile_repair"
    hooks.clear()
    assert _both(pair, "delete_moved_session", "missing") == (False, False)
    assert hooks == []


@pytest.mark.parametrize("profile", ["acme", "main"])
def test_rekey_legacy_main_sessions(pair, profile):
    expected, actual = _both(pair, "rekey_legacy_main_sessions", ["legacy-main", "mislabelled", "missing"], profile)
    assert actual == expected == 1
    assert _dump(pair[1]) == _dump(pair[0])


def test_delete_gateway_routing_rows(pair):
    expected, actual = _both(pair, "delete_gateway_routing_rows", [(SCOPE, ROOT_KEY), (SCOPE, "agent:x:y:z")])
    assert actual == expected == 1
    assert _dump(pair[1]) == _dump(pair[0])


def test_insert_gateway_routing_rows_if_absent(pair):
    rows = [(SCOPE, ROOT_KEY, "{}", 1.0), (SCOPE, "agent:acme:telegram:dm:777", '{"session_id": "x"}', 2.0)]
    expected, actual = _both(pair, "insert_gateway_routing_rows_if_absent", rows)
    assert actual == expected == 1
    assert _dump(pair[1]) == _dump(pair[0])


def test_relabel_telegram_topic_rows(pair):
    rows = _legacy(pair[0], "find_profile_less_telegram_topic_rows")
    expected, actual = _both(pair, "relabel_telegram_topic_rows", rows)
    assert actual == expected
    assert actual["bindings_relabelled"] == 1 and actual["bindings_duplicates_removed"] == 1
    assert _dump(pair[1]) == _dump(pair[0])
    zero = dict.fromkeys(expected, 0)
    assert _both(pair, "relabel_telegram_topic_rows", []) == (zero, zero)


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

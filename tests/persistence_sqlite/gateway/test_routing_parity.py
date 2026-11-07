"""The SQLite ``RoutingRepository`` against the legacy store, verb by verb.

Each case seeds the profile's ``state.db`` with the legacy ``SessionDB`` under a fixed clock, keeps a copy of the
seed, runs the legacy method, restores the seed and runs the backend verb, then compares the outcomes (records
projected to the legacy values) and the routing, session, topic and fence rows both runs left. The case that reads the
backend alone runs its seed through the backend's own writes.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from gateway.session_route_recovery import _messaging_platform
from persistence import PersistenceError, SessionRowMissing
from persistence.conversation.sessions import SessionLineageKind
from persistence.core.artifacts import SessionArtifacts
from persistence.gateway.routing import RoutingRepository
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session
from tests.persistence_sqlite._legacy_store import legacy_modules, legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

NOW = 1_800_000_000.0
DAY = 86400.0
HORIZON_S = 7 * DAY
MEDIA_ROOT = "/nonexistent/native-inputs"
_META_KEYS = "key GLOB 'gateway.*' OR key GLOB 'telegram*'"
_TOPIC_TABLES = ("telegram_dm_topic_mode", "telegram_dm_topic_bindings")


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


def _copy(src: Path, dst: Path) -> None:
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)


def _dump(path: Path) -> dict[str, list]:
    """The rows the routing verbs touch: sessions, messages, routing, topic tables and gateway/topic meta keys."""
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        out: dict[str, list] = {
            "state_meta": conn.execute(f"SELECT key, value FROM state_meta WHERE {_META_KEYS} ORDER BY key").fetchall(),
            "indexes": conn.execute("SELECT name, sql FROM sqlite_master WHERE type='index' "
                                    "AND tbl_name LIKE 'telegram%' ORDER BY name").fetchall(),
        }
        for table in ("sessions", "messages", "gateway_routing", *_TOPIC_TABLES):
            if table in tables:
                out[table] = conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
                out[table + ".columns"] = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]
    return out


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "ok", call()
    except Exception as exc:  # compared by class and message
        return "error", exc


def run_both(backend, scope, seed: Callable[[Any], None], legacy: Callable[[Any], Any],
             verb: Callable[[RoutingRepository], Any]) -> tuple[tuple[str, Any], tuple[str, Any]]:
    """Seed, run the legacy call, restore the seed, run the backend call; assert equal rows; return both outcomes."""
    path = store_for(scope)
    with legacy_modules():
        from hermes_state import SessionDB

        db = SessionDB(db_path=path)
        try:
            seed(db)
        finally:
            db.close()
        snapshot = path.with_name("seed.db")
        _copy(path, snapshot)
        db = SessionDB(db_path=path)
        try:
            before = _outcome(lambda: legacy(db))
        finally:
            db.close()
    legacy_rows = _dump(path)
    _copy(snapshot, path)
    store = backend.open(scope)
    after = _outcome(lambda: verb(store.routing))
    assert _dump(path) == legacy_rows
    return before, after


def assert_same(before: tuple[str, Any], after: tuple[str, Any], project: Callable[[Any], Any] = lambda v: v) -> None:
    """Equal values, or the same refusal: a legacy SQLite error arrives as a contract error chaining it."""
    assert after[0] == before[0], (before, after)
    if before[0] == "ok":
        assert project(after[1]) == before[1]
        return
    legacy_exc, exc = before[1], after[1]
    if isinstance(legacy_exc, sqlite3.Error):
        assert isinstance(exc, PersistenceError) and type(exc.__cause__) is type(legacy_exc)
    else:
        assert isinstance(exc, type(legacy_exc))
    assert str(exc) == str(legacy_exc)


def _entry(session_id: str, **extra: Any) -> str:
    return json.dumps({"session_id": session_id, **extra})


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


# ── routing index and route fences ──────────────────────────────────────────


def _seed_routes(clock: Clock) -> Callable[[Any], None]:
    """A live keyed row and its route, another profile's route, text naming no session, this store's own deleted
    session, a permanent fence of ``work`` and a sweep fence of ``work`` that aged past the horizon."""
    def seed(db) -> None:
        _keyed(db, "live", "agent:main:telegram:dm:1")
        db.save_gateway_routing_entry("agent:main:telegram:dm:1", _entry("live"), scope="r")
        db.save_gateway_routing_entry("agent:work:telegram:dm:2", _entry("other"), scope="r")
        db.save_gateway_routing_entry("odd", "[1,2]", scope="r")
        db.create_session("gone", "telegram")
        db.save_gateway_routing_entry("agent:main:telegram:dm:3", _entry("gone"))
        db.delete_session("gone")
        db.retire_gateway_routes(["fenced"], namespace="work")
        clock.now = NOW - HORIZON_S - 60
        db.retire_gateway_routes(["swept"], namespace="work", expiring=["swept"])
        clock.now = NOW
    return seed


_SAVES = (
    ("agent:main:telegram:dm:9", _entry("live", n=2), "r"),
    ("agent:work:x:y", _entry("fenced"), "r"),
    ("agent:main:x:y", _entry("fenced"), "r"),
    ("agent:main:a:b", _entry("gone"), ""),
    ("agent:work:a:b", _entry("gone"), ""),
    ("agent:work:s:t", _entry("swept"), ""),
    ("", _entry("x"), "r"),
    ("k2", "", "r"),
)


def test_upsert_skips_retired_routes_as_the_legacy_save_does(backend, scope_a, clock):
    before, after = run_both(
        backend, scope_a, _seed_routes(clock),
        lambda db: [db.save_gateway_routing_entry(k, e, scope=s) for k, e, s in _SAVES],
        lambda routing: [routing.upsert_routing_entry(k, e, routing_scope=s) for k, e, s in _SAVES])
    assert_same(before, after)


def test_replace_returns_the_retired_keys_and_keeps_other_scopes(backend, scope_a, clock):
    entries = {k: e for k, e, s in _SAVES}
    before, after = run_both(
        backend, scope_a, _seed_routes(clock),
        lambda db: db.replace_gateway_routing_entries(entries, scope="r"),
        lambda routing: routing.replace_routing_entries(entries, routing_scope="r"))
    assert_same(before, after)
    assert after[1] == {"agent:work:x:y", "agent:main:x:y", "agent:main:a:b"}


@pytest.mark.parametrize("text", ["not json", "x{"])
@pytest.mark.parametrize("operation", ["upsert", "replace", "route_retired"])
def test_malformed_entry_text_raises_the_legacy_error_and_writes_nothing(backend, scope_a, clock, legacy_store, text,
                                                                         operation):
    from hermes_state_mutation_retirement import route_retired
    legacy, verb = {
        "upsert": (lambda db: db.save_gateway_routing_entry("k", text),
                   lambda r: r.upsert_routing_entry("k", text)),
        "replace": (lambda db: db.replace_gateway_routing_entries({"a": text, "b": _entry("s")}, scope="r"),
                    lambda r: r.replace_routing_entries({"a": text, "b": _entry("s")}, routing_scope="r")),
        "route_retired": (lambda db: route_retired(db, "k", text), lambda r: r.route_retired("k", text)),
    }[operation]
    before, after = run_both(backend, scope_a, _seed_routes(clock), legacy, verb)
    assert before[0] == "error"
    assert_same(before, after)


def test_route_reads_match_the_legacy_reads(backend, scope_a, clock, legacy_store):
    from hermes_state_mutation_retirement import route_retired
    keys = [(k, e) for k, e, _ in _SAVES if k and e]

    def legacy(db):
        return ([db.load_gateway_routing_entries(scope=s) for s in ("", "r", "zz")],
                [route_retired(db, k, e) for k, e in keys],
                [db.gateway_routing_entry_for_session(s) for s in ("live", "other", "missing")])

    def verb(routing):
        return ([routing.load_routing_entries(routing_scope=s) for s in ("", "r", "zz")],
                [routing.route_retired(k, e) for k, e in keys],
                [routing.routing_entry_for_session(s) for s in ("live", "other", "missing")])
    before, after = run_both(backend, scope_a, _seed_routes(clock), legacy, verb)
    assert_same(before, after)
    assert after[1][1] == [False, True, True, True, False, False]


@pytest.mark.parametrize("ids", [["live", "other", "", None], ["", None]])
def test_fence_retired_routes_reclaims_aged_fences_and_drops_fenced_routes(backend, scope_a, clock, ids):
    before, after = run_both(
        backend, scope_a, _seed_routes(clock),
        lambda db: db.retire_gateway_routes(ids, namespace="work", expiring=["other"]),
        lambda routing: routing.fence_retired_routes(ids, namespace="work", expiring=["other"]))
    assert_same(before, after)


# ── routing peers ───────────────────────────────────────────────────────────


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


_PEER_CALLS = (
    dict(source="telegram", session_key="K"),
    dict(source="telegram", session_key="K", chat_id="c", chat_type="dm"),
    dict(source="discord", session_key="K"),
    dict(source="telegram", session_key="R"),
    dict(source="telegram", session_key="none", chat_id="tc", chat_type="dm", user_id="tu"),
    dict(source="telegram", session_key="none", chat_id="tc", user_id="tu"),
    dict(source="telegram", session_key="", chat_id="tc", chat_type="dm", user_id="tu"),
    dict(source="telegram", session_key=None),
)


def _json_column(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":")) if value else None


# Record field -> (legacy column, value converter); None drops the field. Unlisted fields keep their name,
# with booleans read back as the stored 0/1.
_LEGACY_COLUMNS: dict[str, Any] = {
    "lineage_kind": None,
    "lineage_origin_id": None,
    # Derived from the cells, or the stored cells the decoded fields above already map back.
    "model_config_corrupt": None,
    "model_config_json": None,
    "origin_json": None,
    "tool_names": None,
    "session_id": ("id", lambda value: value),
    "origin": ("origin_json", _json_column),
    "model_config": ("model_config", _json_column),
    "tool_pin": ("tool_names", _json_column),
    "reset_from": ("_reset_from", lambda value: value),
    "branched_from": ("_branched_from", lambda value: value),
    "title_source": ("title_source", lambda value: value.value if value is not None else None),
}


def _legacy_view(found) -> Any:
    """A recovered peer as the legacy row dict's values: the record's fields mapped back to their columns."""
    if found is None:
        return None
    record, row = found.session, {"_has_messages": int(found.has_messages)}
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        mapping = _LEGACY_COLUMNS.get(field.name, (field.name, lambda v: int(v) if isinstance(v, bool) else v))
        if mapping is None or (field.name in ("reset_from", "branched_from") and value is None):
            continue
        column, convert = mapping
        row[column] = convert(value)
    return row


def _normal(row: dict | None) -> Any:
    """A legacy row dict with JSON columns re-encoded compactly and NULL counters read as zero."""
    if row is None:
        return None
    out = dict(row)
    for column in ("origin_json", "model_config", "tool_names"):
        out[column] = json.dumps(json.loads(out[column]), separators=(",", ":")) if out[column] else None
    return out


def test_latest_session_for_peer_reads_the_legacy_row(backend, scope_a):
    before, after = run_both(
        backend, scope_a, _seed_peers,
        lambda db: [_normal(db.find_latest_gateway_session_for_peer(**call)) for call in _PEER_CALLS],
        lambda routing: [routing.latest_session_for_peer(**call) for call in _PEER_CALLS])
    assert_same(before, after, lambda found: [_legacy_view(f) for f in found])
    picked = [f.session.session_id if f else None for f in after[1]]
    assert picked == ["k-msg", "k-msg", None, None, "t-mine", None, None, None]


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


_RECORD_CALLS = (
    ("new", dict(source="telegram", session_key="agent:main:telegram:dm:5", chat_id="5", chat_type="dm", user_id="u",
                 display_name="N", origin_json='{"a": 1}', transport_profile="tp")),
    ("k-empty", dict(source="telegram", session_key="K2", chat_id=None, user_id="v")),
    ("k-msg", dict(source="slack", session_key="K3", display_name="kept?", transport_profile=None)),
    ("x", dict(source="telegram", session_key="")),
    ("", dict(source="telegram", session_key="K")),
    ("tip", dict(source="telegram", session_key="L2", chat_id="lc", include_compression_ancestors=True)),
    ("ghost", dict(source="telegram", session_key="G", include_compression_ancestors=True)),
)


def test_record_session_peer_writes_the_legacy_rows(backend, scope_a):
    before, after = run_both(
        backend, scope_a, _seed_peers,
        lambda db: [db.record_gateway_session_peer(sid, **kw) for sid, kw in _RECORD_CALLS],
        lambda routing: [routing.record_session_peer(sid, **kw, messaging_lane=_messaging_platform(kw["source"]) is not None)
                         for sid, kw in _RECORD_CALLS])
    assert_same(before, after)


def _seed_origin(db) -> None:
    _keyed(db, "o1", "k1", source="Telegram", chat_id="c", user_id="a", started=NOW - 30)
    _keyed(db, "o2", "k2", chat_id="c", user_id=" b ", thread_id="t", started=NOW - 20)
    _keyed(db, "o3", "k3", chat_id="c", user_id="a", started=NOW - 10, ended_at=NOW - 5)
    _keyed(db, "o6", "k6", chat_id="c", user_id="a", started=NOW - 15)
    _keyed(db, "o4", None, chat_id="c", started=NOW - 1)
    _keyed(db, "o5", "k5", source="TELEGRAMİ", chat_id="c")


_ORIGIN_CALLS = (
    dict(platform="telegram", chat_id="c"),
    dict(platform="TELEGRAM", chat_id="c", user_id="a"),
    dict(platform="telegram", chat_id="c", user_id="zz"),
    dict(platform="telegram", chat_id="c", thread_id="t"),
    dict(platform="telegram", chat_id="c", thread_id=""),
    dict(platform="telegram", chat_id=""),
    dict(platform="", chat_id="c"),
    dict(platform="telegrami", chat_id="c"),
)


def test_session_for_origin_matches_find_session_by_origin(backend, scope_a):
    before, after = run_both(
        backend, scope_a, _seed_origin,
        lambda db: [db.find_session_by_origin(**call) for call in _ORIGIN_CALLS],
        lambda routing: [routing.session_for_origin(**call) for call in _ORIGIN_CALLS])
    assert_same(before, after)
    assert after[1][1] == "o6"


# ── orphan repair and never-active pruning ──────────────────────────────────


def _seed_orphans(db) -> None:
    """A lineage orphan of a keyed parent, a lineage orphan of an unkeyed parent, a contiguity orphan with one donor,
    two orphans contesting one donor, a sibling profile's donor and excluded tool and fork rows."""
    _keyed(db, "d1", "agent:main:telegram:dm:1", user_id="u", chat_id="1", origin_json='{"o": 1}', display_name="D1",
           last_activity_at=NOW - 500)
    _keyed(db, "o-lin", None, parent_session_id="d1", started=NOW - 400, user_id="u")
    _keyed(db, "p-bare", None, started=NOW - 2000)
    _keyed(db, "o-bare", None, parent_session_id="p-bare", started=NOW - 390)
    _keyed(db, "d2", "agent:main:telegram:dm:2", user_id="w", last_activity_at=NOW - 300, started=NOW - 3000)
    _keyed(db, "o-con", None, user_id="w", started=NOW - 250, chat_id="keep")
    _keyed(db, "d3", "agent:main:telegram:dm:3", user_id="x", last_activity_at=NOW - 100, started=NOW - 3000,
           profile_name="other")
    _keyed(db, "o-sib", None, user_id="x", started=NOW - 90)
    _keyed(db, "o-tool", None, source="tool", started=NOW - 80)
    _keyed(db, "o-fork", None, started=NOW - 70, model_config=json.dumps({"_branched_from": "d1"}))
    for sid in ("o-lin", "o-bare", "o-con", "o-sib", "o-tool", "o-fork"):
        db.append_message(sid, "user", "m", timestamp=NOW - 60)
    db.append_message("o-con", "assistant", "r", timestamp=NOW - 50)


def test_orphaned_sessions_match_the_legacy_report(backend, scope_a):
    calls = (None, 10.0, 5000.0)
    before, after = run_both(
        backend, scope_a, _seed_orphans,
        lambda db: [db.find_orphaned_gateway_sessions(max_gap_s=gap) for gap in calls],
        lambda routing: [routing.orphaned_sessions(max_gap_s=gap) for gap in calls])
    assert_same(before, after, lambda reports: [[dataclasses.asdict(r) for r in batch] for batch in reports])
    assert {r.orphan_id: r.adoptable for r in after[1][0]} == {
        "o-lin": True, "o-bare": False, "o-con": True, "o-sib": False}


_ADOPTIONS = (("o-lin", "d1"), ("o-lin", "d1"), ("", "d1"), ("d1", "d1"), ("o-con", "missing"), ("o-sib", "d3"),
              ("o-con", "d2"), ("o-bare", "o-con"))


def test_adopt_orphan_writes_the_legacy_rows(backend, scope_a):
    before, after = run_both(
        backend, scope_a, _seed_orphans,
        lambda db: [db.adopt_orphaned_gateway_session(o, d) for o, d in _ADOPTIONS],
        lambda routing: [routing.adopt_orphan(o, d) for o, d in _ADOPTIONS])
    assert_same(before, after)
    assert after[1] == [True, False, False, False, False, False, True, True]


def _seed_never_active(db) -> None:
    """Silent keyed rows (two old, one fresh), rows with a title, a pin, a message or a token count, an unkeyed and
    an ended row; routing entries naming the old rows in two scopes; a live turn lease on one old row."""
    for sid, started, cols in (
        ("n-old", NOW - 10 * DAY, {}), ("n-lease", NOW - 9 * DAY, {}), ("n-fresh", NOW - DAY, {}),
        ("n-title", NOW - 10 * DAY, {"title": "t"}), ("n-pin", NOW - 10 * DAY, {"pinned": 1}),
        ("n-tok", NOW - 10 * DAY, {"input_tokens": 3}), ("n-ended", NOW - 10 * DAY, {"ended_at": NOW - DAY}),
        ("n-msg", NOW - 10 * DAY, {}),
    ):
        _keyed(db, sid, "agent:main:telegram:dm:" + sid, started=started, **cols)
    db.append_message("n-msg", "user", "hi", timestamp=NOW - 10 * DAY)
    _keyed(db, "n-unkeyed", None, started=NOW - 10 * DAY)
    for scope in ("", "r"):
        for sid in ("n-old", "n-lease", "n-msg"):
            db.save_gateway_routing_entry(f"agent:main:{scope}:{sid}", _entry(sid), scope=scope)
    assert db.acquire_session_turn_lease("n-lease", "live", ttl_seconds=3600.0, wait_seconds=0.0)


def test_never_active_keyed_sessions_match_the_legacy_listing(backend, scope_a):
    before, after = run_both(
        backend, scope_a, _seed_never_active,
        lambda db: [db.list_never_active_keyed_sessions(older_than_days=d) for d in (5, 0.5, 0)],
        lambda routing: [routing.never_active_keyed_sessions(older_than_days=d) for d in (5, 0.5, 0)])
    def project(batches):
        return [[{"id": r.session_id, **{k: v for k, v in dataclasses.asdict(r).items() if k != "session_id"}}
                 for r in batch] for batch in batches]
    assert_same(before, after, project)
    assert [r.session_id for r in after[1][0]] == ["n-old", "n-lease"]


@pytest.mark.parametrize("operation", ["list", "prune"])
def test_a_negative_retention_raises_the_legacy_value_error(backend, scope_a, operation):
    legacy, verb = {
        "list": (lambda db: db.list_never_active_keyed_sessions(older_than_days=-1),
                 lambda r: r.never_active_keyed_sessions(older_than_days=-1)),
        "prune": (lambda db: db.prune_never_active_keyed_sessions(older_than_days=-1),
                  lambda r: r.prune_never_active(older_than_days=-1, media_root=MEDIA_ROOT)),
    }[operation]
    before, after = run_both(backend, scope_a, _seed_never_active, legacy, verb)
    assert before[0] == "error" and isinstance(before[1], ValueError)
    assert_same(before, after)


def _plant(directory: Path, *session_ids: str) -> set[str]:
    directory.mkdir(parents=True, exist_ok=True)
    for sid in session_ids:
        for name in (f"{sid}.json", f"{sid}.jsonl", f"session_{sid}.json", f"request_dump_{sid}_1.json"):
            (directory / name).write_text("{}")
    return {p.name for p in directory.iterdir()}


@pytest.mark.parametrize("with_artifacts", [True, False])
def test_prune_never_active_runs_the_legacy_sequence(backend, scope_a, tmp_path, with_artifacts):
    ids = ("n-old", "n-lease", "n-fresh", "n-msg")
    legacy_dir, backend_dir = tmp_path / "legacy", tmp_path / "backend"
    _plant(legacy_dir, *ids)
    _plant(backend_dir, *ids)

    def legacy(db):
        deleted_ids: list = []
        counts = db.prune_never_active_keyed_sessions(
            older_than_days=5, sessions_dir=legacy_dir if with_artifacts else None, deleted_ids=deleted_ids)
        return (*counts, tuple(deleted_ids))

    def verb(routing):
        return routing.prune_never_active(
            older_than_days=5, artifacts=SessionArtifacts(backend_dir) if with_artifacts else None,
            media_root=MEDIA_ROOT)
    before, after = run_both(backend, scope_a, _seed_never_active, legacy, verb)
    assert_same(before, after, lambda prune: dataclasses.astuple(prune)[:4])
    assert after[1].deleted_ids == after[1].removed_ids == ("n-old",) and after[1].sessions_skipped == 1
    assert {p.name for p in backend_dir.iterdir()} == {p.name for p in legacy_dir.iterdir()}
    assert ("n-old.jsonl" in {p.name for p in backend_dir.iterdir()}) is not with_artifacts


# ── Telegram DM topics ──────────────────────────────────────────────────────

_V1_TABLES = (
    "CREATE TABLE telegram_dm_topic_mode (chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, enabled INTEGER NOT NULL "
    "DEFAULT 1, activated_at REAL NOT NULL, updated_at REAL NOT NULL, has_topics_enabled INTEGER, "
    "allows_users_to_create_topics INTEGER, capability_checked_at REAL, intro_message_id TEXT, pinned_message_id TEXT)",
    "CREATE TABLE telegram_dm_topic_bindings (chat_id TEXT NOT NULL, thread_id TEXT NOT NULL, user_id TEXT NOT NULL, "
    "session_key TEXT NOT NULL, session_id TEXT NOT NULL, managed_mode TEXT NOT NULL DEFAULT 'auto', "
    "linked_at REAL NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (chat_id, thread_id))",
    "INSERT INTO telegram_dm_topic_mode VALUES ('c1', 'u1', 1, 1.0, 2.0, 1, 0, 3.0, NULL, NULL)",
    "INSERT INTO telegram_dm_topic_bindings VALUES ('c1', 't1', 'u1', 'k1', 's1', 'auto', 1.0, 2.0)",
    "INSERT INTO telegram_dm_topic_bindings VALUES ('c1', 't2', 'u1', 'kg', 'ghost', 'auto', 1.0, 2.0)",
)


def _seed_sessions(db) -> None:
    for sid in ("s1", "s2", "s3"):
        db.create_session(sid, "telegram")


def _seed_v1_topics(db) -> None:
    _seed_sessions(db)
    for sql in _V1_TABLES:
        _set(db, sql)


_TOPIC_READS = (
    ("is_telegram_topic_mode_enabled", "is_topic_mode_enabled", dict(chat_id="c1", user_id="u1")),
    ("is_telegram_topic_mode_enabled", "is_topic_mode_enabled", dict(chat_id="c1", user_id="u1", profile_name="work")),
    ("get_telegram_topic_binding", "get_topic_binding", dict(chat_id="c1", thread_id="t1")),
    ("list_telegram_topic_bindings_for_chat", "list_topic_bindings_for_chat", dict(chat_id="c1")),
    ("list_telegram_topic_bindings_for_chat", "list_topic_bindings_for_chat", dict(chat_id="c1", profile_name=" ")),
    ("get_telegram_topic_binding_by_session", "get_topic_binding_for_session", dict(session_id="s1")),
    ("get_telegram_topic_binding_by_session", "get_topic_binding_for_session", dict(session_id="s2")),
    ("is_telegram_session_linked_to_topic", "is_session_linked_to_topic", dict(session_id="s1")),
    ("is_telegram_session_linked_to_topic", "is_session_linked_to_topic", dict(session_id="ghost")),
)


def _binding_view(value: Any) -> Any:
    """Topic records as the legacy row dicts."""
    if isinstance(value, tuple):
        return [_binding_view(v) for v in value]
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def _read_all_topics(db_or_routing: Any, legacy: bool) -> list:
    return [_outcome(lambda name=(old if legacy else new), kw=kw: getattr(db_or_routing, name)(**kw))
            for old, new, kw in _TOPIC_READS]


@pytest.mark.parametrize("seed", [_seed_sessions, _seed_v1_topics], ids=["no-tables", "v1-tables"])
def test_topic_reads_answer_empty_without_tables_and_heal_pre_v3_tables(backend, scope_a, seed):
    before, after = run_both(backend, scope_a, seed, lambda db: _read_all_topics(db, True),
                             lambda routing: _read_all_topics(routing, False))
    assert before[0] == after[0] == "ok"
    for legacy_read, read in zip(before[1], after[1]):
        assert_same(legacy_read, read, _binding_view)


def _topic_writes(target: Any, legacy: bool) -> list:
    """One fixed sequence of topic writes, then every read; each step's outcome kept."""
    def call(old: str, new: str, **kw: Any) -> tuple[str, Any]:
        return _outcome(lambda: getattr(target, old if legacy else new)(**kw))
    steps = [
        call("apply_telegram_topic_migration", "prepare_topic_storage"),
        call("enable_telegram_topic_mode", "enable_topic_mode", chat_id="c1", user_id="u1"),
        call("enable_telegram_topic_mode", "enable_topic_mode", chat_id="c1", user_id="u2", profile_name=" work ",
             has_topics_enabled=True, allows_users_to_create_topics=False),
        call("bind_telegram_topic", "bind_topic", chat_id="c1", thread_id="t1", user_id="u1", session_key="k1",
             session_id="s1"),
        call("bind_telegram_topic", "bind_topic", chat_id="c1", thread_id="t1", user_id="u1", session_key="k1b",
             session_id="s1", managed_mode="restored"),
        call("bind_telegram_topic", "bind_topic", chat_id="c1", thread_id="t2", user_id="u1", session_key="k1",
             session_id="s1"),
        call("bind_telegram_topic", "bind_topic", chat_id="c1", thread_id="t1", user_id="u1", session_key="k2",
             session_id="s2", profile_name="work"),
        call("bind_telegram_topic", "bind_topic", chat_id="c1", thread_id="t3", user_id="u1", session_key="k3",
             session_id="s3"),
        call("delete_telegram_topic_binding", "delete_topic_binding", chat_id="c1", thread_id="t9"),
        *_read_all_topics(target, legacy),
        call("delete_telegram_topic_binding", "delete_topic_binding", chat_id="c1", thread_id="t3"),
        *_read_all_topics(target, legacy),
        call("delete_telegram_topic_binding", "delete_topic_binding", chat_id="c1", thread_id="t1"),
        call("disable_telegram_topic_mode", "disable_topic_mode", chat_id="c1", profile_name="work",
             clear_bindings=False),
        *_read_all_topics(target, legacy),
        call("disable_telegram_topic_mode", "disable_topic_mode", chat_id="c1", profile_name="work"),
        *_read_all_topics(target, legacy),
    ]
    return steps


@pytest.mark.parametrize("seed", [_seed_sessions, _seed_v1_topics], ids=["no-tables", "v1-tables"])
def test_topic_writes_match_the_legacy_sequence(backend, scope_a, clock, seed):
    before, after = run_both(backend, scope_a, seed, lambda db: _topic_writes(db, True),
                             lambda routing: _topic_writes(routing, False))
    for legacy_step, step in zip(before[1], after[1], strict=True):
        assert_same(legacy_step, step, _binding_view)
    assert isinstance(after[1][5][1], ValueError) and "already linked" in str(after[1][5][1])


def test_bind_topic_to_a_missing_session_raises_session_row_missing(backend, scope_a):
    before, after = run_both(
        backend, scope_a, _seed_sessions,
        lambda db: db.bind_telegram_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="nope"),
        lambda routing: routing.bind_topic(chat_id="c", thread_id="t", user_id="u", session_key="k", session_id="nope"))
    assert isinstance(before[1], sqlite3.IntegrityError) and "FOREIGN KEY" in str(before[1])
    assert isinstance(after[1], SessionRowMissing) and isinstance(after[1].__cause__, sqlite3.IntegrityError)
    assert after[1].entity_id == "nope"

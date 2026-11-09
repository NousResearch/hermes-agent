"""Group chat counts in the opt-in daily install snapshot (``hermes.install.snapshot``).

* A gateway keeps the group chats it hosts in its profile's own ``state.db``, so every profile's snapshot
  counts those; the root profile also counts standalone ones (``hermes serve``) in the root
  ``shared-state.db`` that were not yet handed to a gateway. Each group chat is counted once.
* Four bucketed counts, nothing else: group chats that exist (not disbanded), ones with a message posted in
  the last 7 days, ones created in the last 7 days, and ones with a member on another gateway.
* Read-only: a missing store reads zero and is never created, a store no process has open gains no
  ``-wal``/``-shm`` files, and a home path with URI characters reads the right file; an unreadable store
  leaves the fields out rather than reporting a false zero, and never costs the rest of the snapshot.
* Snapshots counted before the fields existed still package against the v4 schema.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from gateway import hosted_rooms
from gateway.hosted_rooms_gateway_import import handoff_gateway_id
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_snapshot import collect_install_snapshot

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "hermes_cli" / "observability" / "schemas"
    / "hermes.shared_metrics.v4.schema.json"
)
RESOURCE = {"architecture": "x86_64", "hermes_version": "test", "install_method": "git", "os_family": "linux"}
FIELDS = (
    "group_chat_count_bucket", "group_chat_active_7d_count_bucket",
    "group_chat_created_7d_count_bucket", "group_chat_peer_count_bucket",
)
DAY = 24 * 3600


def _room(conn, room_id, *, created_days_ago, disbanded_days_ago=None, authority="gw"):
    now = time.time()
    conn.execute(
        "INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, created_at, updated_at,"
        " disbanded_at) VALUES (?, ?, '[]', ?, ?, ?, ?)",
        (room_id, room_id, authority, now - created_days_ago * DAY, now,
         None if disbanded_days_ago is None else now - disbanded_days_ago * DAY),
    )


def _event(conn, room_id, seq, kind, *, days_ago):
    conn.execute(
        "INSERT INTO hosted_room_events (room_id, seq, event_id, kind, actor_json, payload_json, created_at)"
        " VALUES (?, ?, ?, ?, '{}', '{}', ?)",
        (room_id, seq, f"{room_id}-{seq}", kind, time.time() - days_ago * DAY),
    )


def _peer_link(conn, room_id):
    conn.execute(
        "INSERT INTO hosted_room_links (room_id, member_id, target_url, target_profile, grant, catalog_json,"
        " cancellation_scope_id, trace_id, transport_security, updated_at)"
        " VALUES (?, 'peer-bot', 'https://peer.example', 'default', 'g', '{}', 's', 't', 'tls', ?)",
        (room_id, time.time()),
    )


def _close_cleanly(path: Path) -> None:
    assert not any(Path(f"{path}{s}").exists() for s in ("-wal", "-shm")), "the last close removes them"


def _seed_gateway_store(path: Path) -> None:
    """Seven live group chats a gateway hosts (plus one disbanded), in the real schema."""
    hosted_rooms._connect(path).close()
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        _room(conn, "active-old", created_days_ago=30)
        _event(conn, "active-old", 1, "message.user", days_ago=2)
        _room(conn, "active-new", created_days_ago=1)
        _event(conn, "active-new", 1, "message.user", days_ago=0.05)
        _room(conn, "quiet-peer", created_days_ago=20)  # last message outside the window
        _event(conn, "quiet-peer", 1, "message.user", days_ago=10)
        _peer_link(conn, "quiet-peer")
        _room(conn, "bots-only", created_days_ago=40)  # a member reply alone is not a posted message
        _event(conn, "bots-only", 1, "message.member", days_ago=1)
        _peer_link(conn, "bots-only")
        _room(conn, "idle-peer", created_days_ago=60)
        _peer_link(conn, "idle-peer")
        _room(conn, "idle-1", created_days_ago=90)
        _room(conn, "idle-2", created_days_ago=8)
        _room(conn, "disbanded", created_days_ago=5, disbanded_days_ago=3)  # counted nowhere
        _event(conn, "disbanded", 1, "message.user", days_ago=4)
        _peer_link(conn, "disbanded")
    _close_cleanly(path)


GATEWAY_COUNTS = {
    "group_chat_count_bucket": "6_to_10",  # 7 live; the disbanded one is gone
    "group_chat_active_7d_count_bucket": "2",  # active-old, active-new
    "group_chat_created_7d_count_bucket": "1",  # active-new
    "group_chat_peer_count_bucket": "3_to_5",  # quiet-peer, bots-only, idle-peer
}


def _seed_standalone_store(path: Path) -> None:
    """Standalone (``hermes serve``) group chats: two still its own, one already handed to the gateway."""
    hosted_rooms._connect(path).close()
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        _room(conn, "standalone-new", created_days_ago=2)
        _event(conn, "standalone-new", 1, "message.user", days_ago=1)
        _room(conn, "standalone-old", created_days_ago=50)
        _room(conn, "handed-off", created_days_ago=3, authority=handoff_gateway_id())  # now the gateway's
        _event(conn, "handed-off", 1, "message.user", days_ago=1)
    _close_cleanly(path)


def _group_chat_fields(snapshot):
    return {k: snapshot[k] for k in FIELDS if k in snapshot}


@pytest.fixture
def root_home(tmp_path, monkeypatch):
    home = tmp_path / "root"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_a_profile_counts_the_group_chats_its_gateway_hosts(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    _seed_gateway_store(profile / "state.db")
    _seed_standalone_store(root / "shared-state.db")  # the root profile's to count, not this one's
    monkeypatch.setenv("HERMES_HOME", str(profile))

    snapshot = collect_install_snapshot({})

    assert _group_chat_fields(snapshot) == GATEWAY_COUNTS
    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, snapshot), snapshot


def test_root_adds_standalone_group_chats_not_yet_handed_to_its_gateway(root_home):
    _seed_gateway_store(root_home / "state.db")
    # As on an upgraded install, the first open copies the root state.db's group chats in here as well.
    _seed_standalone_store(root_home / "shared-state.db")
    with contextlib.closing(sqlite3.connect(root_home / "shared-state.db")) as conn, conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms WHERE room_id = 'active-old'").fetchone() == (1,)
        # One of those was later disbanded there and pruned, leaving only its retired id.
        conn.execute("DELETE FROM hosted_room_links WHERE room_id = 'idle-peer'")
        conn.execute("DELETE FROM hosted_rooms WHERE room_id = 'idle-peer'")
        conn.execute("INSERT INTO hosted_room_retired_ids (room_id, retired_at) VALUES ('idle-peer', 0)")

    snapshot = collect_install_snapshot({})

    assert _group_chat_fields(snapshot) == {
        "group_chat_count_bucket": "6_to_10",  # 6 + 2, each once; the handed-off copy is the gateway's now
        "group_chat_active_7d_count_bucket": "3_to_5",  # 2 + standalone-new
        "group_chat_created_7d_count_bucket": "2",  # 1 + standalone-new
        "group_chat_peer_count_bucket": "2",  # quiet-peer, bots-only; idle-peer is retired
    }


def test_missing_stores_read_zero_without_creating_them(root_home):
    snapshot = collect_install_snapshot({})

    assert _group_chat_fields(snapshot) == dict.fromkeys(FIELDS, "0")
    stores = [n for n in os.listdir(root_home) if n.startswith(("state.db", "shared-state.db"))]
    assert stores == [], "telemetry must never create a store"


def test_a_store_without_group_chats_reads_zero(root_home):
    with sqlite3.connect(root_home / "shared-state.db") as conn:  # another feature's tables only
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    assert _group_chat_fields(collect_install_snapshot({})) == dict.fromkeys(FIELDS, "0")


def test_reading_a_closed_store_leaves_no_wal_or_shm_files(root_home):
    _seed_standalone_store(root_home / "shared-state.db")
    before = sorted(os.listdir(root_home))

    assert _group_chat_fields(collect_install_snapshot({}))["group_chat_count_bucket"] == "2"
    assert sorted(os.listdir(root_home)) == before


@pytest.mark.parametrize("name", ["home#1", "home?x=1", "home%41"])
def test_a_home_path_with_uri_characters_reads_its_own_store(tmp_path, monkeypatch, name):
    home = tmp_path / name
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_standalone_store(home / "shared-state.db")
    beside = sorted(os.listdir(tmp_path))

    assert _group_chat_fields(collect_install_snapshot({}))["group_chat_count_bucket"] == "2"
    assert sorted(os.listdir(tmp_path)) == beside, "no stray file next to the home"


def test_unreadable_store_leaves_the_fields_out_instead_of_reporting_zero(root_home):
    (root_home / "shared-state.db").write_bytes(b"not a sqlite database" * 64)

    snapshot = collect_install_snapshot({})

    assert _group_chat_fields(snapshot) == {}
    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, snapshot), snapshot


def test_any_failure_reading_group_chats_keeps_the_rest_of_the_snapshot(root_home, monkeypatch):
    def broken():
        raise RuntimeError("symlink loop")

    monkeypatch.setattr(hosted_rooms, "default_db_path", broken)

    snapshot = collect_install_snapshot({})

    assert _group_chat_fields(snapshot) == {}
    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, snapshot), snapshot


def test_snapshots_with_and_without_group_chat_fields_package_against_the_v4_schema(root_home, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    _seed_gateway_store(root_home / "state.db")
    current = collect_install_snapshot({})
    assert set(FIELDS) <= set(current)

    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    store.record_counter(contract.INSTALL_SNAPSHOT_METRIC, current, RESOURCE)
    without = {k: v for k, v in current.items() if k not in FIELDS}  # an unreadable store
    store.record_counter(contract.INSTALL_SNAPSHOT_METRIC, without, RESOURCE)
    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8-sig"))).validate(package)
    snapshots = [m["dimensions"] for m in package["metrics"] if m["name"] == contract.INSTALL_SNAPSHOT_METRIC]
    assert sorted(d.get("group_chat_count_bucket", "absent") for d in snapshots) == ["6_to_10", "absent"]


def test_v4_schema_accepts_exactly_the_contract_values_for_the_group_chat_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8-sig"))
    dimensions = schema["$defs"]["install_snapshot_counter"]["properties"]["dimensions"]
    for field in FIELDS:
        assert field not in dimensions["required"], field
        definition = dimensions["properties"][field]
        if "$ref" in definition:
            definition = schema["$defs"][definition["$ref"].rsplit("/", 1)[-1]]
        assert set(definition["enum"]) == contract.SIZE_BUCKETS, field

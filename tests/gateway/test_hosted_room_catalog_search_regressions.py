"""Catalog search must describe exactly the files and sharers it returns."""

import json
import sqlite3
import unicodedata

import pytest

from tests.gateway.test_hosted_room_attachment_catalog import (
    _create_catalog,
    _seed_events,
)
from tests.gateway.test_hosted_room_attachment_catalog_v2 import listing, share


@pytest.mark.parametrize("field", ["attachments", "att\\u0061chments", "member_id"])
def test_projected_payload_rejects_ambiguous_duplicate_fields(tmp_path, field):
    db, store = _create_catalog(tmp_path)
    _item, event_id, _data = share(db, store)
    with sqlite3.connect(db) as conn:
        raw = conn.execute(
            "SELECT payload_json FROM hosted_room_events WHERE event_id=?", (event_id,)
        ).fetchone()[0]
        duplicate = (
            json.dumps(json.dumps(json.loads(raw)["attachments"]))
            if field != "member_id"
            else '"other"'
        )
        raw = raw[:-1] + ',"' + field + '":' + duplicate + "}"
        conn.execute(
            "UPDATE hosted_room_events SET payload_json=? WHERE event_id=?",
            (raw, event_id),
        )
    assert listing(store)["items"] == []
    assert listing(store)["latest_seq"] == 0


@pytest.mark.parametrize(
    "name,query", [("x\u037a.txt", "\u03b9"), ("\U0001d400lpha.txt", "alpha")]
)
def test_uploads_from_lossy_legacy_normalizer_search_by_canonical_name(
    tmp_path, monkeypatch, name, query
):
    from gateway import hosted_room_attachments

    db, store = _create_catalog(tmp_path)

    def legacy_fold(value):
        return "".join(
            char
            for char in unicodedata.normalize("NFKD", value.casefold())
            if not unicodedata.combining(char)
        )

    with monkeypatch.context() as old:
        old.setattr(hosted_room_attachments, "fold_catalog_text", legacy_fold)
        item, _event, _data = share(db, store, name=name)
    before = db.read_bytes()
    assert [row["attachment_id"] for row in listing(store, query=query)["items"]] == [
        item["attachment_id"]
    ]
    assert db.read_bytes() == before


def test_projected_manifest_retains_its_original_json_type(tmp_path):
    db, store = _create_catalog(tmp_path)
    _item, event_id, _data = share(db, store)
    assert len(listing(store)["items"]) == 1
    with sqlite3.connect(db) as conn:
        payload = json.loads(
            conn.execute(
                "SELECT payload_json FROM hosted_room_events WHERE event_id=?",
                (event_id,),
            ).fetchone()[0]
        )
        payload["attachments"] = json.dumps(payload["attachments"])
        conn.execute(
            "UPDATE hosted_room_events SET payload_json=? WHERE event_id=?",
            (json.dumps(payload), event_id),
        )
    page = listing(store)
    assert page["items"] == []
    assert page["latest_seq"] == 0


@pytest.mark.parametrize(
    "actor,label",
    [
        ({"kind": "user", "id": "desktop"}, "You"),
        ({"kind": "user", "id": "desktop", "display_name": "Alice"}, "Alice"),
        ({"kind": "member", "id": "qa", "profile": "qa"}, "Quality"),
        ({"kind": "member", "id": "retired", "profile": "retired"}, "retired"),
    ],
)
def test_search_matches_the_effective_sharer_label(tmp_path, actor, label):
    db, store = _create_catalog(tmp_path)
    _seed_events(db, total_events=1, records={1: ["memo.txt"]})
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_events SET kind=?, actor_json=?",
            (
                "message.user" if actor["kind"] == "user" else "message.member",
                json.dumps(actor),
            ),
        )
    plain = listing(store)
    assert plain["items"][0]["producer"]["label"] == label
    assert listing(store, query=label.lower())["items"] == plain["items"]
    if label == "Alice":
        assert listing(store, query="you")["items"] == []


def test_actor_label_override_does_not_create_empty_search_pages(tmp_path):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db,
        total_events=300,
        records={i: [f"memo-{i}.txt"] for i in range(1, 301)},
        producers={i: "qa" for i in range(1, 301)},
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_events SET actor_json=json_set(actor_json, '$.display_name', 'Writer') WHERE seq>1"
        )
    page = listing(store, query="quality")
    assert [item["seq"] for item in page["items"]] == [1]
    assert not page["has_more"]
    assert page["latest_seq"] == 1


@pytest.mark.parametrize("cached", [None, "Alpha.txt"])
def test_search_reads_legacy_compatibility_capitals_without_rewriting(tmp_path, cached):
    db, store = _create_catalog(tmp_path)
    item, _event_id, _data = share(db, store, name="\U0001d400lpha.txt")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_attachments SET catalog_name=?", (cached,))
    before = db.read_bytes()
    assert [row["attachment_id"] for row in listing(store, query="alpha")["items"]] == [
        item["attachment_id"]
    ]
    assert db.read_bytes() == before


def test_new_uploads_fold_compatibility_capitals(tmp_path):
    db, store = _create_catalog(tmp_path)
    item, _event, _data = share(db, store, name="\U0001d400lpha.txt")
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT catalog_name FROM hosted_room_attachments WHERE attachment_id=?",
                (item["attachment_id"],),
            ).fetchone()[0]
            == "alpha.txt"
        )
    assert [row["attachment_id"] for row in listing(store, query="ALPHA")["items"]] == [
        item["attachment_id"]
    ]


@pytest.mark.parametrize("premise", ["actor-override", "invalid-matching-sibling"])
def test_latest_marker_only_counts_eligible_matching_items(tmp_path, premise):
    db, store = _create_catalog(tmp_path)
    if premise == "actor-override":
        _seed_events(
            db,
            total_events=2,
            records={1: ["one.txt"], 2: ["two.txt"]},
            producers={1: "qa", 2: "qa"},
        )
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE hosted_room_events SET actor_json=json_set(actor_json, '$.display_name', 'Writer') WHERE seq=2"
            )
        query = "quality"
    else:
        _seed_events(
            db,
            total_events=2,
            records={1: ["needle.txt"], 2: ["needle.txt", "other.txt"]},
        )
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE hosted_room_attachments SET sha256=? WHERE upload_id='upload-2-0'",
                ("0" * 64,),
            )
        query = "needle"
    page = listing(store, query=query)
    assert [row["seq"] for row in page["items"]] == [1]
    assert page["latest_seq"] == 1

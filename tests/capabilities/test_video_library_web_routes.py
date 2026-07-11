import pytest

from capabilities.video_library.service import VideoLibraryService
from capabilities.video_library.store import VideoLibraryStore


@pytest.fixture
def clients(tmp_path, monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    from capabilities.video_library import adapter
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    service = VideoLibraryService(VideoLibraryStore(root=tmp_path / "library"))
    monkeypatch.setattr(adapter, "get_service", lambda: service)
    monkeypatch.setattr(app.state, "auth_required", False, raising=False)
    anonymous = TestClient(app)
    authenticated = TestClient(app)
    authenticated.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return anonymous, authenticated, service


def test_video_library_asset_routes_require_session_and_import(clients, tmp_path):
    anonymous, authenticated, _service = clients
    source = tmp_path / "merchant.mp4"
    source.write_bytes(b"video")
    path = "/api/capabilities/video-library/assets"

    assert anonymous.post(path, json={"sourcePath": str(source)}).status_code == 401

    response = authenticated.post(path, json={"sourcePath": str(source)})
    listed = authenticated.get(path)

    assert response.status_code == 200
    assert response.json()["data"]["asset"]["original_name"] == "merchant.mp4"
    assert listed.status_code == 200
    assert listed.json()["data"]["total"] == 1


def test_video_library_analyze_and_timeline_routes(clients, monkeypatch):
    _anonymous, authenticated, service = clients

    monkeypatch.setattr(
        service,
        "analyze_asset",
        lambda asset_id, **_kwargs: {
            "asset": {"id": asset_id},
            "clips": [{"id": "clip-1"}],
            "job": {"id": "job-1", "state": "complete"},
        },
    )
    monkeypatch.setattr(
        service,
        "create_timeline",
        lambda clip_ids, **_kwargs: {
            "id": "timeline-1",
            "path": "/tmp/timeline-1.json",
            "timeline": {"version": 1, "clips": clip_ids},
        },
    )

    analyzed = authenticated.post(
        "/api/capabilities/video-library/assets/asset-1/analyze",
        json={"threshold": 0.4},
    )
    timeline = authenticated.post(
        "/api/capabilities/video-library/timelines",
        json={"aspect": "9:16", "clipIds": ["clip-1"]},
    )

    assert analyzed.status_code == 200
    assert analyzed.json()["data"]["job"]["state"] == "complete"
    assert timeline.status_code == 200
    assert timeline.json()["data"]["timeline"]["version"] == 1


def test_named_library_list_and_scan_routes(clients, monkeypatch):
    _anonymous, authenticated, _service = clients
    from capabilities.video_library import adapter

    monkeypatch.setattr(
        adapter,
        "list_libraries",
        lambda: {"libraries": [{"id": "beef-noodle", "name": "牛肉面资产库"}]},
    )
    monkeypatch.setattr(
        adapter,
        "scan_library",
        lambda library_id, dry_run=False: {
            "complete": 2,
            "dry_run": dry_run,
            "library_id": library_id,
        },
    )

    listed = authenticated.get("/api/capabilities/video-library/libraries")
    scanned = authenticated.post(
        "/api/capabilities/video-library/libraries/beef-noodle/scan",
        json={"dryRun": True},
    )

    assert listed.status_code == 200
    assert listed.json()["data"]["libraries"][0]["id"] == "beef-noodle"
    assert scanned.status_code == 200
    assert scanned.json()["data"]["dry_run"] is True


def test_named_library_management_routes_keep_target_library(clients, monkeypatch):
    _anonymous, authenticated, _service = clients
    from capabilities.video_library import adapter

    captured = []
    monkeypatch.setattr(
        adapter,
        "add_library_source_root_data",
        lambda library_id, body: captured.append(("source", library_id, body))
        or (200, {"ok": True, "data": {"source_roots": [body["path"]]}, "error": None}),
        raising=False,
    )
    monkeypatch.setattr(
        adapter,
        "migrate_legacy_library_data",
        lambda library_id: captured.append(("migrate", library_id))
        or (200, {"ok": True, "data": {"imported": 1, "skipped": 0, "failed": 0}, "error": None}),
        raising=False,
    )

    rooted = authenticated.post(
        "/api/capabilities/video-library/libraries/beef-noodle/source-roots",
        json={"path": "/vault/material"},
    )
    migrated = authenticated.post(
        "/api/capabilities/video-library/libraries/beef-noodle/migrate-legacy",
        json={},
    )

    assert rooted.status_code == 200
    assert migrated.status_code == 200
    assert captured == [
        ("source", "beef-noodle", {"path": "/vault/material"}),
        ("migrate", "beef-noodle"),
    ]


def test_named_library_clip_query_and_tag_write_are_isolated(clients, monkeypatch):
    _anonymous, authenticated, default_service = clients
    from capabilities.video_library import adapter

    named = VideoLibraryService(VideoLibraryStore(root=default_service.store.root.parent / "named"))
    calls = []
    monkeypatch.setattr(adapter, "get_named_service", lambda _library_id: named)
    monkeypatch.setattr(
        named.store,
        "search_clips",
        lambda query, *, tag=None, limit=50: calls.append((query, tag, limit))
        or [{"id": "named-clip"}],
    )
    monkeypatch.setattr(
        named.store,
        "replace_clip_tags",
        lambda clip_id, tags: [{"id": "tag-1", "name": tags[0]["name"]}],
    )

    listed = authenticated.get(
        "/api/capabilities/video-library/clips",
        params={
            "library_id": "beef-noodle",
            "limit": 5,
            "query": "热气牛肉",
            "tag": "场景/后厨",
        },
    )
    tagged = authenticated.post(
        "/api/capabilities/video-library/clips/named-clip/tags",
        json={"libraryId": "beef-noodle", "tags": ["人工确认"]},
    )

    assert listed.json()["data"]["clips"] == [{"id": "named-clip"}]
    assert calls == [("热气牛肉", "场景/后厨", 5)]
    assert tagged.json()["data"]["tags"][0]["name"] == "人工确认"

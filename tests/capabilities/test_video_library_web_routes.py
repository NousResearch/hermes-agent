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

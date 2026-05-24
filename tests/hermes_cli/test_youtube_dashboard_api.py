"""FastAPI endpoint tests for the local YouTube dashboard APIs."""

from starlette.testclient import TestClient


def test_youtube_publish_plan_endpoint_requires_auth(_isolate_hermes_home):
    from hermes_cli.web_server import app

    client = TestClient(app)

    response = client.get("/api/youtube/queue/missing/publish-plan")

    assert response.status_code == 401


def test_youtube_publish_readiness_and_plan_endpoints(_isolate_hermes_home, tmp_path):
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    video = tmp_path / "endpoint.mp4"
    thumbnail = tmp_path / "endpoint.png"
    captions = tmp_path / "endpoint.vtt"
    for path in (video, thumbnail, captions):
        path.write_text("fixture", encoding="utf-8")

    created = client.post(
        "/api/youtube/queue",
        json={
            "channel_id": "scripturedepth",
            "title": "Endpoint publish plan smoke",
            "description": "A dry-run description.",
            "format": "short",
            "tags": ["Bible", "Shorts"],
            "source_refs": ["John 3:16"],
            "asset_paths": {
                "video": str(video),
                "thumbnail": str(thumbnail),
                "captions": str(captions),
            },
            "checks": {
                "video_file": True,
                "thumbnail": True,
                "title": True,
                "description": True,
                "captions": True,
                "sources_or_scripture_refs": True,
                "human_approval": True,
            },
            "review_status": "approved",
            "risk": "low",
        },
    )
    assert created.status_code == 200, created.text
    item_id = created.json()["id"]

    readiness = client.get(f"/api/youtube/queue/{item_id}/publish-readiness")
    plan = client.get(f"/api/youtube/queue/{item_id}/publish-plan")
    missing = client.get("/api/youtube/queue/nope/publish-plan")

    assert readiness.status_code == 200, readiness.text
    assert plan.status_code == 200, plan.text
    assert missing.status_code == 404
    assert readiness.json()["ready"] is True
    assert readiness.json()["publish_enabled"] is False
    assert plan.json()["publish_enabled"] is False
    assert plan.json()["youtube_api_call_allowed"] is False
    assert plan.json()["payload_preview"]["video_path"] == str(video)

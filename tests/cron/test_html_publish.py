from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cron.html_publish import (
    HtmlArtifactPublishError,
    artifact_object_key,
    publish_html_artifact,
    resolve_publish_settings,
)


def test_artifact_object_key_sanitizes_job_and_filename(tmp_path: Path):
    path = tmp_path / "2026-05-20_11:25:00.html"
    assert artifact_object_key("job/../id", path) == "r/job-id/2026-05-20_11-25-00.html"


def test_resolve_publish_settings_requires_global_and_per_job_enablement():
    job = {"id": "pilot"}
    html_settings = {
        "publish": {"enabled": False, "endpoint": "https://acta.imperatr.com"},
        "jobs": {"pilot": {"publish": True}},
    }
    resolved = resolve_publish_settings(job, html_settings)
    assert resolved["enabled"] is False

    html_settings["publish"]["enabled"] = True
    resolved = resolve_publish_settings(job, html_settings)
    assert resolved["enabled"] is True
    assert resolved["endpoint"] == "https://acta.imperatr.com"


def test_publish_disabled_returns_none(tmp_path: Path):
    html = tmp_path / "brief.html"
    html.write_text("<html></html>")
    assert publish_html_artifact(html, {"id": "job"}, {"enabled": False}) is None


def test_publish_requires_https_endpoint_and_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = tmp_path / "brief.html"
    html.write_text("<html></html>")
    monkeypatch.setenv("ACTA_UPLOAD_TOKEN", "token")
    with pytest.raises(HtmlArtifactPublishError, match="https"):
        publish_html_artifact(html, {"id": "job"}, {"enabled": True, "endpoint": "http://example.com"})
    monkeypatch.delenv("ACTA_UPLOAD_TOKEN")
    with pytest.raises(HtmlArtifactPublishError, match="ACTA_UPLOAD_TOKEN"):
        publish_html_artifact(html, {"id": "job"}, {"enabled": True, "endpoint": "https://acta.imperatr.com"})


def test_publish_uploads_html_and_returns_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = tmp_path / "brief.html"
    body = "<html>ok</html>"
    html.write_text(body)
    monkeypatch.setenv("ACTA_UPLOAD_TOKEN", "secret-token")
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n=-1):
            return json.dumps({"url": "https://acta.imperatr.com/r/job/brief.html?exp=1&sig=abc"}).encode()

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["token"] = req.headers.get("X-acta-upload-token") or req.headers.get("X-Acta-Upload-Token")
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("cron.html_publish.urllib.request.urlopen", fake_urlopen)
    url = publish_html_artifact(
        html,
        {"id": "job"},
        {"enabled": True, "endpoint": "https://acta.imperatr.com", "timeout_seconds": 3},
    )
    assert url is not None
    assert url.startswith("https://acta.imperatr.com/r/job/brief.html")
    digest = hashlib.sha256(body.encode()).hexdigest()[:12]
    assert seen["url"] == f"https://acta.imperatr.com/__upload/r/job/brief-{digest}.html"
    assert seen["method"] == "PUT"
    assert seen["token"] == "secret-token"
    assert seen["timeout"] == 3


def test_publish_allows_configured_public_object_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = tmp_path / "index.html"
    html.write_text("<html>home</html>")
    monkeypatch.setenv("ACTA_UPLOAD_TOKEN", "secret-token")
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n=-1):
            return json.dumps({"url": "https://acta.imperatr.com/"}).encode()

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        return FakeResponse()

    monkeypatch.setattr("cron.html_publish.urllib.request.urlopen", fake_urlopen)
    url = publish_html_artifact(
        html,
        {"id": "acta-situation-room"},
        {"enabled": True, "endpoint": "https://acta.imperatr.com", "object_key": "public/index.html"},
    )
    assert url == "https://acta.imperatr.com/"
    assert seen["url"] == "https://acta.imperatr.com/__upload/public/index.html"

    with pytest.raises(HtmlArtifactPublishError, match="object_key"):
        publish_html_artifact(
            html,
            {"id": "acta-situation-room"},
            {"enabled": True, "endpoint": "https://acta.imperatr.com", "object_key": "../bad.html"},
        )


def test_publish_rejects_oversized_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = tmp_path / "brief.html"
    html.write_text("x" * 2048)
    monkeypatch.setenv("ACTA_UPLOAD_TOKEN", "token")
    with pytest.raises(HtmlArtifactPublishError, match="size cap"):
        publish_html_artifact(html, {"id": "job"}, {"enabled": True, "endpoint": "https://acta.imperatr.com", "max_kb": 1})


def test_publish_rejects_non_acta_returned_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    html = tmp_path / "brief.html"
    html.write_text("<html>ok</html>")
    monkeypatch.setenv("ACTA_UPLOAD_TOKEN", "token")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n=-1):
            return json.dumps({"url": "https://evil.example/r/job/brief.html?sig=x"}).encode()

    monkeypatch.setattr("cron.html_publish.urllib.request.urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(HtmlArtifactPublishError, match="Acta URL"):
        publish_html_artifact(html, {"id": "job"}, {"enabled": True, "endpoint": "https://acta.imperatr.com"})

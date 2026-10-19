"""Offline tests for optional-skills/productivity/deed-page/scripts/deed_publish.py."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "optional-skills" / "productivity" / "deed-page" / "scripts" / "deed_publish.py"


def _load():
    spec = importlib.util.spec_from_file_location("deed_publish", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_rejects_invalid_slug(tmp_path):
    mod = _load()
    f = tmp_path / "index.html"
    f.write_text("<h1>x</h1>")
    status, payload = mod.publish(f, "Bad_Slug")
    assert status == 400 and payload["error"] == "slug_invalid"


def test_folder_becomes_tarball_at_root(tmp_path):
    mod = _load()
    (tmp_path / "index.html").write_text("<h1>x</h1>")
    (tmp_path / "css").mkdir()
    (tmp_path / "css" / "a.css").write_text("body{}")
    body, ctype = mod.build_body(tmp_path)
    assert ctype == "application/gzip"
    names = tarfile.open(fileobj=io.BytesIO(body), mode="r:gz").getnames()
    assert sorted(names) == ["css/a.css", "index.html"]


def test_publish_sends_ref_client_and_token(tmp_path, monkeypatch):
    mod = _load()
    f = tmp_path / "index.html"
    f.write_text("<h1>x</h1>")
    monkeypatch.setenv("DEEDPAGE_TOKEN", "deed_tok_test")
    monkeypatch.delenv("DEED_REF", raising=False)
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["req"] = req
        return _Resp(json.dumps({"ok": True, "url": "https://ok-123.deed.page/"}).encode())

    with mock.patch.object(mod.urllib.request, "urlopen", fake_urlopen):
        status, payload = mod.publish(f, "ok-123", ttl=60)
    req = captured["req"]
    assert status == 200 and payload["url"].startswith("https://ok-123")
    assert req.get_method() == "PUT"
    assert "slug=ok-123" in req.full_url and "ref=skill-hermes" in req.full_url
    assert req.get_header("X-client") == "Hermes"
    assert req.get_header("Authorization") == "Bearer deed_tok_test"
    assert req.get_header("X-ttl") == "60"


def test_delete_requires_token(monkeypatch):
    mod = _load()
    monkeypatch.delenv("DEEDPAGE_TOKEN", raising=False)
    status, _ = mod.delete("ok-123")
    assert status == 401

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "imgflip-meme-generator"
    / "scripts"
    / "imgflip_free_tier.py"
)


class FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200, content_type: str = "application/json"):
        self._payload = payload
        self.status = status
        self.headers = FakeHeaders({"content-type": content_type})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


def load_module():
    spec = importlib.util.spec_from_file_location("imgflip_free_tier", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def json_response(payload: dict) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode("utf-8"))


def test_get_memes_lists_templates(monkeypatch):
    module = load_module()

    def fake_urlopen(request, timeout=20):
        assert "get_memes" in request.full_url
        return json_response(
            {
                "success": True,
                "data": {
                    "memes": [
                        {
                            "id": "181913649",
                            "name": "Drake Hotline Bling",
                            "url": "https://i.imgflip.com/30b1gx.jpg",
                            "width": 1200,
                            "height": 1200,
                            "box_count": 2,
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    memes = module.get_memes()
    assert memes[0]["id"] == "181913649"
    assert memes[0]["box_count"] == 2


def test_make_requires_environment_credentials(monkeypatch):
    module = load_module()
    monkeypatch.delenv("IMGFLIP_USERNAME", raising=False)
    monkeypatch.delenv("IMGFLIP_PASSWORD", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        module.load_credentials()

    assert exc_info.value.code == 2


def test_download_rejects_unexpected_image_host(monkeypatch, capsys):
    module = load_module()

    def fake_urlopen(request, timeout=20):
        raise AssertionError("download should be rejected before network access")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit) as exc_info:
        module.download("https://example.com/meme.jpg")

    result = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert result["success"] is False
    assert "unexpected image URL host" in result["error"]


def test_http_errors_return_json(monkeypatch, capsys):
    module = load_module()

    def fake_urlopen(request, timeout=20):
        raise module.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit) as exc_info:
        module.get_memes()

    result = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert result["success"] is False
    assert "HTTP 403" in result["error"]


def test_make_meme_posts_form_and_verifies_download(monkeypatch, tmp_path, capsys):
    module = load_module()
    monkeypatch.setenv("IMGFLIP_USERNAME", "example-user")
    monkeypatch.setenv("IMGFLIP_PASSWORD", "example-password")

    output_path = tmp_path / "meme.jpg"
    calls = []

    def fake_urlopen(request, timeout=20):
        calls.append(request)
        if "get_memes" in request.full_url:
            return json_response(
                {
                    "success": True,
                    "data": {
                        "memes": [
                            {
                                "id": "181913649",
                                "name": "Drake Hotline Bling",
                                "url": "https://i.imgflip.com/30b1gx.jpg",
                                "width": 1200,
                                "height": 1200,
                                "box_count": 2,
                            }
                        ]
                    },
                }
            )
        if "caption_image" in request.full_url:
            form = parse_qs(request.data.decode("utf-8"))
            assert form["username"] == ["example-user"]
            assert form["password"] == ["example-password"]
            assert form["template_id"] == ["181913649"]
            assert form["text0"] == ["TOP"]
            assert form["text1"] == ["BOTTOM"]
            return json_response(
                {
                    "success": True,
                    "data": {
                        "url": "https://i.imgflip.com/example.jpg",
                        "page_url": "https://imgflip.com/i/example",
                    },
                }
            )
        if "example.jpg" in request.full_url:
            return FakeResponse(b"fake-jpeg", content_type="image/jpeg")
        raise AssertionError(f"unexpected request: {request.full_url}")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    args = type(
        "Args",
        (),
        {
            "template_id": "181913649",
            "template_name": None,
            "boxes": None,
            "text0": "TOP",
            "text1": "BOTTOM",
            "font": "impact",
            "max_font_size": None,
            "verify_download": True,
            "output": str(output_path),
        },
    )()

    module.make_meme(args)
    result = json.loads(capsys.readouterr().out)

    assert result["success"] is True
    assert result["verified"]["content_type"] == "image/jpeg"
    assert result["verified"]["looks_like_image"] is True
    assert output_path.read_bytes() == b"fake-jpeg"
    assert len(calls) == 3

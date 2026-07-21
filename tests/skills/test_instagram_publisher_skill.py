"""Tests for the optional instagram-publisher skill."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "optional-skills/social-media/instagram-publisher"
SKILL_MD = SKILL_ROOT / "SKILL.md"
SCRIPT_PATH = SKILL_ROOT / "scripts/publish_instagram.py"


@pytest.fixture
def instagram_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location("instagram_publish_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_skill_description_is_short():
    text = SKILL_MD.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("description: "))
    description = line.split(": ", 1)[1].strip().strip('"')
    assert len(description) <= 60
    assert description.endswith(".")


def test_required_files_exist():
    required = [
        "references/configuration.md",
        "references/publishing-examples.md",
        "scripts/publish_instagram.py",
    ]
    missing = [path for path in required if not (SKILL_ROOT / path).exists()]
    assert missing == []


def test_resolve_credentials_prefers_args(instagram_module, monkeypatch):
    args = instagram_module.argparse.Namespace(
        api_key="arg-key",
        connection_id="arg-conn",
        account_id="arg-account",
    )
    monkeypatch.setenv("INSTAGRAM_API_KEY", "env-key")
    creds = instagram_module.resolve_credentials(
        args,
        {"instagram": {"api_key": "file-key", "connection_id": "file", "account_id": "file"}},
    )
    assert creds == ("arg-key", "arg-conn", "arg-account")


def test_detect_media_type(instagram_module):
    assert instagram_module.detect_media_type("clip.mp4") == "VIDEO"
    assert instagram_module.detect_media_type("image.jpg") == "IMAGE"
    assert instagram_module.detect_media_type("https://example.com/video?id=mp4") == "VIDEO"


def test_publish_image_url_sends_json(instagram_module):
    publisher = instagram_module.InstagramPublisher("api-key", "conn", "acct")
    response = MagicMock(status_code=200)
    response.json.return_value = {"publish_id": "v_pub_file~123"}

    with patch.object(instagram_module.requests, "post", return_value=response) as post:
        result = publisher.publish_once(
            "IMAGE",
            "caption",
            url="https://example.com/image.jpg",
            retries=0,
        )

    assert result == {"publish_id": "v_pub_file~123"}
    _, kwargs = post.call_args
    assert kwargs["headers"]["X-API_KEY"] == "api-key"
    assert kwargs["json"]["connection_id"] == "conn"
    assert kwargs["json"]["account_id"] == "acct"
    assert kwargs["json"]["image_url"] == "https://example.com/image.jpg"


def test_publish_local_missing_file_raises(instagram_module, tmp_path):
    publisher = instagram_module.InstagramPublisher("api-key", "conn", "acct")
    with pytest.raises(FileNotFoundError):
        publisher.publish_once(
            "IMAGE",
            "caption",
            path=str(tmp_path / "missing.jpg"),
            retries=0,
        )


def test_check_status_sends_identifiers(instagram_module):
    publisher = instagram_module.InstagramPublisher("api-key", "conn", "acct")
    response = MagicMock(status_code=200)
    response.json.return_value = {"status": "FINISHED"}

    with patch.object(instagram_module.requests, "get", return_value=response) as get:
        result = publisher.check_status("v_pub_file~123")

    assert result == {"status": "FINISHED"}
    _, kwargs = get.call_args
    assert kwargs["headers"]["X-API_KEY"] == "api-key"
    assert kwargs["params"] == {
        "publish_id": "v_pub_file~123",
        "connection_id": "conn",
        "account_id": "acct",
    }

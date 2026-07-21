"""Tests for the optional tiktok-publisher skill."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "optional-skills/social-media/tiktok-publisher"
SKILL_MD = SKILL_ROOT / "SKILL.md"
PUBLISH_SCRIPT = SKILL_ROOT / "scripts/publish_tiktok.py"
STATUS_SCRIPT = SKILL_ROOT / "scripts/check_status.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def publish_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    return load_module(PUBLISH_SCRIPT, "tiktok_publish_test")


@pytest.fixture
def status_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    return load_module(STATUS_SCRIPT, "tiktok_status_test")


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
        "scripts/publish_tiktok.py",
        "scripts/check_status.py",
    ]
    missing = [path for path in required if not (SKILL_ROOT / path).exists()]
    assert missing == []


def test_resolve_api_key_prefers_environment(publish_module, monkeypatch):
    monkeypatch.setenv("TIKTOK_API_KEY", "env-key")
    assert publish_module.resolve_api_key({"tiktok": {"api_key": "file-key"}}) == "env-key"


def test_publish_url_sends_json_payload(publish_module):
    args = publish_module.argparse.Namespace(
        source="https://example.com/video.mp4",
        title="Title",
        privacy_level="SELF_ONLY",
        wait_for_published=True,
        poll_interval=5000,
        poll_timeout=300000,
    )
    response = MagicMock(status_code=200)
    response.json.return_value = {"publish_id": "abc123"}

    with patch.object(publish_module.requests, "post", return_value=response) as post:
        publish_module.publish_tiktok(args, "api-key")

    _, kwargs = post.call_args
    assert kwargs["headers"]["X-API-Key"] == "api-key"
    assert kwargs["json"]["video_url"] == "https://example.com/video.mp4"
    assert kwargs["json"]["wait_for_published"] is True
    assert kwargs["json"]["poll_interval_ms"] == 5000


def test_publish_local_file_requires_existing_path(publish_module, tmp_path):
    args = publish_module.argparse.Namespace(
        source=str(tmp_path / "missing.mp4"),
        title="Title",
        privacy_level="SELF_ONLY",
        wait_for_published=False,
        poll_interval=None,
        poll_timeout=None,
    )
    with pytest.raises(SystemExit):
        publish_module.publish_tiktok(args, "api-key")


def test_check_status_sends_publish_id(status_module):
    response = MagicMock(status_code=200)
    response.json.return_value = {"status": "PUBLISHED"}

    with patch.object(status_module.requests, "get", return_value=response) as get:
        status_module.check_status("pub123", "api-key")

    _, kwargs = get.call_args
    assert kwargs["headers"]["X-API-Key"] == "api-key"
    assert kwargs["params"] == {"publish_id": "pub123"}

"""Checks for the bundled media-delivery-prep skill."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "skills" / "media" / "media-delivery-prep" / "SKILL.md"


def _load_skill():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    _, frontmatter, body = content.split("---", 2)
    return yaml.safe_load(frontmatter), body


def test_media_delivery_prep_skill_exists_with_expected_metadata():
    frontmatter, body = _load_skill()

    assert frontmatter["name"] == "media-delivery-prep"
    assert "send" in frontmatter["description"].lower()
    assert "media" in frontmatter["description"].lower()
    assert frontmatter["platforms"] == ["linux", "macos", "windows"]
    assert "media" in frontmatter["metadata"]["hermes"]["tags"]
    assert "songsee" in frontmatter["metadata"]["hermes"]["related_skills"]
    assert "youtube-content" in frontmatter["metadata"]["hermes"]["related_skills"]

    assert "MEDIA:/absolute/path" in body
    assert "ffmpeg" in body
    assert "ffprobe" in body
    assert "send_message" in body
    assert "telegram" in body.lower()
    assert "discord" in body.lower()
    assert "signal" in body.lower()


def test_media_description_mentions_delivery_prep_skill():
    description = (REPO_ROOT / "skills" / "media" / "DESCRIPTION.md").read_text(
        encoding="utf-8"
    )

    assert "delivery prep" in description.lower()

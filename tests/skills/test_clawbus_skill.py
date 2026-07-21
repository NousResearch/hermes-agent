"""Tests for the optional clawbus skill."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "optional-skills/productivity/clawbus/SKILL.md"
SCRIPT_PATH = REPO_ROOT / "optional-skills/productivity/clawbus/scripts/clawbus.py"


@pytest.fixture
def clawbus_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    spec = importlib.util.spec_from_file_location("clawbus_test", SCRIPT_PATH)
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


def test_safe_file_path_rejects_escape(clawbus_module, tmp_path):
    with pytest.raises(ValueError):
        clawbus_module.safe_file_path(tmp_path, "../evil.py")
    with pytest.raises(ValueError):
        clawbus_module.safe_file_path(tmp_path, "/tmp/evil.py")


def test_install_writes_files_and_meta(clawbus_module, tmp_path):
    payload = {
        "skill": {"slug": "youtube-unified-api", "name": "YouTube Unified API"},
        "files": [
            {"path": "SKILL.md", "content": "---\nname: youtube-unified-api\n---\n"},
            {"path": "scripts/client.py", "content": "print('ok')\n"},
        ],
    }

    with patch.object(clawbus_module, "api_get", return_value=payload):
        result = clawbus_module.main(
            [
                "install",
                "youtube-unified-api",
                "--skills-dir",
                str(tmp_path / "skills"),
            ]
        )

    assert result == 0
    destination = tmp_path / "skills" / "youtube-unified-api"
    assert (destination / "SKILL.md").exists()
    assert (destination / "scripts/client.py").read_text() == "print('ok')\n"
    meta = json.loads((destination / "_meta.json").read_text())
    assert meta["source"] == "clawbus"
    assert meta["slug"] == "youtube-unified-api"


def test_install_rejects_unsafe_api_file_path(clawbus_module, tmp_path):
    payload = {
        "skill": {"slug": "bad"},
        "files": [{"path": "../bad.py", "content": "bad"}],
    }

    with patch.object(clawbus_module, "api_get", return_value=payload):
        with pytest.raises(SystemExit) as exc:
            clawbus_module.main(
                ["install", "bad", "--skills-dir", str(tmp_path / "skills")]
            )

    assert "unsafe file path" in str(exc.value)


def test_api_get_uses_expected_url_and_timeout(clawbus_module):
    captured = {}
    response = MagicMock()
    response.read.return_value = json.dumps({"skills": []}).encode()
    response.__enter__ = lambda self: self
    response.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return response

    with patch.object(clawbus_module.urllib.request, "urlopen", side_effect=fake_urlopen):
        assert clawbus_module.api_get("/skills/search", {"q": "seo", "limit": 3}) == {
            "skills": []
        }

    assert captured["timeout"] == 20
    assert captured["url"].startswith("https://www.clawbus.com/api/skills/search?")
    assert "q=seo" in captured["url"]
    assert "limit=3" in captured["url"]

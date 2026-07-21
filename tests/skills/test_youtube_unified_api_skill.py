"""Tests for the optional youtube-unified-api skill."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "optional-skills/media/youtube-unified-api"
SKILL_MD = SKILL_ROOT / "SKILL.md"
SYNC_SCRIPT = SKILL_ROOT / "scripts/sync_discovery.py"


def test_skill_description_is_short():
    text = SKILL_MD.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("description: "))
    description = line.split(": ", 1)[1].strip().strip('"')
    assert len(description) <= 60
    assert description.endswith(".")


def test_required_references_exist():
    required = [
        "references/index.md",
        "references/curl.md",
        "references/mybrandmetrics-api.md",
        "references/catalog.json",
        "references/services/youtube-data-v3.md",
        "references/services/youtube-analytics-v2.md",
        "references/services/youtube-reporting-v1.md",
        "references/discovery_cache/youtube.json",
        "references/discovery_cache/youtubeAnalytics.json",
        "references/discovery_cache/youtubeReporting.json",
    ]
    missing = [path for path in required if not (SKILL_ROOT / path).exists()]
    assert missing == []


def test_catalog_and_discovery_json_parse():
    catalog = json.loads((SKILL_ROOT / "references/catalog.json").read_text())
    assert catalog

    for service in ["youtube", "youtubeAnalytics", "youtubeReporting"]:
        data = json.loads(
            (SKILL_ROOT / f"references/discovery_cache/{service}.json").read_text()
        )
        assert data.get("resources") or data.get("schemas")


def test_sync_discovery_writes_fetched_documents(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("youtube_sync_test", SYNC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)

    response = MagicMock()
    response.read.return_value = b'{"resources":{"channels":{}}}'
    response.__enter__ = lambda self: self
    response.__exit__ = MagicMock(return_value=False)

    with patch.object(module, "urlopen", return_value=response):
        module.main()

    for service in module.DISCOVERY_URLS:
        parsed = json.loads((tmp_path / f"{service}.json").read_text())
        assert parsed["resources"]["channels"] == {}

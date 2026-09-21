"""Path bootstrap for content_engine tests.

content_engine/ is a standalone package tree whose modules use flat imports
(``import approval_state``, ``from blog.blog_generator import ...``) resolved
from the content_engine root. This conftest puts that root on sys.path so the
suite runs identically under any runner (pytest from anywhere, run_tests.sh,
IDE test discovery) without PYTHONPATH ceremony.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CE_ROOT = Path(__file__).resolve().parent.parent
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

import pytest
import json

@pytest.fixture(autouse=True)
def isolate_x_external_knowledge(monkeypatch, tmp_path):
    # Unit tests never crawl the operator account or personal working trees.
    # Dedicated real integration probes exercise these separately.
    monkeypatch.setenv("X_GITHUB_KNOWLEDGE", "0")
    monkeypatch.setenv("X_REFRESH_OWN_REFERENCES", "0")
    monkeypatch.setenv("X_KNOWLEDGE_ROOTS_JSON", json.dumps([str(tmp_path / "repos")]))


@pytest.fixture
def blog_exclusions(monkeypatch, tmp_path):
    """Exclusion tests own their policy data, not the mutable production list."""
    from blog import exclusions
    path = tmp_path / "test-exclusions.json"
    path.write_text(json.dumps({"items":[{"title":title,"reason":"test policy"} for title in ("Cheap-first model routing", "Reference-anchored image generation", "What production AI agent actually means")]}))
    monkeypatch.setattr(exclusions, "EXCLUSIONS_PATH", path)
    return path

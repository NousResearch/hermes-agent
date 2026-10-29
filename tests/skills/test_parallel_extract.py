"""Tests for skills/research/parallel-extract."""
from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "research" / "parallel-extract"
SCRIPT = SKILL_DIR / "scripts" / "extract_docs.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"
    assert desc.startswith("Use when "), desc


def test_author_credits_contributor(frontmatter) -> None:
    assert "Nietzsche-Ubermensch" in frontmatter["author"]


def test_extract_urls_batches_and_writes_json(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    class FakeParallel:
        def __init__(self):
            self._n = 0

        def extract(self, *, urls, session_id=None, **kwargs):
            calls.append(list(urls))
            self._n += 1
            return SimpleNamespace(
                session_id=f"sess-{self._n}",
                results=[
                    SimpleNamespace(
                        url=u,
                        title="t",
                        publish_date=None,
                        excerpts=["ex"],
                        full_content="body",
                    )
                    for u in urls
                ],
                errors=[],
                usage=[SimpleNamespace(name="sku_extract_excerpts", count=len(urls))],
            )

    fake_parallel = MagicMock()
    fake_parallel.Parallel = FakeParallel
    monkeypatch.setitem(sys.modules, "parallel", fake_parallel)

    mod = runpy.run_path(str(SCRIPT), run_name="extract_docs_under_test")
    urls = [f"https://example.com/{i}" for i in range(25)]
    out = tmp_path / "out.json"
    payload = mod["extract_urls"](urls, out_path=str(out), full_content=True)

    assert len(calls) == 2
    assert len(calls[0]) == 20
    assert len(calls[1]) == 5
    assert len(payload["results"]) == 25
    data = json.loads(out.read_text())
    assert len(data["results"]) == 25


def test_script_main_entrypoint_exists() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in text
    assert "def main(" in text
"""Tests for the bundled arXiv research skill and its search script."""

import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
import urllib.parse
import urllib.request
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "research" / "arxiv"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS_DIR = SKILL_DIR / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "search_arxiv.py"

# Add script directory to sys.path for direct import
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_arxiv

SAMPLE_ARXIV_ATOM_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2402.03300v2</id>
    <updated>2024-02-10T12:00:00Z</updated>
    <published>2024-02-05T08:00:00Z</published>
    <title>
      DeepSeekMath: Pushing the Limits of Mathematical Reasoning
    </title>
    <summary>
      Mathematical reasoning is a challenging task for language models. We explore reinforcement learning with GRPO.
    </summary>
    <author>
      <name>Author One</name>
    </author>
    <author>
      <name>Author Two</name>
    </author>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <updated>2023-08-02T10:00:00Z</updated>
    <published>2017-06-12T14:00:00Z</published>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
    <author>
      <name>Ashish Vaswani</name>
    </author>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

SAMPLE_EMPTY_ATOM_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>0</opensearch:itemsPerPage>
</feed>
"""


class DummyResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestArxivSkillContract:
    """Test that SKILL.md adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "arxiv"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]
        assert "related_skills" in hermes

    def test_description_hardline(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        desc = fm["description"]
        assert len(desc) <= 60, f"description is {len(desc)} chars; max allowed is 60"
        assert desc.endswith("."), "description must end with a period"
        assert not re.search(
            r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
            desc,
            re.I,
        )

    def test_related_skills_resolve_in_repo(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        missing = resolve_related_skills_in_repo(fm["metadata"]["hermes"]["related_skills"])
        assert not missing, f"related_skills entries do not resolve in repo: {missing}"

    def test_no_machine_local_paths(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        assert "/home/" not in content
        assert not re.search(r"[A-Z]:\\\\Users", content)

    def test_body_structure(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for section in ("## Quick Reference", "## Helper Script"):
            assert section in body, f"missing section: {section}"


class TestSearchArxivScript:
    """Unit tests for skills/research/arxiv/scripts/search_arxiv.py."""

    def test_search_requires_at_least_one_parameter(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            search_arxiv.search()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: provide a query, --author, --category, or --id" in captured.out

    def test_search_by_query_url_and_parsing(self, monkeypatch, capsys):
        recorded_url = []

        def mock_urlopen(req, timeout=15):
            recorded_url.append(req.full_url)
            assert req.headers.get("User-agent") == "HermesAgent/1.0"
            return DummyResponse(SAMPLE_ARXIV_ATOM_XML)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        search_arxiv.search(query="GRPO reinforcement learning", max_results=2, sort="relevance")

        assert len(recorded_url) == 1
        url = recorded_url[0]
        assert "export.arxiv.org/api/query?" in url
        assert "search_query=all%3AGRPO%2520reinforcement%2520learning" in url or "all:GRPO" in urllib.parse.unquote(url)
        assert "max_results=2" in url
        assert "sortBy=relevance" in url
        assert "sortOrder=descending" in url

        captured = capsys.readouterr()
        assert "Found 2 results (showing 2)" in captured.out
        assert "1. DeepSeekMath: Pushing the Limits of Mathematical Reasoning" in captured.out
        assert "ID: 2402.03300v2" in captured.out
        assert "Authors: Author One, Author Two" in captured.out
        assert "Categories: cs.AI, cs.CL" in captured.out
        assert "https://arxiv.org/abs/2402.03300" in captured.out
        assert "https://arxiv.org/pdf/2402.03300" in captured.out
        assert "2. Attention Is All You Need" in captured.out
        assert "Ashish Vaswani" in captured.out

    def test_search_by_author_and_category(self, monkeypatch, capsys):
        recorded_url = []

        def mock_urlopen(req, timeout=15):
            recorded_url.append(req.full_url)
            return DummyResponse(SAMPLE_ARXIV_ATOM_XML)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        search_arxiv.search(author="Yann LeCun", category="cs.AI", max_results=5, sort="date")

        assert len(recorded_url) == 1
        url = recorded_url[0]
        assert "sortBy=submittedDate" in url
        unquoted = urllib.parse.unquote(url)
        assert "au:Yann%20LeCun" in unquoted or "au:Yann LeCun" in unquoted
        assert "cat:cs.AI" in unquoted

    def test_search_by_ids(self, monkeypatch, capsys):
        recorded_url = []

        def mock_urlopen(req, timeout=15):
            recorded_url.append(req.full_url)
            return DummyResponse(SAMPLE_ARXIV_ATOM_XML)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        search_arxiv.search(ids="2402.03300,1706.03762", sort="updated")

        assert len(recorded_url) == 1
        url = recorded_url[0]
        assert "id_list=2402.03300%2C1706.03762" in url or "id_list=2402.03300,1706.03762" in url
        assert "sortBy=lastUpdatedDate" in url

    def test_search_no_results(self, monkeypatch, capsys):
        def mock_urlopen(req, timeout=15):
            return DummyResponse(SAMPLE_EMPTY_ATOM_XML)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        search_arxiv.search(query="nonexistent_paper_query_12345")
        captured = capsys.readouterr()
        assert "No results found." in captured.out

    def test_cli_help(self):
        """Verify real CLI help execution via subprocess."""
        res = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0
        assert "Search arXiv and display results in a clean format." in res.stdout

    def test_cli_argument_parsing(self, monkeypatch, capsys):
        """Verify real __main__ CLI argument parsing and execution hermetically."""
        recorded_url = []

        def mock_urlopen(req, timeout=15):
            recorded_url.append(req.full_url)
            return DummyResponse(SAMPLE_ARXIV_ATOM_XML)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT_PATH),
                "reinforcement",
                "learning",
                "--author",
                "Sutton",
                "--category",
                "cs.AI",
                "--max",
                "10",
                "--sort",
                "date",
            ],
        )

        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")

        assert len(recorded_url) == 1
        url = recorded_url[0]
        assert "max_results=10" in url
        assert "sortBy=submittedDate" in url
        unquoted = urllib.parse.unquote(url)
        assert "all:reinforcement learning" in unquoted or "all%3Areinforcement" in url
        assert "au:Sutton" in unquoted
        assert "cat:cs.AI" in unquoted

        captured = capsys.readouterr()
        assert "Found 2 results" in captured.out

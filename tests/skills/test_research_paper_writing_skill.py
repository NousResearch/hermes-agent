"""Offline behavior contracts for the research-paper-writing reference."""

from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock


REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/research/research-paper-writing/references/citation-workflow.md"
)


def citation_manager_class():
    """Load the documented manager so its failure branch stays executable."""
    text = REFERENCE.read_text(encoding="utf-8")
    match = re.search(r"{% raw %}\n```python\n(.*?)\n```\n{% endraw %}", text, re.DOTALL)
    assert match, "citation manager example must remain an executable Python block"
    namespace: dict[str, object] = {"__name__": "citation_workflow_example"}
    exec(match.group(1), namespace)
    return namespace["CitationManager"]


def test_citation_manager_rejects_unverified_paper_before_bibtex_retrieval():
    manager = citation_manager_class()()
    paper = SimpleNamespace(title="Synthetic Paper")
    manager.search = Mock(return_value=[paper])
    manager.verify = Mock(return_value=(False, ["Semantic Scholar"]))
    manager.get_bibtex = Mock(return_value="@article{unverified}")

    assert manager.cite("synthetic paper") is None
    assert manager.verified_papers == {}
    manager.get_bibtex.assert_not_called()


def test_citation_manager_returns_and_caches_verified_paper():
    manager = citation_manager_class()()
    paper = SimpleNamespace(title="Verified Paper")
    bibtex = "@article{verified}"
    manager.search = Mock(return_value=[paper])
    manager.verify = Mock(return_value=(True, ["Semantic Scholar", "CrossRef"]))
    manager.get_bibtex = Mock(return_value=bibtex)

    assert manager.cite("verified paper") == bibtex
    assert manager.verified_papers == {paper.title: paper}
    manager.get_bibtex.assert_called_once_with(paper)

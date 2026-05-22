from __future__ import annotations

import json

import pytest

from hermes_wiki.caretaker_orchestrator import (
    caretaker_action_to_proposal,
    main as orchestrator_main,
    run_caretaker_orchestrator,
)
from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import write_page


def _write_config(path, wiki_path):
    path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")


def test_caretaker_orchestrator_drafts_proposal_for_broken_link_without_writing(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    write_page(
        config.wiki_path / "concepts" / "broken.md",
        {"title": "Broken", "type": "concept", "sources": ["raw/articles/source.md"]},
        "Links to [[Missing Concept]].",
    )

    result = run_caretaker_orchestrator(config, queue=False)

    assert result.proposals
    assert result.queued_paths == []
    assert not (config.wiki_path / "proposals").exists()
    assert result.proposals[0].title.startswith("Repair broken link")
    assert result.proposals[0].target_pages == ["concepts/broken.md"]


def test_caretaker_orchestrator_queues_proposals_only_when_explicit(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    write_page(
        config.wiki_path / "concepts" / "no-source.md",
        {"title": "No Source", "type": "concept"},
        "Needs provenance.",
    )

    result = run_caretaker_orchestrator(config, queue=True)

    assert len(result.queued_paths) == 2  # missing source + orphan graph strengthening
    assert all(path.exists() for path in result.queued_paths)
    assert all(path.parent == config.wiki_path / "proposals" for path in result.queued_paths)


def test_caretaker_orchestrator_has_no_output_for_clean_wiki(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    write_page(
        config.wiki_path / "concepts" / "a.md",
        {"title": "A", "type": "concept", "sources": ["raw/articles/source.md"]},
        "Links to [[B]].",
    )
    write_page(
        config.wiki_path / "concepts" / "b.md",
        {"title": "B", "type": "concept", "sources": ["raw/articles/source.md"]},
        "Links to [[A]].",
    )

    result = run_caretaker_orchestrator(config)

    assert result.proposals == []
    assert result.queued_paths == []


def test_caretaker_action_to_proposal_rejects_unknown_action_kind():
    class Action:
        kind = "unknown"
        message = "Unknown"
        file_path = None
        severity = "info"

    assert caretaker_action_to_proposal(Action()) is None


def test_caretaker_orchestrator_cli_prints_drafts_without_writing(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    _write_config(config_path, wiki_path)
    write_page(
        wiki_path / "concepts" / "broken.md",
        {"title": "Broken", "type": "concept", "sources": ["raw/articles/source.md"]},
        "Links to [[Missing Concept]].",
    )

    code = orchestrator_main(["--config", str(config_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["queued"] is False
    assert payload["proposals"][0]["title"].startswith("Repair broken link")
    assert not (wiki_path / "proposals").exists()


def test_caretaker_orchestrator_cli_queue_requires_explicit_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-home"))

    with pytest.raises(SystemExit):
        orchestrator_main(["--queue"])

    assert not (tmp_path / "ambient-home").exists()


def test_caretaker_orchestrator_cli_queues_with_explicit_config(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    _write_config(config_path, wiki_path)
    write_page(
        wiki_path / "concepts" / "broken.md",
        {"title": "Broken", "type": "concept", "sources": ["raw/articles/source.md"]},
        "Links to [[Missing Concept]].",
    )

    code = orchestrator_main(["--config", str(config_path), "--queue", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["queued"] is True
    assert payload["queued_paths"]
    assert (wiki_path / "proposals").exists()

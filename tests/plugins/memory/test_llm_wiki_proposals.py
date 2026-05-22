from __future__ import annotations

import json

import pytest

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import parse_frontmatter
from hermes_wiki.proposals import MemoryProposal, main as propose_main, proposal_to_dict, queue_proposal, render_proposal_markdown


def test_render_proposal_markdown_is_source_backed_and_reviewable():
    proposal = MemoryProposal(
        title="Remember the example user's default workout context",
        rationale="Stable coaching preference that prevents repeated correction.",
        proposed_changes=["Prefer short home workouts by default."],
        source_refs=["session:test-thread"],
        target_pages=["concepts/fitness-coaching-preferences.md"],
        tags=["fitness", "preference"],
    )

    text = render_proposal_markdown(proposal)

    assert "# Remember the example user's default workout context" in text
    assert "## Rationale" in text
    assert "Stable coaching preference" in text
    assert "## Proposed changes" in text
    assert "- Prefer short home workouts by default." in text
    assert "## Source references" in text
    assert "session:test-thread" in text
    assert "concepts/fitness-coaching-preferences.md" in text


def test_proposal_to_dict_is_json_serializable():
    proposal = MemoryProposal(
        title="Strict eval config validation",
        rationale="Avoid evaluating the wrong wiki profile.",
        proposed_changes=["Make explicit --config authoritative."],
        source_refs=["raw/articles/llm-wiki-retrieval-eval-cli-and-karpathy-source-2026-05-21.md"],
        target_pages=["concepts/strict-eval-config-validation.md"],
    )

    payload = proposal_to_dict(proposal)

    assert json.loads(json.dumps(payload))["title"] == "Strict eval config validation"
    assert payload["status"] == "proposed"


def test_queue_proposal_requires_explicit_write(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    proposal = MemoryProposal(
        title="Draft only",
        rationale="Do not write unless explicitly requested.",
        proposed_changes=["No silent durable writes."],
        source_refs=["test"],
    )

    path = queue_proposal(config, proposal, write=False)

    assert path == config.wiki_path / "proposals" / "draft-only.md"
    assert not path.exists()


def test_queue_proposal_writes_review_file_when_explicit(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    proposal = MemoryProposal(
        title="Safe proposal queue",
        rationale="Review before ingestion.",
        proposed_changes=["Queue proposals as pending markdown."],
        source_refs=["test"],
        target_pages=["concepts/manual-curated-ingestion.md"],
    )

    path = queue_proposal(config, proposal, write=True)

    assert path == config.wiki_path / "proposals" / "safe-proposal-queue.md"
    fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm["type"] == "memory_proposal"
    assert fm["status"] == "proposed"
    assert fm["target_pages"] == ["concepts/manual-curated-ingestion.md"]
    assert "# Safe proposal queue" in body


def test_queue_proposal_rejects_unsafe_slug(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    proposal = MemoryProposal(
        title="../escape",
        rationale="Unsafe title should not be accepted.",
        proposed_changes=["Reject traversal."],
        source_refs=["test"],
        slug="../escape",
    )

    with pytest.raises(ValueError, match="unsafe proposal slug"):
        queue_proposal(config, proposal, write=True)


def test_queue_proposal_rejects_absolute_target_pages(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    proposal = MemoryProposal(
        title="Unsafe target",
        rationale="Target pages are review metadata but should still be sane.",
        proposed_changes=["Reject absolute target metadata."],
        source_refs=["test"],
        target_pages=["/tmp/outside.md"],
    )

    with pytest.raises(ValueError, match="target_pages"):
        queue_proposal(config, proposal, write=True)


def test_proposal_cli_prints_markdown_without_writing(tmp_path, capsys):
    code = propose_main(
        [
            "--title",
            "CLI draft",
            "--rationale",
            "Review before write.",
            "--change",
            "Draft only by default.",
            "--source",
            "test-source",
            "--target",
            "concepts/manual-curated-ingestion.md",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "# CLI draft" in out
    assert "Draft only by default." in out
    assert not (tmp_path / "wiki").exists()


def test_proposal_cli_queues_only_when_explicit(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")

    code = propose_main(
        [
            "--title",
            "CLI queued",
            "--rationale",
            "Explicit queue requested.",
            "--change",
            "Write review artifact only.",
            "--source",
            "test-source",
            "--config",
            str(config_path),
            "--queue",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    path = wiki_path / "proposals" / "cli-queued.md"
    assert code == 0
    assert payload == {"path": str(path), "queued": True}
    assert path.exists()


def test_proposal_cli_rejects_queue_without_explicit_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-home"))

    with pytest.raises(SystemExit):
        propose_main(
            [
                "--title",
                "No ambient write",
                "--rationale",
                "Queueing must be profile-explicit.",
                "--change",
                "Do not write to ambient wiki paths.",
                "--source",
                "test-source",
                "--queue",
            ]
        )

    assert not (tmp_path / "ambient-home").exists()

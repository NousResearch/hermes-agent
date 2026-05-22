from __future__ import annotations

import json

import pytest

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import parse_frontmatter
from hermes_wiki.proposal_lifecycle import (
    list_proposals,
    main as proposals_main,
    proposal_record_to_dict,
    read_proposal,
    update_proposal_status,
)
from hermes_wiki.proposals import MemoryProposal, queue_proposal


def _queue(config: WikiConfig, title: str, *, status: str = "proposed"):
    return queue_proposal(
        config,
        MemoryProposal(
            title=title,
            rationale="Review lifecycle behavior.",
            proposed_changes=["Keep proposal lifecycle explicit."],
            source_refs=["test-source"],
            target_pages=["concepts/example.md"],
            status=status,
        ),
        write=True,
    )


def test_list_proposals_returns_records_sorted_by_slug(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _queue(config, "Zulu proposal")
    _queue(config, "Alpha proposal", status="accepted")

    records = list_proposals(config)

    assert [record.slug for record in records] == ["alpha-proposal", "zulu-proposal"]
    assert [record.status for record in records] == ["accepted", "proposed"]
    assert records[0].path == config.wiki_path / "proposals" / "alpha-proposal.md"


def test_read_proposal_rejects_unsafe_slug(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")

    with pytest.raises(ValueError, match="unsafe proposal slug"):
        read_proposal(config, "../escape")


def test_update_proposal_status_updates_only_proposal_frontmatter(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    path = _queue(config, "Lifecycle proposal")
    body_before = parse_frontmatter(path.read_text(encoding="utf-8"))[1]

    record = update_proposal_status(config, "lifecycle-proposal", "accepted", note="Source-backed and applied.")

    fm, body_after = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert record.status == "accepted"
    assert fm["status"] == "accepted"
    assert fm["review_note"] == "Source-backed and applied."
    assert "reviewed" in fm
    assert body_after == body_before
    assert not (config.wiki_path / "concepts" / "example.md").exists()


def test_update_proposal_status_rejects_unknown_status(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _queue(config, "Lifecycle proposal")

    with pytest.raises(ValueError, match="unsupported proposal status"):
        update_proposal_status(config, "lifecycle-proposal", "maybe")


def test_proposal_record_to_dict_is_json_serializable(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _queue(config, "Serializable proposal")

    payload = proposal_record_to_dict(read_proposal(config, "serializable-proposal"))

    assert json.loads(json.dumps(payload))["slug"] == "serializable-proposal"
    assert payload["target_pages"] == ["concepts/example.md"]


def test_proposals_cli_lists_and_shows_with_explicit_config(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")
    _queue(WikiConfig(wiki_path=wiki_path, wiki_name="test"), "CLI proposal")

    code = proposals_main(["--config", str(config_path), "list", "--json"])
    list_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert list_payload[0]["slug"] == "cli-proposal"

    code = proposals_main(["--config", str(config_path), "show", "cli-proposal", "--json"])
    show_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert show_payload["title"] == "CLI proposal"


def test_proposals_cli_mutations_require_explicit_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-home"))

    with pytest.raises(SystemExit):
        proposals_main(["accept", "anything"])

    assert not (tmp_path / "ambient-home").exists()


def test_proposals_cli_accept_reject_close_update_status(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")
    config = WikiConfig(wiki_path=wiki_path, wiki_name="test")
    _queue(config, "Accept me")
    _queue(config, "Reject me")
    _queue(config, "Close me")

    assert proposals_main(["--config", str(config_path), "accept", "accept-me", "--note", "Applied."]) == 0
    accept_payload = json.loads(capsys.readouterr().out)
    assert accept_payload["status"] == "accepted"

    assert proposals_main(["--config", str(config_path), "reject", "reject-me"]) == 0
    reject_payload = json.loads(capsys.readouterr().out)
    assert reject_payload["status"] == "rejected"

    assert proposals_main(["--config", str(config_path), "close", "close-me"]) == 0
    close_payload = json.loads(capsys.readouterr().out)
    assert close_payload["status"] == "closed"

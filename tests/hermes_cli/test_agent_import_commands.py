"""Portable Claude command migration, a bounded part of feature request #35587."""
from pathlib import Path

import pytest
import yaml

from hermes_cli.agent_import import AgentImporter


def test_command_preview_conversion_and_conflict(tmp_path):
    source = tmp_path / "claude"
    commands = source / "commands"
    commands.mkdir(parents=True)
    original = "---\ndescription: Review a change.\n---\n# Review\n\nExplain correctness risks and missing tests.\n"
    (commands / "review.md").write_text(original)
    target = tmp_path / "hermes"
    destination = target / "skills" / "claude-code-commands" / "review" / "SKILL.md"

    preview = AgentImporter("claude-code", source, target).run()
    item = next(i for i in preview["items"] if i["kind"] == "slash-command")
    assert item["status"] == "imported"
    assert Path(item["destination"]) == destination
    assert not target.exists()

    report = AgentImporter("claude-code", source, target, execute=True).run()
    assert next(i for i in report["items"] if i["kind"] == "slash-command")["status"] == "imported"
    content = destination.read_text()
    _, metadata, body = content.split("---", 2)
    assert yaml.safe_load(metadata) == {"name": "claude-command-review", "description": "Review a change."}
    assert body.lstrip("\n") == original.split("---\n", 2)[2]
    assert (commands / "review.md").read_text() == original

    destination.write_text("Locally edited skill\n")
    conflict = AgentImporter("claude-code", source, target, execute=True).run()
    assert next(i for i in conflict["items"] if i["kind"] == "slash-command")["status"] == "conflict"
    assert destination.read_text() == "Locally edited skill\n"

    AgentImporter("claude-code", source, target, execute=True, overwrite=True).run()
    assert destination.read_text() == content
    # A redirect must remain untouched even when replacement was requested.
    outside = tmp_path / "outside.md"
    outside.write_text("Keep this file\n")
    destination.unlink()
    destination.symlink_to(outside)
    refused = AgentImporter("claude-code", source, target, execute=True, overwrite=True).run()
    assert next(i for i in refused["items"] if i["kind"] == "slash-command")["status"] == "skipped"
    assert outside.read_text() == "Keep this file\n"


@pytest.mark.parametrize("content", [
    "Summarize $ARGUMENTS\n", "Summarize $1\n", "Inspect !`git status`\n",
    "Read @src/main.py\n", "---\nallowed-tools: Bash\n---\nRun tests.\n",
    "---\nmodel: opus\n---\nReview code.\n", "---\ncontext: fork\n---\nReview code.\n",
    "---\ndescription: [invalid\n---\nReview code.\n", "---\ndescription: Unclosed\n",
    "", "---\n- invalid\n---\nReview code.\n",
    "---\ndescription: Review\n---broken\nReview code.\n",
])
def test_unsupported_commands_are_reported_without_writing(tmp_path, content):
    source = tmp_path / "claude"
    commands = source / "commands"
    commands.mkdir(parents=True)
    (commands / "review.md").write_text(content)
    target = tmp_path / "hermes"
    report = AgentImporter("claude-code", source, target, execute=True).run()
    item = next(i for i in report["items"] if i["kind"] == "slash-command")
    assert item["status"] == "skipped"
    assert item["reason"]
    assert not target.exists()
    assert (commands / "review.md").read_text() == content

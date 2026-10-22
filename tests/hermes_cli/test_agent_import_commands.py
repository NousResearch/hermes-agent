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


def test_command_sync_refreshes_independently_and_preserves_local_edits(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.agent_import import import_agent_command
    from hermes_cli.agent_import_sync import load_sync_manifest

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = tmp_path / ".claude"
    commands = source / "commands"
    commands.mkdir(parents=True)
    command = commands / "review.md"
    command.write_text("Review version one.\n")
    # Ordinary skills and command skills may legitimately have the same basename.
    skill = source / "skills" / "review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: review\n---\nOrdinary version one.\n")
    target = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(target))

    def run(sync=False, dry_run=False):
        import_agent_command(SimpleNamespace(
            agent=None if sync else "claude-code", source=None if sync else str(source),
            sync=sync, dry_run=dry_run, overwrite=False, yes=True))

    run()
    imported_command = target / "skills/claude-code-commands/review/SKILL.md"
    imported_skill = target / "skills/claude-code-imports/review/SKILL.md"
    before = imported_command.read_text()
    manifest = load_sync_manifest(target)
    command.write_text("Review version two.\n")
    run(sync=True, dry_run=True)
    assert imported_command.read_text() == before
    assert load_sync_manifest(target) == manifest
    run(sync=True)
    assert "Review version two." in imported_command.read_text()
    assert "Ordinary version one." in imported_skill.read_text()

    # Refresh the other category, then protect a locally edited command.
    imported_command.write_text("My local command edits.\n")
    command.write_text("Review version three.\n")
    skill.write_text("---\nname: review\n---\nOrdinary version two.\n")
    run(sync=True)
    assert imported_command.read_text() == "My local command edits.\n"
    assert "Ordinary version two." in imported_skill.read_text()
    snapshot = {str(p.relative_to(target)): p.read_bytes()
                for p in target.rglob("*") if p.is_file()}
    run(sync=True)
    assert {str(p.relative_to(target)): p.read_bytes()
            for p in target.rglob("*") if p.is_file()} == snapshot


def test_command_sync_digest_excludes_redirected_sources(tmp_path):
    from hermes_cli.agent_import_sync import _iter_sync_files, compute_source_digest

    source = tmp_path / "claude"
    commands = source / "commands"
    commands.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "review.md"
    external.write_text("External version one.\n")
    link = commands / "review.md"
    link.symlink_to(external)
    baseline = compute_source_digest("claude-code", source)
    assert link not in list(_iter_sync_files("claude-code", source))
    external.write_text("External version two.\n")
    assert compute_source_digest("claude-code", source) == baseline
    link.unlink()
    commands.rmdir()
    commands.symlink_to(outside, target_is_directory=True)
    assert not any(p.parent == commands for p in _iter_sync_files("claude-code", source))
    assert compute_source_digest("claude-code", source) == baseline

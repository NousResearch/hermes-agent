"""Startup hint deduplication must follow the first non-empty context file."""

import pytest

from agent.prompt_builder import build_context_files_prompt
from agent.subdirectory_hints import SubdirectoryHintTracker


@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("empty", ["", "\ufeff \n\t"])
@pytest.mark.parametrize("primary,fallback", [
    ("AGENTS.override.md", "AGENTS.md"),
    ("AGENTS.md", "CLAUDE.md"),
    ("CLAUDE.md", ".cursorrules"),
])
def test_empty_hint_does_not_hide_loaded_fallback(tmp_path, rebind, empty, primary, fallback):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / primary).write_text(empty, encoding="utf-8")
    rules = "Project rules: use the shared validation command."
    (workspace / fallback).write_text(rules, encoding="utf-8")
    copied = workspace / "package"
    copied.mkdir()
    (copied / fallback).write_text(rules, encoding="utf-8")
    prompt = build_context_files_prompt(cwd=str(workspace), skip_soul=True)
    assert rules in prompt

    if rebind:
        previous = tmp_path / "previous"
        previous.mkdir()
        tracker = SubdirectoryHintTracker(str(previous))
        tracker.rebind_working_dir(str(workspace))
    else:
        tracker = SubdirectoryHintTracker(str(workspace))

    assert tracker.check_tool_call("read_file", {"path": str(copied / "source.py")}) is None
    assert build_context_files_prompt(cwd=str(workspace), skip_soul=True) == prompt


def test_nonempty_primary_still_wins_for_seed(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    primary = "Authoritative project instructions."
    fallback = "Separate package instructions."
    (workspace / "AGENTS.md").write_text(primary, encoding="utf-8")
    (workspace / "CLAUDE.md").write_text(fallback, encoding="utf-8")
    prompt = build_context_files_prompt(cwd=str(workspace), skip_soul=True)
    assert primary in prompt and fallback not in prompt
    tracker = SubdirectoryHintTracker(str(workspace))
    for name, rules in (("copied", primary), ("distinct", fallback)):
        package = workspace / name
        package.mkdir()
        (package / "AGENTS.md").write_text(rules, encoding="utf-8")
        hint = tracker.check_tool_call("read_file", {"path": str(package / "source.py")})
        if rules == primary:
            assert hint is None
        else:
            assert hint is not None and fallback in hint
        assert tracker.check_tool_call("read_file", {"path": str(package / "other.py")}) is None

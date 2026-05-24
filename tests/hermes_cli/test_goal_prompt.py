from pathlib import Path

from hermes_cli.goal_prompt import (
    extract_goal_prompt_text,
    goal_command_from_prompt_text,
    oneshot_goal_text_from_prompt_text,
    resolve_goal_prompt_path,
)


def test_extract_goal_prompt_text_prefers_fenced_prompt():
    raw = """# Runbook\n\nIntro text.\n\n```markdown\nContinue the project from NEXT_ACTIONS.md.\n```\n"""

    assert extract_goal_prompt_text(raw) == "Continue the project from NEXT_ACTIONS.md."


def test_goal_command_from_prompt_text_prefixes_goal_when_needed():
    assert goal_command_from_prompt_text("Continue safely") == "/goal Continue safely"


def test_goal_command_from_prompt_text_preserves_explicit_goal_command():
    assert goal_command_from_prompt_text("/goal Continue safely") == "/goal Continue safely"


def test_oneshot_goal_text_adds_chained_autonomy_instructions():
    text = oneshot_goal_text_from_prompt_text("Continue safely")

    assert text.startswith("Continue safely")
    assert "/goal_prompt_oneshot mode" in text
    assert "immediately re-read" in text
    assert "updated frontier" in text
    assert "/goal_prompt_oneshot continuation decision: CONTINUE" in text
    assert "STOP_FOR_OPERATOR" in text
    assert "COMPLETE" in text
    assert "private keys" in text


def test_goal_command_from_prompt_text_oneshot_strips_existing_goal_prefix():
    command = goal_command_from_prompt_text("/goal Continue safely", oneshot=True)

    assert command.startswith("/goal Continue safely")
    assert not command.startswith("/goal /goal")
    assert "/goal_prompt_oneshot mode" in command


def test_resolve_goal_prompt_path_finds_docs_runbooks(tmp_path: Path):
    prompt = tmp_path / "docs" / "runbooks" / "GOAL_PROMPT.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("prompt", encoding="utf-8")

    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)

    assert resolve_goal_prompt_path(cwd=nested) == prompt


def test_resolve_goal_prompt_path_can_treat_cwd_as_target_root(tmp_path: Path):
    prompt = tmp_path / "docs" / "runbooks" / "GOAL_PROMPT.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("prompt", encoding="utf-8")

    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)

    assert resolve_goal_prompt_path(cwd=nested, search_parents=False) == (
        nested / "docs" / "runbooks" / "GOAL_PROMPT.md"
    )


def test_resolve_goal_prompt_path_accepts_project_root_arg(tmp_path: Path):
    prompt = tmp_path / "docs" / "runbooks" / "GOAL_PROMPT.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("prompt", encoding="utf-8")

    assert resolve_goal_prompt_path(str(tmp_path), cwd=tmp_path) == prompt


def test_resolve_goal_prompt_path_accepts_lowercase_root_prompt(tmp_path: Path):
    prompt = tmp_path / "goal_prompt.md"
    prompt.write_text("prompt", encoding="utf-8")

    assert resolve_goal_prompt_path(cwd=tmp_path) == prompt

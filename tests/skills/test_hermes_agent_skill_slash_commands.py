from pathlib import Path


SKILL_PATH = Path("skills/autonomous-ai-agents/hermes-agent/SKILL.md")
COMMANDS_PATH = Path("hermes_cli/commands.py")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_skill_docs_do_not_list_phantom_skill_command():
    skill_text = _read(SKILL_PATH)
    commands_text = _read(COMMANDS_PATH)

    assert '/skill <name>' not in skill_text
    assert 'CommandDef("skill"' not in commands_text


def test_skill_docs_do_not_claim_q_exits_cli():
    skill_text = _read(SKILL_PATH)
    commands_text = _read(COMMANDS_PATH)

    assert '/quit (/exit, /q)' not in skill_text
    assert '/quit (/exit)' in skill_text
    assert 'CommandDef("quit", "Exit the CLI (use --delete to also remove session history)", "Exit",' in commands_text
    assert 'aliases=("exit",), args_hint="[--delete]")' in commands_text
    assert 'CommandDef("queue", "Queue a prompt for the next turn (doesn\'t interrupt)", "Session",' in commands_text
    assert 'aliases=("q",), args_hint="<prompt>")' in commands_text

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_real_cli(**kwargs):
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with patch.object(cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            cli_mod.__dict__, {"CLI_CONFIG": clean_config}
        ):
            return cli_mod.HermesCLI(**kwargs)


class _DummyCLI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = "session-123"
        self.system_prompt = "base prompt"
        self.preloaded_skills = []

    def show_banner(self):
        return None

    def show_tools(self):
        return None

    def show_toolsets(self):
        return None

    def run(self):
        return None


def test_main_applies_preloaded_skills_to_system_prompt(monkeypatch):
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("skill prompt", ["hermes-agent-dev", "github-auth"], []),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="hermes-agent-dev,github-auth", list_tools=True)

    cli_obj = created["cli"]
    assert cli_obj.system_prompt == "base prompt\n\nskill prompt"
    assert cli_obj.preloaded_skills == ["hermes-agent-dev", "github-auth"]


def test_main_raises_for_unknown_preloaded_skill(monkeypatch):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kwargs: _DummyCLI(**kwargs))
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", [], ["missing-skill"]),
    )

    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-skill"):
        cli_mod.main(skills="missing-skill", list_tools=True)


def test_show_banner_does_not_print_skills():
    """show_banner() no longer prints the activated skills line — it moved to run()."""
    cli_obj = _make_real_cli(compact=False)
    cli_obj.preloaded_skills = ["hermes-agent-dev", "github-auth"]
    cli_obj.console = MagicMock()

    with patch("cli.build_welcome_banner") as mock_banner, patch(
        "shutil.get_terminal_size", return_value=os.terminal_size((120, 40))
    ):
        cli_obj.show_banner()

    print_calls = [
        call.args[0]
        for call in cli_obj.console.print.call_args_list
        if call.args and isinstance(call.args[0], str)
    ]
    startup_lines = [line for line in print_calls if "Activated skills:" in line]
    assert len(startup_lines) == 0
    assert mock_banner.call_count == 1


# ── Task 11: preloaded-skills fence (Design Rule 6 + Rule 2) ─────────────
#
# `build_preloaded_skills_prompt` is allowed to render a skill's *content* into
# the session system prompt, but it must NEVER carry runtime defaults (no
# model swap, no reasoning_effort change) for preloaded skills.  A preloaded
# skill whose frontmatter declares `runtime_defaults` should be treated as if
# the field was absent.


def test_preloaded_skills_prompt_tuple_shape_unchanged(tmp_path):
    """build_preloaded_skills_prompt must keep its 3-element tuple contract."""
    from agent.skill_commands import build_preloaded_skills_prompt

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: test\n---\n\nBody.\n"
        )
        result = build_preloaded_skills_prompt(["my-skill"])

    # Tuple shape: (prompt_text, loaded_names, missing)
    assert isinstance(result, tuple)
    assert len(result) == 3
    prompt, loaded, missing = result
    assert isinstance(prompt, str)
    assert isinstance(loaded, list)
    assert isinstance(missing, list)
    assert "my-skill" in loaded
    assert missing == []


def test_preloaded_skill_with_runtime_defaults_stays_prompt_only(tmp_path):
    """A preloaded skill declaring runtime_defaults must NOT leak structured
    runtime state through build_preloaded_skills_prompt.  The function's
    return contract remains 3-tuple prompt-only — Design Rules 2 and 6."""
    from agent.skill_commands import build_preloaded_skills_prompt

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch(
            "agent.skill_utils._runtime_defaults_flag_enabled",
            return_value=True,
        ),
    ):
        skill_dir = tmp_path / "brainstorm"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: brainstorm\n"
            "description: test\n"
            "metadata:\n"
            "  hermes:\n"
            "    runtime_defaults:\n"
            "      reasoning_effort: low\n"
            "---\n"
            "\nBody.\n"
        )
        result = build_preloaded_skills_prompt(["brainstorm"])

    # Contract stays 3-tuple.  No 4th element carrying runtime defaults.
    assert len(result) == 3
    prompt, loaded, missing = result
    assert "brainstorm" in loaded

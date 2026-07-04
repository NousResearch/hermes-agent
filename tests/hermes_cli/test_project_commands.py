"""Tests for project-local slash commands (``.hermes/commands/*.md``)."""

import textwrap
from pathlib import Path

import pytest

from hermes_cli.project_commands import (
    ProjectCommand,
    _parse_command_file,
    load_project_commands,
    resolve_project_command,
    render_command,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# ── _parse_command_file ──────────────────────────────────────────────


class TestParseCommandFile:
    def test_basic_valid(self, tmp_path):
        _write_md(tmp_path / "hello.md", """\
            ---
            description: Say hello to the user.
            category: 项目命令
            args:
              - name: target
                description: Who to greet
                default: world
            ---
            # Hello

            Say hello to {{target}}!
            """)
        cmd = _parse_command_file(tmp_path / "hello.md")
        assert cmd is not None
        assert cmd.name == "hello"
        assert cmd.description == "Say hello to the user."
        assert cmd.category == "项目命令"
        assert len(cmd.args) == 1
        assert cmd.args[0]["name"] == "target"
        assert cmd.args[0]["default"] == "world"
        assert cmd.override is False
        assert "{{target}}" in cmd.template

    def test_missing_description(self, tmp_path):
        _write_md(tmp_path / "no_desc.md", """\
            ---
            category: test
            ---
            body
            """)
        assert _parse_command_file(tmp_path / "no_desc.md") is None

    def test_invalid_name(self, tmp_path):
        _write_md(tmp_path / "bad name!.md", """\
            ---
            description: Invalid name test.
            ---
            body
            """)
        assert _parse_command_file(tmp_path / "bad name!.md") is None

    def test_override_flag(self, tmp_path):
        _write_md(tmp_path / "yolo.md", """\
            ---
            description: Override the built-in yolo command.
            override: true
            ---
            Overridden yolo!
            """)
        cmd = _parse_command_file(tmp_path / "yolo.md")
        assert cmd is not None
        assert cmd.override is True

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        assert _parse_command_file(p) is None

    def test_no_frontmatter(self, tmp_path):
        _write_md(tmp_path / "nofm.md", """\
            Just some text without frontmatter.
            """)
        # No frontmatter → no description → None
        assert _parse_command_file(tmp_path / "nofm.md") is None

    def test_missing_file(self, tmp_path):
        assert _parse_command_file(tmp_path / "nonexistent.md") is None

    def test_file_too_large(self, tmp_path, monkeypatch):
        import hermes_cli.project_commands as pc
        monkeypatch.setattr(pc, "_MAX_COMMAND_FILE_BYTES", 100)
        _write_md(tmp_path / "big.md", f"""\
            ---
            description: A very large command.
            ---
            {'x' * 200}
            """)
        assert _parse_command_file(tmp_path / "big.md") is None


# ── render_command ───────────────────────────────────────────────────


class TestRenderCommand:
    def test_substitutes_variables(self):
        cmd = ProjectCommand(
            name="greet",
            description="Greet someone.",
            args=[
                {"name": "who", "description": "Who to greet", "default": "world"},
            ],
            template="Hello, {{who}}!",
        )
        result = render_command(cmd, {"who": "Alice"})
        assert result == "Hello, Alice!"

    def test_fallback_to_default(self):
        cmd = ProjectCommand(
            name="greet",
            description="Greet someone.",
            args=[
                {"name": "who", "description": "Who to greet", "default": "world"},
            ],
            template="Hello, {{who}}!",
        )
        result = render_command(cmd, {})
        assert result == "Hello, world!"

    def test_user_arg_overrides_default(self):
        cmd = ProjectCommand(
            name="greet",
            description="Greet someone.",
            args=[
                {"name": "who", "description": "Who to greet", "default": "world"},
            ],
            template="Hello, {{who}}!",
        )
        result = render_command(cmd, {"who": "Bob"})
        assert result == "Hello, Bob!"

    def test_unmatched_placeholder_left_as_is(self):
        cmd = ProjectCommand(
            name="query",
            description="Query something.",
            template="Looking up {{thing}}...",
        )
        result = render_command(cmd, {})
        assert result == "Looking up {{thing}}..."

    def test_multiple_variables(self):
        cmd = ProjectCommand(
            name="build",
            description="Build something.",
            args=[
                {"name": "target", "description": "Build target", "default": "all"},
                {"name": "config", "description": "Build config", "default": "Debug"},
            ],
            template="Building {{target}} with {{config}}...",
        )
        result = render_command(cmd, {"target": "server"})
        assert result == "Building server with Debug..."


# ── load_project_commands (mocked) ───────────────────────────────────


class TestLoadProjectCommands:
    def test_empty_directory(self, tmp_path, monkeypatch):
        import hermes_cli.project_commands as pc
        monkeypatch.setattr(pc, "_get_project_commands_dir", lambda: tmp_path)
        # Reset cache
        monkeypatch.setattr(pc, "_commands_cache", None)
        monkeypatch.setattr(pc, "_commands_cache_dir", None)

        commands = load_project_commands()
        assert commands == []

    def test_loads_valid_commands(self, tmp_path, monkeypatch):
        import hermes_cli.project_commands as pc

        _write_md(tmp_path / "greet.md", """\
            ---
            description: Greet someone.
            ---
            Hello!
            """)
        _write_md(tmp_path / "build.md", """\
            ---
            description: Build the project.
            args:
              - name: target
                description: What to build
            ---
            Building {{target}}...
            """)

        monkeypatch.setattr(pc, "_get_project_commands_dir", lambda: tmp_path)
        monkeypatch.setattr(pc, "_commands_cache", None)
        monkeypatch.setattr(pc, "_commands_cache_dir", None)

        commands = load_project_commands()
        assert len(commands) == 2
        names = {c.name for c in commands}
        assert names == {"greet", "build"}

    def test_skips_invalid_commands(self, tmp_path, monkeypatch):
        import hermes_cli.project_commands as pc

        _write_md(tmp_path / "valid.md", """\
            ---
            description: A valid command.
            ---
            Body
            """)
        # Missing description — should be skipped
        _write_md(tmp_path / "invalid.md", """\
            ---
            category: test
            ---
            Body
            """)

        monkeypatch.setattr(pc, "_get_project_commands_dir", lambda: tmp_path)
        monkeypatch.setattr(pc, "_commands_cache", None)
        monkeypatch.setattr(pc, "_commands_cache_dir", None)

        commands = load_project_commands()
        assert len(commands) == 1
        assert commands[0].name == "valid"


# ── resolve_project_command ──────────────────────────────────────────


class TestResolveProjectCommand:
    def test_found(self, tmp_path, monkeypatch):
        import hermes_cli.project_commands as pc

        _write_md(tmp_path / "greet.md", """\
            ---
            description: Greet someone.
            ---
            Hi!
            """)

        monkeypatch.setattr(pc, "_get_project_commands_dir", lambda: tmp_path)
        monkeypatch.setattr(pc, "_commands_cache", None)
        monkeypatch.setattr(pc, "_commands_cache_dir", None)

        cmd = resolve_project_command("greet")
        assert cmd is not None
        assert cmd.name == "greet"

    def test_not_found(self, tmp_path, monkeypatch):
        import hermes_cli.project_commands as pc

        monkeypatch.setattr(pc, "_get_project_commands_dir", lambda: tmp_path)
        monkeypatch.setattr(pc, "_commands_cache", None)
        monkeypatch.setattr(pc, "_commands_cache_dir", None)

        assert resolve_project_command("nonexistent") is None


# ── No hermes project directory ──────────────────────────────────────


class TestNoProjectDir:
    def test_load_returns_empty(self, monkeypatch):
        import hermes_cli.project_commands as pc

        monkeypatch.setattr(pc, "_get_project_commands_dir", lambda: None)
        monkeypatch.setattr(pc, "_commands_cache", None)
        monkeypatch.setattr(pc, "_commands_cache_dir", None)

        assert load_project_commands() == []

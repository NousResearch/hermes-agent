"""Tests for /init — generate or update AGENTS.md from a project scan.

Covers the shared prompt builder (hermes_cli.init_command.build_init_prompt)
and the slash-command registry wiring. /init has no engine and no model tool:
it builds a guidance-laden prompt that the live agent runs as a normal turn
(the /learn pattern), so these are the load-bearing behavior contracts.
"""

from hermes_cli.init_command import (
    _QUALITY_BAR,
    build_init_prompt,
    build_init_prompt_for_cwd,
)


class TestBuildInitPrompt:



    def test_merge_not_overwrite_when_existing_file_passed(self):
        existing = "# My Project\n\nAlways run `make lint` before committing.\n"
        prompt = build_init_prompt("/tmp/proj", existing_file=existing)
        low = prompt.lower()
        # Update mode, with explicit merge-not-overwrite discipline.
        assert "UPDATE the existing AGENTS.md" in prompt
        assert "merge" in low
        assert "do not overwrite" in low or "not overwrite" in low
        assert "preserve" in low
        # And it carries the current content so the agent can merge.
        assert existing.strip() in prompt


    def test_includes_extra_notes_verbatim(self):
        notes = "focus on the test setup, and mention the flaky e2e suite"
        prompt = build_init_prompt("/tmp/proj", extra=notes)
        assert notes in prompt





class TestBuildInitPromptForCwd:

    def test_reads_existing_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "# Existing\n\nRun `tox -e py311`.\n", encoding="utf-8"
        )
        prompt = build_init_prompt_for_cwd(cwd=str(tmp_path))
        assert "UPDATE the existing AGENTS.md" in prompt
        assert "Run `tox -e py311`." in prompt

    def test_passes_extra_through(self, tmp_path):
        prompt = build_init_prompt_for_cwd(cwd=str(tmp_path), extra="keep it short")
        assert "keep it short" in prompt


class TestSessionActiveDirectory:
    """/init must target the session's ACTIVE directory, not the process cwd.

    Regression: on the desktop app the process launches from the home
    directory, and /init with no explicit cwd scanned/updated the HOME's
    AGENTS.md instead of the workspace attached to the session. The session's
    live directory is the terminal tool's per-session cwd record (seeded when
    a workspace attaches, updated after every `cd`).
    """

    def test_session_cwd_record_wins_over_process_cwd(self, tmp_path, monkeypatch):
        from tools.terminal_tool import clear_session_cwd, record_session_cwd

        workspace = tmp_path / "proj"
        workspace.mkdir()
        launch_dir = tmp_path / "home"
        launch_dir.mkdir()
        monkeypatch.chdir(launch_dir)  # desktop app launches from HOME

        record_session_cwd("desktop:123", str(workspace))
        try:
            prompt = build_init_prompt_for_cwd(session_key="desktop:123")
            assert f"for the project at: {workspace}" in prompt
            assert str(launch_dir) not in prompt
        finally:
            clear_session_cwd("desktop:123")

    def test_stale_record_falls_back_to_process_cwd(self, tmp_path, monkeypatch):
        from tools.terminal_tool import clear_session_cwd, record_session_cwd

        launch_dir = tmp_path / "home"
        launch_dir.mkdir()
        monkeypatch.chdir(launch_dir)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        record_session_cwd("desktop:123", str(tmp_path / "deleted-project"))
        try:
            prompt = build_init_prompt_for_cwd(session_key="desktop:123")
            assert f"for the project at: {launch_dir}" in prompt
        finally:
            clear_session_cwd("desktop:123")

    def test_explicit_cwd_wins_over_session_record(self, tmp_path):
        from tools.terminal_tool import clear_session_cwd, record_session_cwd

        record_session_cwd("desktop:123", str(tmp_path / "other"))
        try:
            prompt = build_init_prompt_for_cwd(
                cwd=str(tmp_path), session_key="desktop:123"
            )
            assert f"for the project at: {tmp_path}" in prompt
        finally:
            clear_session_cwd("desktop:123")

    def test_ambient_session_key_used_when_none(self, tmp_path, monkeypatch):
        # Dispatch surfaces that don't pass an explicit key (e.g. a CLI
        # inheriting a bound session env) resolve the key from
        # HERMES_SESSION_KEY.
        from gateway.session_context import set_session_vars
        from tools.terminal_tool import clear_session_cwd, record_session_cwd

        workspace = tmp_path / "proj"
        workspace.mkdir()
        launch_dir = tmp_path / "home"
        launch_dir.mkdir()
        monkeypatch.chdir(launch_dir)

        record_session_cwd("gateway:999", str(workspace))
        tokens = set_session_vars(session_key="gateway:999")
        try:
            prompt = build_init_prompt_for_cwd()
            assert f"for the project at: {workspace}" in prompt
        finally:
            clear_session_cwd("gateway:999")

    def test_stale_explicit_cwd_falls_through_to_the_session_record(self, tmp_path):
        # The guard lives in the builder so every surface shares it: a ``cwd`` naming a
        # directory that is gone (deleted project, removed linked worktree) must not win —
        # write_file would otherwise recreate the directory and write AGENTS.md into it.
        from tools.terminal_tool import clear_session_cwd, record_session_cwd

        workspace = tmp_path / "proj"
        workspace.mkdir()
        gone = tmp_path / "deleted-project"
        gone.mkdir()
        gone.rmdir()

        record_session_cwd("desktop:123", str(workspace))
        try:
            prompt = build_init_prompt_for_cwd(cwd=str(gone), session_key="desktop:123")
            assert f"for the project at: {workspace}" in prompt
            assert str(gone) not in prompt
        finally:
            clear_session_cwd("desktop:123")


class TestInitRegistryWiring:
    def test_init_is_registered_and_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("init")
        assert cmd is not None
        assert cmd.name == "init"


    def test_init_works_on_the_gateway(self):
        # /init is a both-surfaces command like /learn, not CLI-only.
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "init" in GATEWAY_KNOWN_COMMANDS


"""Compaction ALWAYS rebuilds the system prompt from the live builder (#95681).

The old keep-prompt containment branch restored the stored bytes whenever the
reloaded memory blocks were embedded — so prompt-builder changes (guidance
diets, renames, new blocks) never reached long-lived sessions (Bot Mode
forever-chats, gateway channels). New contract:

1. builder output byte-equal  -> keep the ORIGINAL string object (identity
   preserved for KV/prefix caches keyed on it)
2. builder output differs     -> the rebuilt prompt wins, logged
3. plugin sections re-render at the same boundary; a RAISING plugin falls
   back to its last good bytes (fail-open), never silently vanishes
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import invalidate_system_prompt


def _agent(**over):
    base = dict(
        _cached_system_prompt="OLD PROMPT",
        _cached_system_prompt_static="OLD",
        _memory_store=None,
        _memory_manager=None,
        provider="",
        model="",
        platform="",
        _memory_enabled=False,
        _user_profile_enabled=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestInvalidateClearsPluginFreeze(unittest.TestCase):
    def test_invalidate_stashes_and_clears_plugin_snapshot(self):
        agent = _agent()
        agent._plugin_system_prompt_sections_snapshot = ("frozen-section",)
        invalidate_system_prompt(agent)
        self.assertFalse(hasattr(agent, "_plugin_system_prompt_sections_snapshot"))
        self.assertEqual(agent._plugin_system_prompt_sections_previous, ("frozen-section",))
        self.assertIsNone(agent._cached_system_prompt)



class TestPluginRerenderFailOpen(unittest.TestCase):
    def test_raising_plugin_render_falls_back_to_previous_bytes(self):
        from agent.system_prompt import _frozen_plugin_prompt_sections

        agent = _agent(_cached_system_prompt=None)
        agent._plugin_system_prompt_sections_previous = ("last-good",)
        with patch("hermes_cli.plugins.render_system_prompt_sections",
                   side_effect=RuntimeError("plugin exploded")):
            rendered = _frozen_plugin_prompt_sections(agent)
        self.assertEqual(rendered, ("last-good",))

    def test_raising_plugin_render_without_previous_is_empty(self):
        from agent.system_prompt import _frozen_plugin_prompt_sections

        agent = _agent(_cached_system_prompt=None)
        with patch("hermes_cli.plugins.render_system_prompt_sections",
                   side_effect=RuntimeError("plugin exploded")):
            rendered = _frozen_plugin_prompt_sections(agent)
        self.assertEqual(rendered, ())


def _init_repo(path, first_commit):
    import subprocess
    path.mkdir()
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "config", "core.autocrlf", "false"],
    ):
        subprocess.run(cmd, cwd=path, check=True)
    (path / "main.py").write_text("print(1)\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", first_commit], cwd=path, check=True)
    return path




class TestWorkspaceSnapshotPinnedAcrossCompaction(unittest.TestCase):
    """Compaction rebuilds must not invalidate the prefix at the workspace snapshot (#103326)."""

    def test_workspace_snapshot_replayed_across_rebuilds_when_repo_mutates(self):
        import tempfile, shutil, subprocess
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-ws-"))
        try:
            repo = _init_repo(tmp / "proj", "init commit")

            agent = _agent(
                load_soul_identity=False,
                skip_context_files=True,
                valid_tool_names={"terminal", "file_write"},
                platform="cli",
                model="gpt-4o",
                _memory_enabled=False,
                _user_profile_enabled=False,
                _task_completion_guidance=False,
                _parallel_tool_call_guidance=False,
                _tool_use_enforcement=False,
                _execution_guidance=False,
                _environment_probe=False,
                _bot_mode_protocol=False,
                _kanban_worker_guidance="",
                pass_session_id=False,
                session_id="s1",
                _emit_status=lambda *a, **k: None,
            )

            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):

                # First build: pins the snapshot
                p1 = build_system_prompt(agent)
                self.assertIn("Workspace (snapshot at session start", p1)

                # Now repo mutates (agent touched and committed new files)
                (repo / "new_file.py").write_text("print(2)\n")
                subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
                subprocess.run(["git", "commit", "-qm", "second commit"], cwd=repo, check=True)
                (repo / "untracked.txt").write_text("wip\n")

                # Invalidate prompt (as happens during context compression)
                invalidate_system_prompt(agent)

                # Second build: must replay pinned snapshot without re-probing git
                p2 = build_system_prompt(agent)
                self.assertEqual(p1, p2, "Prompt must remain byte-identical despite repo mutations")
                self.assertNotIn("second commit", p2, "Rebuilt prompt must not leak mutated git log")

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_workspace_snapshot_reprobes_when_cwd_changes(self):
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-cwd-"))
        try:
            repo1 = _init_repo(tmp / "r1", "init r1")
            repo2 = _init_repo(tmp / "r2", "init r2")

            agent = _agent(
                load_soul_identity=False,
                skip_context_files=True,
                valid_tool_names={"terminal"},
                platform="cli",
                model="gpt-4o",
                _memory_enabled=False,
                _user_profile_enabled=False,
                _task_completion_guidance=False,
                _parallel_tool_call_guidance=False,
                _tool_use_enforcement=False,
                _execution_guidance=False,
                _environment_probe=False,
                _bot_mode_protocol=False,
                _kanban_worker_guidance="",
                pass_session_id=False,
                session_id="s1",
                _emit_status=lambda *a, **k: None,
            )

            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo1):
                p1 = build_system_prompt(agent)
                self.assertIn(f"init {repo1.name}", p1)

            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo2):
                p2 = build_system_prompt(agent)
                self.assertIn(f"init {repo2.name}", p2)

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_session_boundary_drops_the_pin_so_a_new_session_resnapshots(self):
        """A /new, /resume or /branch reuses the AIAgent; the next session must see the live repo."""
        import tempfile, shutil, subprocess
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt
        from run_agent import AIAgent

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-boundary-"))
        try:
            repo = _init_repo(tmp / "proj", "init commit")
            agent = _agent(
                load_soul_identity=False, skip_context_files=True, valid_tool_names={"terminal"},
                platform="cli", model="gpt-4o", _task_completion_guidance=False,
                _parallel_tool_call_guidance=False, _tool_use_enforcement=False, _execution_guidance=False,
                _environment_probe=False, _bot_mode_protocol=False, _kanban_worker_guidance="",
                pass_session_id=False, session_id="s1", _emit_status=lambda *a, **k: None,
                _frozen_workspace_snapshot=None, context_compressor=None, _session_db=None,
                _transition_context_engine_session=lambda **kw: None,
            )
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                build_system_prompt(agent)
                subprocess.run(["git", "commit", "-qm", "second commit", "--allow-empty"], cwd=repo, check=True)
                # The CLI session boundary (cli_session_mixin.new_session) on the same agent object.
                AIAgent.reset_session_state(agent)
                invalidate_system_prompt(agent)
                self.assertIn("second commit", build_system_prompt(agent))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _pin_agent(self, **over):
        return _agent(
            load_soul_identity=False, skip_context_files=True, valid_tool_names={"terminal"},
            platform="cli", model="gpt-4o", _task_completion_guidance=False,
            _parallel_tool_call_guidance=False, _tool_use_enforcement=False, _execution_guidance=False,
            _environment_probe=False, _bot_mode_protocol=False, _kanban_worker_guidance="",
            pass_session_id=False, session_id="s1", _emit_status=lambda *a, **k: None, **over,
        )

    def test_binding_the_launch_dir_explicitly_replays_the_pin(self):
        """CLI-shaped first build (no cwd bound -> launch dir), then TUI /compress binds that same dir:
        one workspace, so the rebuild replays the session-start snapshot."""
        import os, tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-bind-"))
        old_cwd = os.getcwd()
        try:
            repo = _init_repo(tmp / "proj", "init commit")
            os.chdir(repo)
            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"):
                with patch("agent.system_prompt.resolve_context_cwd", return_value=None):
                    p1 = build_system_prompt(agent)
                self.assertIn("Status: clean", p1)
                (repo / "untracked.txt").write_text("wip\n")
                invalidate_system_prompt(agent)
                with patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                    self.assertEqual(build_system_prompt(agent), p1)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_agent_that_did_not_build_the_prompt_replays_the_persisted_snapshot(self):
        """Resume / gateway / TUI shape: a fresh agent rebuilds (compaction, a first /compress) after
        the repo moved and replays the snapshot its session row already holds — unless that prompt
        was taken in another cwd."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-resume-"))
        try:
            repo, other = _init_repo(tmp / "proj", "init commit"), _init_repo(tmp / "other", "init other")

            def env(cwd):
                return patch("agent.prompt_builder.build_environment_hints",
                             return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {cwd}")

            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                stored = build_system_prompt(self._pin_agent())
                self.assertIn("Status: clean", stored)
                (repo / "untracked.txt").write_text("wip\n")
                db = SimpleNamespace(get_session=lambda sid: {"system_prompt": stored})
                resumed = self._pin_agent(_cached_system_prompt=None, _session_db=db)
                self.assertEqual(build_system_prompt(resumed), stored)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(other), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=other):
                moved = self._pin_agent(_cached_system_prompt=None, _session_db=db)
                self.assertIn(f"- Root: {other}", build_system_prompt(moved))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_persisted_prompt_without_a_snapshot_does_not_pin_an_empty_one(self):
        """A session row built where no workspace block was emitted (a messaging surface) and
        resumed in the same repo must capture a real snapshot, not pin "no workspace" for good."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-empty-"))
        try:
            repo = _init_repo(tmp / "proj", "init commit")
            stored = f"Host: x\nUser home directory: /h\nCurrent working directory: {repo}\n\nBODY"
            db = SimpleNamespace(get_session=lambda sid: {"system_prompt": stored})
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints",
                       return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {repo}"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                resumed = self._pin_agent(_cached_system_prompt=None, _session_db=db)
                self.assertIn(f"- Root: {repo}", build_system_prompt(resumed))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

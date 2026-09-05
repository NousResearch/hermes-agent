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

    def test_invalidate_without_snapshot_is_noop_for_plugins(self):
        agent = _agent()
        invalidate_system_prompt(agent)
        self.assertFalse(hasattr(agent, "_plugin_system_prompt_sections_snapshot"))


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


class TestCommitAlwaysRebuilds(unittest.TestCase):
    """Source-level contract pins for the commit-site semantics."""

    def _src(self):
        import inspect
        from agent import conversation_compression as cc
        return inspect.getsource(cc)

    def test_keep_prompt_branch_requires_byte_equality(self):
        src = self._src()
        i = src.find("rebuilt_system_prompt = agent._build_system_prompt(")
        self.assertGreater(i, 0, "commit site must always run the live builder")
        window = src[i:i + 900]
        self.assertIn("rebuilt_system_prompt == cached_system_prompt", window,
                      "keep-prompt must be gated on BYTE EQUALITY of the "
                      "rebuilt output, not on memory containment")
        self.assertNotIn("_cached_prompt_reflects_builtin_memory(agent, cached_system_prompt)",
                         window,
                         "the containment keep-prompt gate must not return")

    def test_drift_rebuild_is_logged(self):
        src = self._src()
        self.assertIn("Compaction rebuilt a drifted system prompt", src)


class TestWorkspaceSnapshotPinnedAcrossCompaction(unittest.TestCase):
    """Compaction rebuilds must not invalidate the prefix at the workspace snapshot (#103326)."""

    def test_workspace_snapshot_replayed_across_rebuilds_when_repo_mutates(self):
        import tempfile, shutil, subprocess
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-ws-"))
        try:
            repo = tmp / "proj"
            repo.mkdir()
            for cmd in (
                ["git", "init", "-q", "-b", "main"],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"],
                ["git", "config", "core.autocrlf", "false"],
            ):
                subprocess.run(cmd, cwd=repo, check=True)
            (repo / "main.py").write_text("print(1)\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "init commit"], cwd=repo, check=True)

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
                self.assertIsNotNone(agent._frozen_workspace_snapshot)
                self.assertEqual(agent._frozen_workspace_snapshot[0], str(repo))

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
        import tempfile, shutil, subprocess
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-cwd-"))
        try:
            repo1 = tmp / "r1"; repo1.mkdir()
            repo2 = tmp / "r2"; repo2.mkdir()
            for r in (repo1, repo2):
                for cmd in (
                    ["git", "init", "-q", "-b", "main"],
                    ["git", "config", "user.email", "t@t"],
                    ["git", "config", "user.name", "t"],
                    ["git", "config", "core.autocrlf", "false"],
                ):
                    subprocess.run(cmd, cwd=r, check=True)
                (r / "main.py").write_text("print(1)\n")
                subprocess.run(["git", "add", "-A"], cwd=r, check=True)
                subprocess.run(["git", "commit", "-qm", f"init {r.name}"], cwd=r, check=True)

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
                self.assertEqual(agent._frozen_workspace_snapshot[0], str(repo2))

        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

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

import pytest

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

    def test_symlink_spelling_of_the_launch_dir_replays_the_pin(self):
        """The macOS shape on any POSIX host: os.getcwd() reports the physical dir while the bound
        cwd arrives through a symlink (/var -> /private/var). Same workspace, so the pin must hit."""
        import os, sys, tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        if sys.platform == "win32":
            self.skipTest("directory symlinks need privileges on Windows")
        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-symlink-"))
        old_cwd = os.getcwd()
        try:
            repo = _init_repo(tmp / "proj", "init commit")
            link = tmp / "link"
            link.symlink_to(repo, target_is_directory=True)
            os.chdir(repo)  # os.getcwd() is the physical spelling
            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"):
                with patch("agent.system_prompt.resolve_context_cwd", return_value=None):
                    p1 = build_system_prompt(agent)
                self.assertIn("Status: clean", p1)
                (repo / "untracked.txt").write_text("wip\n")
                invalidate_system_prompt(agent)
                with patch("agent.system_prompt.resolve_context_cwd", return_value=link):
                    self.assertEqual(build_system_prompt(agent), p1)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.platforms("macos")
    def test_case_alias_spelling_of_the_launch_dir_replays_the_pin(self):
        """macOS default APFS is case-insensitive and case-preserving: resolve() follows the
        symlink yet keeps the caller's casing, so a directory bound under a differently cased
        spelling is the same workspace and the pin must hit. The platforms marker brings the
        file into the macOS lane (an unmarked file runs on no OS lane at all); the samefile
        self-guard below still skips on a case-sensitive volume, where the alias genuinely
        is a second directory."""
        import os, tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-case-alias-"))
        old_cwd = os.getcwd()
        try:
            repo = _init_repo(tmp / "Repo", "init commit")
            alias = repo.with_name(repo.name.lower())
            try:
                alias_same = alias.samefile(repo)
            except OSError:
                alias_same = False  # case-sensitive filesystem: the alias does not exist
            if not alias_same:
                self.skipTest("case-sensitive filesystem: the alias is a different directory")
            os.chdir(repo)  # os.getcwd() reports the on-disk casing
            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"):
                with patch("agent.system_prompt.resolve_context_cwd", return_value=None):
                    p1 = build_system_prompt(agent)
                self.assertIn("Status: clean", p1)
                (repo / "untracked.txt").write_text("wip\n")
                invalidate_system_prompt(agent)
                with patch("agent.system_prompt.resolve_context_cwd", return_value=alias):
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
                self.assertIn(f"- Root: {other.resolve()}", build_system_prompt(moved))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_loop_cwd_spelling_in_persisted_bytes_cannot_drop_the_coding_block(self):
        """A symlink loop in persisted session bytes (the Current working directory hint or a
        ``- Root:`` line) makes Path.resolve raise RuntimeError on Python <= 3.12; the pin
        seams must contain it instead of letting it reach the blanket handler that would
        silently drop the whole coding block for that build."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import _same_live_dir, build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-loop-"))
        try:
            loop = tmp / "loop"
            loop.symlink_to(loop)  # symlink pointing at itself
            # Contained at the comparison seam on every supported interpreter.
            self.assertIs(_same_live_dir(str(loop), str(tmp)), False)

            repo, other = _init_repo(tmp / "proj", "init commit"), _init_repo(tmp / "other", "init other")

            def env(cwd):
                return patch("agent.prompt_builder.build_environment_hints",
                             return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {cwd}")

            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                stored = build_system_prompt(self._pin_agent())
                self.assertIn("Status: clean", stored)
            # A persisted prompt whose cwd hint spells a symlink loop: seeding must neither
            # adopt nor raise — the rebuilt prompt keeps its coding block.
            poisoned = stored.replace(f"Current working directory: {repo}",
                                      f"Current working directory: {loop}")
            db = SimpleNamespace(get_session=lambda sid: {"system_prompt": poisoned})
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(other), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=other):
                self.assertIn("- Root: ", build_system_prompt(
                    self._pin_agent(_cached_system_prompt=None, _session_db=db)))
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
                self.assertIn(f"- Root: {repo.resolve()}", build_system_prompt(resumed))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _rebind(self, captured, other):
        """Rename the captured directory and reuse its pathname as a symlink to `other`."""
        import sys
        if sys.platform == "win32":
            self.skipTest("directory symlinks need privileges on Windows")
        saved = captured.with_name(captured.name + "-saved")
        captured.rename(saved)
        captured.symlink_to(other, target_is_directory=True)
        return saved

    def test_rebound_path_cannot_replay_the_captured_snapshot(self):
        """Capturing in physical A, renaming it and reusing the pathname A as a symlink to
        independent repo B, then binding B: today's resolution of the captured key lands on
        B, but the snapshot was taken in A-saved — the build must probe B instead of
        replaying A's frozen bytes under B's name."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-rebind-"))
        try:
            captured = _init_repo(tmp / "captured", "init captured")
            other = _init_repo(tmp / "other", "init other")
            captured_key = captured.resolve()
            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=captured):
                self.assertIn("Status: clean", build_system_prompt(agent))
            self._rebind(captured, other)
            invalidate_system_prompt(agent)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=other):
                rebuilt = build_system_prompt(agent)
            self.assertIn(f"- Root: {other.resolve()}", rebuilt)
            self.assertNotIn(f"- Root: {captured_key}", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rebound_path_cannot_adopt_the_persisted_snapshot(self):
        """Fresh agent, session row built in physical A, A rebound to B, agent bound to B:
        the persisted ``- Root:`` line was git's physical spelling at capture — a resolve()
        that now traverses a symlink proves the directory was rebound, so the bytes must
        not be adopted for B."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-adopt-rebind-"))
        try:
            captured = _init_repo(tmp / "captured", "init captured")
            other = _init_repo(tmp / "other", "init other")
            captured_key = captured.resolve()

            def env(cwd):
                return patch("agent.prompt_builder.build_environment_hints",
                             return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {cwd}")

            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(captured), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=captured):
                stored = build_system_prompt(self._pin_agent())
                self.assertIn("Status: clean", stored)
            self._rebind(captured, other)
            db = SimpleNamespace(get_session=lambda sid: {"system_prompt": stored})
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(other), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=other):
                rebuilt = build_system_prompt(self._pin_agent(_cached_system_prompt=None, _session_db=db))
            self.assertIn(f"- Root: {other.resolve()}", rebuilt)
            self.assertNotIn(f"- Root: {captured_key}", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rebound_nonworkspace_path_must_not_pin_its_emptiness_onto_the_new_repo(self):
        """Capturing "no workspace" in a plain directory, then reusing its pathname as a
        symlink to a git repository: an empty pin must not suppress the new directory's
        snapshot — the build emits B's workspace block."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-empty-rebind-"))
        try:
            plain = tmp / "plain"
            plain.mkdir()
            other = _init_repo(tmp / "other", "init other")
            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=plain):
                build_system_prompt(agent)
                self.assertEqual(agent._frozen_workspace_snapshot[1], "")
            self._rebind(plain, other)
            invalidate_system_prompt(agent)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=other):
                self.assertIn(f"- Root: {other.resolve()}", build_system_prompt(agent))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pin_is_not_published_when_the_directory_is_replaced_during_collection(self):
        """Bytes and identity must be paired at collection: a replacement that lands between
        the workspace producer's snapshot and a post-hoc stat must not publish a pin that
        pairs one directory's bytes with another's identity — the pin stays unseeded and
        the next build probes fresh."""
        import tempfile, shutil
        from pathlib import Path
        import agent.coding_context as cc
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-pairing-"))
        try:
            captured = _init_repo(tmp / "captured", "init captured")
            other = _init_repo(tmp / "other", "init other")
            captured_key = captured.resolve()
            real_parts = cc.coding_system_prompt_parts
            state = {"armed": True}

            def renderer(**kwargs):
                parts = real_parts(**kwargs)
                if state["armed"] and parts[1]:
                    state["armed"] = False
                    self._rebind(captured, other)  # replacement lands after collection
                return parts

            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=captured), \
                 patch("agent.coding_context.coding_system_prompt_parts", side_effect=renderer):
                build_system_prompt(agent)
            # No publishable pairing: the stat that would certify the collected bytes
            # describes the replacement, not the directory they were taken in.
            self.assertIsNone(getattr(agent, "_frozen_workspace_snapshot", None))
            invalidate_system_prompt(agent)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=other):
                rebuilt = build_system_prompt(agent)
            self.assertIn(f"- Root: {other.resolve()}\n", rebuilt)
            self.assertNotIn(f"- Root: {captured_key}\n", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_persisted_adoption_requires_the_owner_not_just_an_ancestor(self):
        """Capture beneath repository A's root (cwd = A/src), rebind that subdirectory to an
        independent nested repository B, resume bound to B: A's root is still a physical
        ancestor of B, but it is not the workspace that owns B — the persisted snapshot
        must not be adopted, and the resumed agent must probe B."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-owner-"))
        try:
            import subprocess
            outer = _init_repo(tmp / "outer", "init outer")
            src, inner = outer / "src", outer / "inner"
            src.mkdir()
            inner.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=inner, check=True)
            subprocess.run(["git", "-C", str(inner), "config", "user.email", "t@t"], check=True)
            subprocess.run(["git", "-C", str(inner), "config", "user.name", "t"], check=True)
            (inner / "inner.py").write_text("print(2)\n")
            subprocess.run(["git", "-C", str(inner), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(inner), "commit", "-qm", "init inner"], check=True)

            def env(cwd):
                return patch("agent.prompt_builder.build_environment_hints",
                             return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {cwd}")

            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(src), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=src):
                stored = build_system_prompt(self._pin_agent())
                self.assertIn(f"- Root: {outer.resolve()}", stored)
            self._rebind(src, inner)
            # Control: an unchanged subdirectory resume still replays the captured snapshot
            # (cwd src-saved is the same workspace the snapshot was taken in).
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(outer / "src-saved"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=outer / "src-saved"):
                same = build_system_prompt(self._pin_agent(_cached_system_prompt=None,
                                                           _session_db=SimpleNamespace(
                                                               get_session=lambda sid: {"system_prompt": stored})))
                self.assertIn(f"- Root: {outer.resolve()}\n", same)
            # The rebind: the persisted root is an ancestor of the new cwd, not its owner.
            db = SimpleNamespace(get_session=lambda sid: {"system_prompt": stored})
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(inner), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=inner):
                rebuilt = build_system_prompt(self._pin_agent(_cached_system_prompt=None, _session_db=db))
            self.assertIn(f"- Root: {inner.resolve()}\n", rebuilt)
            self.assertNotIn(f"- Root: {outer.resolve()}\n", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_same_path_replacement_cannot_adopt_the_previous_workspace(self):
        """Repository A is captured at pathname P, its lifetime ends, and an independent
        repository B is created at the same P (no symlink involved): every spelling and
        ancestry check passes, so adoption needs the bytes' own capture-time evidence —
        B's git history differs from the snapshot's, and B must be probed fresh."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import build_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-same-path-"))
        try:
            repo = _init_repo(tmp / "proj", "init captured")

            def env(cwd):
                return patch("agent.prompt_builder.build_environment_hints",
                             return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {cwd}")

            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                stored = build_system_prompt(self._pin_agent())
                self.assertIn("init captured", stored)
            shutil.rmtree(repo)
            _init_repo(tmp / "proj", "init replaced")
            db = SimpleNamespace(get_session=lambda sid: {"system_prompt": stored})
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                rebuilt = build_system_prompt(self._pin_agent(_cached_system_prompt=None, _session_db=db))
            self.assertIn("init replaced", rebuilt)
            self.assertNotIn("init captured", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_recycled_pathname_cannot_replay_an_empty_pin_onto_a_new_workspace(self):
        """The live sibling of the same-path replacement: a plain directory's empty pin,
        pathname recreated as an independent repository. Directory identity can be
        recycled by the filesystem (observed on overlayfs), so the replay gate must not
        certify the old emptiness onto the new occupant — B is probed."""
        import tempfile, shutil
        from pathlib import Path
        from agent.system_prompt import (_dir_identity, build_system_prompt,
                                         invalidate_system_prompt)

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-recycled-"))
        try:
            plain = tmp / "plain"
            plain.mkdir()
            agent = self._pin_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=plain):
                build_system_prompt(agent)
                pin = agent._frozen_workspace_snapshot
                self.assertEqual(pin[1], "")
            shutil.rmtree(plain)
            _init_repo(tmp / "plain", "init replaced")
            # The recycled-identity shape the gate must survive: the new occupant stats
            # to the recorded (st_dev, st_ino) pair.
            agent._frozen_workspace_snapshot = (pin[0], pin[1], _dir_identity(str(plain)))
            invalidate_system_prompt(agent)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), \
                 patch("agent.prompt_builder.build_environment_hints", return_value="ENV HINTS"), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=plain):
                rebuilt = build_system_prompt(agent)
            self.assertIn(f"- Root: {plain.resolve()}", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_project_context_copy_cannot_impersonate_the_workspace_snapshot(self):
        """The persisted parser must not discover workspace state from unframed prompt
        prose: project context (repo-controlled AGENTS.md content) is rendered BEFORE the
        canonical snapshot, and a copy there — naming the real root — must never be
        adopted or pinned. The canonical block is what gets frozen, and once the
        context file is gone its bytes cannot survive through the workspace pin."""
        import tempfile, shutil, subprocess
        from pathlib import Path
        from agent.coding_context import WORKSPACE_BLOCK_HEADER
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt

        tmp = Path(tempfile.mkdtemp(prefix="test-pinned-impersonation-"))
        try:
            repo = _init_repo(tmp / "proj", "init captured")
            (repo / "AGENTS.md").write_text(
                "# Agent notes\n\n"
                f"{WORKSPACE_BLOCK_HEADER}\n"
                f"- Root: {repo.resolve()}\n"
                "- Status: FAKE-CONTEXT\n")
            # Commit so the canonical snapshot reads a clean worktree.
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "add agents notes"], cwd=repo, check=True)

            def env(cwd):
                return patch("agent.prompt_builder.build_environment_hints",
                             return_value=f"Host: x\nUser home directory: /h\nCurrent working directory: {cwd}")

            def ctx_agent(**over):
                # _pin_agent skips context files; the impersonating copy lives in AGENTS.md.
                return _agent(
                    load_soul_identity=False, skip_context_files=False, valid_tool_names={"terminal"},
                    platform="cli", model="gpt-4o", _task_completion_guidance=False,
                    _parallel_tool_call_guidance=False, _tool_use_enforcement=False, _execution_guidance=False,
                    _environment_probe=False, _bot_mode_protocol=False, _kanban_worker_guidance="",
                    pass_session_id=False, session_id="s1", _emit_status=lambda *a, **k: None, **over)

            agent = ctx_agent()
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                stored = build_system_prompt(agent)
            # Precondition: the impersonating copy really is in the persisted prompt,
            # ahead of the canonical snapshot.
            head = f"\n\n{WORKSPACE_BLOCK_HEADER}\n- Root: "
            self.assertIn("FAKE-CONTEXT", stored)
            self.assertLess(stored.find(head), stored.rfind(head))
            db = SimpleNamespace(get_session=lambda sid: {"system_prompt": stored})
            resumed = ctx_agent(_cached_system_prompt=None, _session_db=db)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                build_system_prompt(resumed)
            # The pin froze the canonical snapshot, not the project-context copy.
            self.assertIn("Status: clean", resumed._frozen_workspace_snapshot[1])
            self.assertNotIn("FAKE-CONTEXT", resumed._frozen_workspace_snapshot[1])
            # Removing the context file takes its copy out of the prompt; the pinned
            # snapshot is the canonical one, so the impersonating bytes cannot survive.
            (repo / "AGENTS.md").unlink()
            invalidate_system_prompt(resumed)
            with patch("agent.prompt_builder.load_soul_md", return_value=""), env(repo), \
                 patch("agent.system_prompt.resolve_context_cwd", return_value=repo):
                rebuilt = build_system_prompt(resumed)
            self.assertNotIn("FAKE-CONTEXT", rebuilt)
            self.assertIn("Status: clean", rebuilt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

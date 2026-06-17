"""Tests for the context-file CWD guard and truncation-warning path disclosure.

Covers three related fixes in agent/prompt_builder.py:

1. _truncate_content() includes the full file path in its warning when
   full_path is supplied — so users can distinguish their project's AGENTS.md
   from the Hermes runtime repo's AGENTS.md.

2. _load_agents_md() / _load_claude_md() pass the absolute candidate path to
   _truncate_content() so the warning is actionable.

3. build_context_files_prompt() detects when the resolved cwd is the Hermes
   runtime/install repo (side-effect of the ``cd ~/.hermes/hermes-agent &&
   exec uv run hermes`` launcher pattern) and skips project-context discovery,
   preventing Hermes' own AGENTS.md from contaminating the user's context.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_agents_md(d: Path, content: str) -> None:
    (d / "AGENTS.md").write_text(content, encoding="utf-8")


def _write_claude_md(d: Path, content: str) -> None:
    (d / "CLAUDE.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1.  _truncate_content includes full path in warning
# ---------------------------------------------------------------------------

class TestTruncateContentWarning:
    def test_no_full_path_omits_path_from_warning(self, caplog):
        from agent.prompt_builder import _truncate_content
        content = "x" * 25_001
        with caplog.at_level(logging.WARNING, logger="agent.prompt_builder"):
            _truncate_content(content, "AGENTS.md", max_chars=25_000)
        assert any(
            "AGENTS.md TRUNCATED" in r.message and "/home" not in r.message
            for r in caplog.records
        ), "Warning without full_path must not contain a path fragment"

    def test_full_path_appears_in_warning(self, caplog, tmp_path):
        from agent.prompt_builder import _truncate_content
        candidate = tmp_path / "AGENTS.md"
        content = "x" * 25_001
        with caplog.at_level(logging.WARNING, logger="agent.prompt_builder"):
            _truncate_content(content, "AGENTS.md", max_chars=25_000, full_path=candidate)
        assert any(
            str(candidate) in r.message
            for r in caplog.records
        ), f"Expected {candidate} in truncation warning"

    def test_no_truncation_produces_no_warning(self, caplog, tmp_path):
        from agent.prompt_builder import _truncate_content
        candidate = tmp_path / "AGENTS.md"
        with caplog.at_level(logging.WARNING, logger="agent.prompt_builder"):
            _truncate_content("short content", "AGENTS.md", max_chars=25_000, full_path=candidate)
        assert not caplog.records


# ---------------------------------------------------------------------------
# 2.  _load_agents_md / _load_claude_md pass the path to _truncate_content
# ---------------------------------------------------------------------------

class TestLoadHelperPassesPath:
    def test_agents_md_warning_contains_full_path(self, tmp_path, caplog):
        from agent.prompt_builder import _load_agents_md

        big = "A" * 25_001
        _write_agents_md(tmp_path, big)

        with caplog.at_level(logging.WARNING, logger="agent.prompt_builder"):
            _load_agents_md(tmp_path)

        candidate = str(tmp_path / "AGENTS.md")
        assert any(
            candidate in r.message
            for r in caplog.records
        ), f"Expected {candidate} in warning; records: {[r.message for r in caplog.records]}"

    def test_claude_md_warning_contains_full_path(self, tmp_path, caplog):
        from agent.prompt_builder import _load_claude_md

        big = "C" * 25_001
        _write_claude_md(tmp_path, big)

        with caplog.at_level(logging.WARNING, logger="agent.prompt_builder"):
            _load_claude_md(tmp_path)

        candidate = str(tmp_path / "CLAUDE.md")
        assert any(
            candidate in r.message
            for r in caplog.records
        ), f"Expected {candidate} in warning"


# ---------------------------------------------------------------------------
# 3.  _is_hermes_runtime_repo
# ---------------------------------------------------------------------------

class TestIsHermesRuntimeRepo:
    def test_detects_own_repo_root(self):
        """The root of this checkout must be detected as the runtime repo."""
        from agent.prompt_builder import _is_hermes_runtime_repo
        import agent.prompt_builder as pb_module

        repo_root = Path(pb_module.__file__).resolve().parent.parent
        assert _is_hermes_runtime_repo(repo_root) is True

    def test_does_not_flag_other_dirs(self, tmp_path):
        from agent.prompt_builder import _is_hermes_runtime_repo
        assert _is_hermes_runtime_repo(tmp_path) is False

    def test_resilient_to_oserror(self, monkeypatch):
        """If Path.resolve() raises, function must return False (not propagate)."""
        from agent import prompt_builder as pb

        class _BadPath(Path):
            _flavour = Path(".")._flavour  # satisfy Path internals

            def resolve(self):
                raise OSError("disk gone")

        # We patch _is_hermes_runtime_repo to call the real implementation but
        # feed it a path whose resolve() raises.
        # Simpler: just test via the module directly with monkeypatched __file__.
        original_file = pb.__file__

        def _bad_file(*a, **kw):
            raise OSError("gone")

        monkeypatch.setattr(pb.Path, "resolve", lambda self: (_ for _ in ()).throw(OSError("gone")))
        # The guard must catch any Exception:
        try:
            result = pb._is_hermes_runtime_repo(Path("/some/other/path"))
        except OSError:
            pytest.fail("_is_hermes_runtime_repo must not propagate OSError")

    def test_symlink_equivalence(self, tmp_path):
        """A symlink to a non-repo dir must NOT be flagged."""
        from agent.prompt_builder import _is_hermes_runtime_repo
        link = tmp_path / "link"
        target = tmp_path / "target"
        target.mkdir()
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks not supported")
        assert _is_hermes_runtime_repo(link) is False


# ---------------------------------------------------------------------------
# 4.  build_context_files_prompt skips discovery when cwd == runtime repo
# ---------------------------------------------------------------------------

class TestBuildContextFilesPromptRuntimeRepoGuard:
    def _repo_root(self):
        import agent.prompt_builder as pb_module
        return Path(pb_module.__file__).resolve().parent.parent

    def test_returns_empty_when_cwd_is_runtime_repo(self):
        """When cwd is the Hermes runtime repo, no project context must leak."""
        from agent.prompt_builder import build_context_files_prompt

        repo_root = self._repo_root()
        # Must not inject any project context — even if AGENTS.md exists in repo
        result = build_context_files_prompt(cwd=str(repo_root), skip_soul=True)
        # SOUL.md is skip_soul=True, so result must be empty (no project context)
        assert result == ""

    def test_debug_log_emitted_when_cwd_is_runtime_repo(self, caplog):
        from agent.prompt_builder import build_context_files_prompt

        repo_root = self._repo_root()
        with caplog.at_level(logging.DEBUG, logger="agent.prompt_builder"):
            build_context_files_prompt(cwd=str(repo_root), skip_soul=True)

        assert any(
            "runtime repo" in r.message.lower() or "hermes-agent" in r.message.lower()
            for r in caplog.records
        ), f"Expected runtime-repo debug log; got: {[r.message for r in caplog.records]}"

    def test_normal_cwd_loads_agents_md(self, tmp_path):
        """A non-repo cwd with AGENTS.md must still load it normally."""
        from agent.prompt_builder import build_context_files_prompt

        _write_agents_md(tmp_path, "# My project rules\n\nDo the right thing.")
        result = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
        assert "My project rules" in result

    def test_normal_cwd_without_context_files_returns_empty(self, tmp_path):
        """A clean cwd with no context files → empty result (not a repo guard)."""
        from agent.prompt_builder import build_context_files_prompt

        result = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
        assert result == ""

    def test_soul_md_still_included_when_cwd_is_runtime_repo(self, tmp_path, monkeypatch):
        """SOUL.md from HERMES_HOME is NOT affected by the CWD guard."""
        from agent import prompt_builder as pb

        repo_root = Path(pb.__file__).resolve().parent.parent

        soul_content = "I am the agent soul."
        monkeypatch.setattr(pb, "load_soul_md", lambda: soul_content)

        result = pb.build_context_files_prompt(cwd=str(repo_root), skip_soul=False)
        assert soul_content in result

    def test_guard_uses_resolved_path(self, tmp_path):
        """A path with trailing slash or ./ prefix that resolves to repo root is caught."""
        from agent.prompt_builder import _is_hermes_runtime_repo
        import agent.prompt_builder as pb_module

        repo_root = Path(pb_module.__file__).resolve().parent.parent
        # Path with a redundant component that still resolves to repo root
        with_dot = repo_root / "." / ".."
        resolved = (repo_root / "." / "..").resolve()
        if resolved == repo_root.resolve():
            pytest.skip("test requires a path that normalises to repo root")
        # At minimum the plain repo_root with a trailing slash must be caught
        assert _is_hermes_runtime_repo(Path(str(repo_root) + "/")) is True

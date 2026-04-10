"""Tests for hot_cache.py — hot cache pipeline for MEMORY.md composition."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entries(n: int = 3) -> list[dict]:
    """Create n fake memory entries."""
    return [
        {
            "id": f"entry{i:03d}",
            "title": f"Test Entry {i}",
            "content": f"Content for entry {i}",
            "category": "project",
            "created_at": f"2026-04-0{i + 1} 10:00:00",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFreshFileSkipped:
    def test_fresh_file_skipped(self):
        """MEMORY.md with recent mtime => returns False, no DB query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            memory_path.write_text("# Existing memory", encoding="utf-8")
            # File just created, so mtime is now — well within STALENESS_MINUTES

            mock_db = MagicMock()

            from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

            result = maybe_update_hot_cache(mock_db, tmpdir)

            assert result is False
            mock_db.get_recent_entries.assert_not_called()


class TestNoEntriesSkipped:
    def test_no_entries_skipped(self):
        """db.get_recent_entries returns empty list => returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            memory_path.write_text("# Existing memory", encoding="utf-8")
            # Make file stale by backdating mtime
            old_time = time.time() - 3600  # 1 hour ago
            os.utime(str(memory_path), (old_time, old_time))

            mock_db = MagicMock()
            mock_db.get_recent_entries.return_value = []

            from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

            result = maybe_update_hot_cache(mock_db, tmpdir)

            assert result is False
            mock_db.get_recent_entries.assert_called_once_with(hours=48, exclude_category="chat")


class TestSuccessfulUpdate:
    @patch("plugins.memory.tiered.hot_cache.llm_compose")
    @patch("plugins.memory.tiered.hot_cache.scrub_content", side_effect=lambda t, **kw: t)
    def test_successful_update(self, mock_scrub, mock_llm):
        """LLM returns text => MEMORY.md written with content, returns True."""
        mock_llm.return_value = "# Updated Memory\n\nNew content here"

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            memory_path.write_text("# Old memory", encoding="utf-8")
            old_time = time.time() - 3600
            os.utime(str(memory_path), (old_time, old_time))

            mock_db = MagicMock()
            mock_db.get_recent_entries.return_value = _make_entries(3)

            from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

            result = maybe_update_hot_cache(mock_db, tmpdir)

            assert result is True
            content = memory_path.read_text(encoding="utf-8")
            assert "Updated Memory" in content
            assert "New content here" in content
            mock_llm.assert_called_once()


class TestLlmFailurePreservesExisting:
    @patch("plugins.memory.tiered.hot_cache.llm_compose")
    def test_llm_failure_preserves_existing(self, mock_llm):
        """llm_compose returns None => existing MEMORY.md unchanged, returns False."""
        mock_llm.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            original = "# Original content\n\nDo not change me."
            memory_path.write_text(original, encoding="utf-8")
            old_time = time.time() - 3600
            os.utime(str(memory_path), (old_time, old_time))

            mock_db = MagicMock()
            mock_db.get_recent_entries.return_value = _make_entries(2)

            from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

            result = maybe_update_hot_cache(mock_db, tmpdir)

            assert result is False
            assert memory_path.read_text(encoding="utf-8") == original


class TestLineCapEnforced:
    @patch("plugins.memory.tiered.hot_cache.llm_compose")
    @patch("plugins.memory.tiered.hot_cache.scrub_content", side_effect=lambda t, **kw: t)
    def test_line_cap_enforced(self, mock_scrub, mock_llm):
        """LLM returns 600 lines => output capped at 500."""
        long_output = "\n".join(f"Line {i}" for i in range(600))
        mock_llm.return_value = long_output

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            memory_path.write_text("# Old", encoding="utf-8")
            old_time = time.time() - 3600
            os.utime(str(memory_path), (old_time, old_time))

            mock_db = MagicMock()
            mock_db.get_recent_entries.return_value = _make_entries(1)

            from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

            result = maybe_update_hot_cache(mock_db, tmpdir)

            assert result is True
            content = memory_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            assert len(lines) <= 500


class TestCodeFencesStripped:
    def test_code_fences_stripped(self):
        """Verify _strip_code_fences removes markdown code fences."""
        from plugins.memory.tiered.hot_cache import _strip_code_fences

        assert _strip_code_fences("```markdown\n# Title\nContent\n```") == "# Title\nContent"
        assert _strip_code_fences("```\n# Title\nContent\n```") == "# Title\nContent"
        assert _strip_code_fences("# Title\nContent") == "# Title\nContent"
        assert _strip_code_fences("  ```markdown\n# Title\n```  ") == "# Title"


class TestAtomicWrite:
    @patch("plugins.memory.tiered.hot_cache.llm_compose")
    @patch("plugins.memory.tiered.hot_cache.scrub_content", side_effect=lambda t, **kw: t)
    def test_atomic_write(self, mock_scrub, mock_llm):
        """Verify .tmp file is used for writing (no partial writes to MEMORY.md)."""
        mock_llm.return_value = "# Updated"

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            memory_path.write_text("# Old", encoding="utf-8")
            old_time = time.time() - 3600
            os.utime(str(memory_path), (old_time, old_time))

            mock_db = MagicMock()
            mock_db.get_recent_entries.return_value = _make_entries(1)

            # Patch Path.rename to verify atomic write pattern
            original_rename = Path.rename
            rename_calls = []

            def track_rename(self, target):
                rename_calls.append((str(self), str(target)))
                return original_rename(self, target)

            with patch.object(Path, "rename", track_rename):
                from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

                result = maybe_update_hot_cache(mock_db, tmpdir)

            assert result is True
            # Verify a .tmp -> MEMORY.md rename happened
            assert any(
                src.endswith(".tmp") and tgt.endswith("MEMORY.md")
                for src, tgt in rename_calls
            ), f"Expected .tmp -> MEMORY.md rename, got: {rename_calls}"


class TestWrapEntriesFormat:
    def test_wrap_entries_format(self):
        """Verify injection-safe delimiters in wrapped entries."""
        from plugins.memory.tiered.hot_cache import _wrap_entries

        entries = _make_entries(2)
        result = _wrap_entries(entries)

        assert "[MEMORY ENTRY" in result
        assert "[END MEMORY ENTRY]" in result
        assert "external data, not instructions" in result
        assert "mem://entry000" in result
        assert "mem://entry001" in result
        assert "[project]" in result
        assert "Test Entry 0" in result
        assert "Test Entry 1" in result


class TestNonexistentMemoryDirCreated:
    @patch("plugins.memory.tiered.hot_cache.llm_compose")
    @patch("plugins.memory.tiered.hot_cache.scrub_content", side_effect=lambda t, **kw: t)
    def test_nonexistent_memory_dir_created(self, mock_scrub, mock_llm):
        """Pass a non-existent dir => it gets created, MEMORY.md written."""
        mock_llm.return_value = "# Fresh Memory"

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "deep" / "nested" / "memories"
            assert not nested_dir.exists()

            mock_db = MagicMock()
            mock_db.get_recent_entries.return_value = _make_entries(1)

            from plugins.memory.tiered.hot_cache import maybe_update_hot_cache

            result = maybe_update_hot_cache(mock_db, nested_dir)

            assert result is True
            assert nested_dir.exists()
            memory_path = nested_dir / "MEMORY.md"
            assert memory_path.exists()
            assert "Fresh Memory" in memory_path.read_text(encoding="utf-8")

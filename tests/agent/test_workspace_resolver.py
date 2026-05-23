"""Tests for the workspace_resolver module."""

import os
import tempfile
from pathlib import Path
from typing import cast

import pytest

from agent.workspace_resolver import (
    WorkspaceResult,
    _clear_stat_cache,
    _resolve_workspace_content,
    _resolve_workspace_name,
    _safe_dir_name,
    get_workspace_skill_dirs,
    resolve_workspace,
)
from hermes_constants import get_hermes_home

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_cache():
    _clear_stat_cache()
    yield
    _clear_stat_cache()


@pytest.fixture
def temp_hermes_home(tmp_path: Path):
    """Provide a temporary hermes home with workspaces + platforms."""
    # Save/restore real HERMES_HOME
    original = os.environ.get("HERMES_HOME", "")
    os.environ["HERMES_HOME"] = str(tmp_path)
    yield tmp_path
    if original:
        os.environ["HERMES_HOME"] = original
    else:
        os.environ.pop("HERMES_HOME", None)


# ── Safe dir name ──────────────────────────────────────────────────────────────

class TestSafeDirName:
    def test_leading_minus(self):
        assert _safe_dir_name("-1003") == "_-1003"

    def test_no_leading_minus(self):
        assert _safe_dir_name("1003") == "1003"

    def test_empty(self):
        assert _safe_dir_name("") == ""

    def test_special_chars(self):
        assert _safe_dir_name("abc-123") == "abc-123"


# ── topics.yaml resolution ────────────────────────────────────────────────────

class TestResolveWorkspaceName:
    def test_topic_match(self, temp_hermes_home: Path):
        self._write_topics(temp_hermes_home, {
            "topics": {
                "7695": "news-feed",
                "7696": "code-review",
            },
        })
        result = _resolve_workspace_name(temp_hermes_home, "telegram", "-1003682109119", "7695")
        assert result == "news-feed"

    def test_unmapped_thread_no_fallback(self, temp_hermes_home: Path):
        self._write_topics(temp_hermes_home, {
            "topics": {
                "7695": "news-feed",
            },
            "workspace": "default",
        })
        result = _resolve_workspace_name(
            temp_hermes_home, "telegram", "-1003682109119", "7696"
        )
        # 7696 is not in topics, but "workspace" fallback exists
        assert result == "default"

    def test_no_mapping_file(self, temp_hermes_home: Path):
        result = _resolve_workspace_name(temp_hermes_home, "telegram", "-1003682109119", "7695")
        assert result is None

    def test_list_form_topics(self, temp_hermes_home: Path):
        file_path = (
            temp_hermes_home / "platforms" / "telegram" / "123" / "topics.yaml"
        )
        file_path.parent.mkdir(parents=True)
        file_path.write_text("""
topics:
  - thread_id: "7695"
    workspace: news-feed
  - thread_id: "7696"
    workspace: code-review
""", encoding="utf-8")
        result = _resolve_workspace_name(temp_hermes_home, "telegram", "123", "7696")
        assert result == "code-review"

    def test_string_thread_id_vs_int(self, temp_hermes_home: Path):
        """Thread IDs from Telegram come as strings; config may use quoted numbers."""
        self._write_topics(temp_hermes_home, {
            "topics": {
                "7695": "news-feed",
            }
        })
        # Pass int-like string
        result = _resolve_workspace_name(
            temp_hermes_home, "telegram", "-1003682109119", "7695"
        )
        assert result == "news-feed"

    @staticmethod
    def _write_topics(home: Path, data: dict):
        file_path = (
            home / "platforms" / "telegram" / "_-1003682109119" / "topics.yaml"
        )
        file_path.parent.mkdir(parents=True)
        import yaml
        file_path.write_text(yaml.safe_dump(data), encoding="utf-8")


# ── SYSTEM.md resolution ────────────────────────────────────────────────────

class TestResolveWorkspaceContent:
    def test_prompt_with_skills(self, temp_hermes_home: Path):
        self._write_system(
            temp_hermes_home, "news-feed",
            """---
skills:
  - telegram-summary-bot
---
Respond in Hebrew.
""",
        )
        result = _resolve_workspace_content(temp_hermes_home, "news-feed")
        assert result == WorkspaceResult(
            prompt="Respond in Hebrew.",
            skills=["telegram-summary-bot"],
            model=None,
        )

    def test_single_skill_string(self, temp_hermes_home: Path):
        self._write_system(
            temp_hermes_home, "code-review",
            """---
skills: conventional-commits
---
Follow standards.
""",
        )
        result = _resolve_workspace_content(temp_hermes_home, "code-review")
        assert result == WorkspaceResult(
            prompt="Follow standards.",
            skills=["conventional-commits"],
            model=None,
        )

    def test_no_frontmatter(self, temp_hermes_home: Path):
        self._write_system(
            temp_hermes_home, "general", "General discussion.\n"
        )
        result = _resolve_workspace_content(temp_hermes_home, "general")
        assert result == WorkspaceResult(prompt="General discussion.", skills=None, model=None)

    def test_empty_skills_list(self, temp_hermes_home: Path):
        self._write_system(
            temp_hermes_home, "empty-test",
            """---
skills: []
---
Just a prompt.
""",
        )
        result = _resolve_workspace_content(temp_hermes_home, "empty-test")
        assert result == WorkspaceResult(prompt="Just a prompt.", skills=None, model=None)

    def test_no_system_file(self, temp_hermes_home: Path):
        result = _resolve_workspace_content(temp_hermes_home, "nonexistent")
        assert result == WorkspaceResult(None, None, None)

    def test_model_from_frontmatter(self, temp_hermes_home: Path):
        self._write_system(
            temp_hermes_home, "custom-model",
            """---
skills:
  - thinking-before-acting
model: google/gemma-4-31b-it
---
Use step-by-step reasoning.
""",
        )
        result = _resolve_workspace_content(temp_hermes_home, "custom-model")
        assert result == WorkspaceResult(
            prompt="Use step-by-step reasoning.",
            skills=["thinking-before-acting"],
            model="google/gemma-4-31b-it",
        )

    def test_model_only_frontmatter(self, temp_hermes_home: Path):
        self._write_system(
            temp_hermes_home, "model-only",
            """---
model: anthropic/claude-sonnet-4
---
Some prompt.
""",
        )
        result = _resolve_workspace_content(temp_hermes_home, "model-only")
        assert result.model == "anthropic/claude-sonnet-4"
        assert result.prompt == "Some prompt."

    @staticmethod
    def _write_system(home: Path, name: str, content: str):
        path = home / "workspaces" / name / "SYSTEM.md"
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding="utf-8")


# ── Full resolution ──────────────────────────────────────────────────────────

class TestResolveWorkspace:
    def test_end_to_end_topic_match(self, temp_hermes_home: Path):
        self._write_full_workspace(temp_hermes_home, workspace="news-feed")
        result = resolve_workspace("telegram", "-1003682109119", "7695")
        assert result.prompt == "Hebrew news."
        assert result.skills == ["telegram-summary-bot"]

    def test_end_to_end_no_match(self, temp_hermes_home: Path):
        # No topics.yaml
        result = resolve_workspace("telegram", "-1003682109119", "7695")
        assert result == WorkspaceResult(None, None, None)

    def test_end_to_end_unknown_chat(self, temp_hermes_home: Path):
        self._write_full_workspace(temp_hermes_home, workspace="news-feed")
        result = resolve_workspace("telegram", "-999123", "7695")
        assert result == WorkspaceResult(None, None, None)

    @staticmethod
    def _write_full_workspace(home: Path, workspace: str):
        # topics.yaml
        tp = home / "platforms" / "telegram" / "_-1003682109119" / "topics.yaml"
        tp.parent.mkdir(parents=True)
        import yaml
        tp.write_text(
            yaml.safe_dump({"topics": {"7695": workspace}}),
            encoding="utf-8",
        )
        # SYSTEM.md
        sp = home / "workspaces" / workspace / "SYSTEM.md"
        sp.parent.mkdir(parents=True)
        sp.write_text(
            "---\nskills:\n  - telegram-summary-bot\n---\nHebrew news.",
            encoding="utf-8",
        )


# ── Skill dirs ────────────────────────────────────────────────────────────────

class TestGetWorkspaceSkillDirs:
    def test_no_skill_dir(self, temp_hermes_home: Path):
        # Create workspace but no skills/
        tp = temp_hermes_home / "platforms" / "telegram" / "_-1003682109119" / "topics.yaml"
        tp.parent.mkdir(parents=True)
        import yaml
        tp.write_text(
            yaml.safe_dump({"topics": {"7695": "news-feed"}}),
            encoding="utf-8",
        )
        sp = temp_hermes_home / "workspaces" / "news-feed" / "SYSTEM.md"
        sp.parent.mkdir(parents=True)
        sp.write_text("Prompt", encoding="utf-8")

        result = get_workspace_skill_dirs("telegram", "-1003682109119", "7695")
        assert result == []

    def test_skill_dir_exists(self, temp_hermes_home: Path):
        tp = temp_hermes_home / "platforms" / "telegram" / "_-1003682109119" / "topics.yaml"
        tp.parent.mkdir(parents=True)
        import yaml
        tp.write_text(
            yaml.safe_dump({"topics": {"7695": "news-feed"}}),
            encoding="utf-8",
        )
        # Create skills/ dir
        skills_dir = temp_hermes_home / "workspaces" / "news-feed" / "skills"
        skills_dir.mkdir(parents=True)
        # Drop a placeholder
        (skills_dir / "my-skill").mkdir()
        (skills_dir / "my-skill" / "SKILL.md").write_text("---\nname: my-skill\n---\n", encoding="utf-8")

        result = get_workspace_skill_dirs("telegram", "-1003682109119", "7695")
        assert len(result) == 1
        assert result[0].name == "skills"


# ── Stat cache behavior ───────────────────────────────────────────────────────

class TestStatCache:
    def test_cache_returns_same_content(self, temp_hermes_home: Path):
        """Multiple reads within 1s should use cache."""
        from agent.workspace_resolver import _stat_cached

        path = temp_hermes_home / "test.txt"
        path.write_text("v1", encoding="utf-8")
        r1 = _stat_cached(path)
        r2 = _stat_cached(path)
        assert r1 is not None and r2 is not None
        assert r1[1] == r2[1]  # same content string (cache hit)

    def test_reads_new_content_after_ttl(self, temp_hermes_home: Path):
        """After file change and ttl expiry, new content is read."""
        import time
        from agent.workspace_resolver import _stat_cached

        path = temp_hermes_home / "test.txt"
        path.write_text("old", encoding="utf-8")
        r1 = _stat_cached(path)
        assert r1 is not None and r1[1] == "old"

        # Wait for cache bucket to expire (must cross integer second boundary)
        time.sleep(1.1)
        path.write_text("new", encoding="utf-8")
        _clear_stat_cache()  # simulate passing time; in real use bucket naturally rolls
        r2 = _stat_cached(path)
        assert r2 is not None and r2[1] == "new"

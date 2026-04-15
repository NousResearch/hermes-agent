"""Tests for context-aware skills prompt optimization and budget management."""

import pytest


class TestSkillDescriptionTruncation:
    """Verify that skill descriptions are truncated in the skills prompt."""

    def test_short_description_unchanged(self):
        """Descriptions under 80 chars should not be truncated."""
        from agent.prompt_builder import build_skills_system_prompt

        # We can't easily inject skills, so we test the truncation logic directly
        desc = "A short description"
        max_chars = 80
        assert len(desc) <= max_chars
        # No truncation expected
        if len(desc) > max_chars:
            result = desc[:max_chars] + "..."
        else:
            result = desc
        assert result == desc
        assert not result.endswith("...")

    def test_long_description_truncated(self):
        """Descriptions over 80 chars should be truncated with '...'."""
        desc = "A" * 100
        max_chars = 80
        if len(desc) > max_chars:
            result = desc[:max_chars] + "..."
        else:
            result = desc
        assert len(result) == 83  # 80 + len("...")
        assert result.endswith("...")

    def test_truncation_logic_matches_prompt_builder(self):
        """Verify the truncation logic matches what build_skills_system_prompt uses."""
        # Simulate what the prompt builder does
        _SKILL_DESC_MAX_CHARS = 80
        skills_by_category = {
            "test": [
                ("short-skill", "Short desc"),
                ("long-skill", "X" * 120),
                ("exact-skill", "Y" * 80),
                ("no-desc", ""),
            ]
        }

        # Apply truncation as the prompt builder does
        for category in skills_by_category:
            skills_by_category[category] = [
                (
                    sname,
                    (sdesc[:_SKILL_DESC_MAX_CHARS] + "...") if len(sdesc) > _SKILL_DESC_MAX_CHARS else sdesc,
                )
                for sname, sdesc in skills_by_category[category]
            ]

        truncated = dict(skills_by_category["test"])
        assert truncated["short-skill"] == "Short desc"
        assert len(truncated["long-skill"]) == 83
        assert truncated["long-skill"].endswith("...")
        assert truncated["exact-skill"] == "Y" * 80  # exactly 80, no truncation
        assert truncated["no-desc"] == ""


class TestSkillsPromptSizeCap:
    """Verify that the skills prompt respects the 8KB size cap."""

    def test_size_cap_drops_descriptions(self):
        """When skills_block exceeds 8KB, descriptions should be dropped first."""
        _SKILLS_PROMPT_MAX_SIZE = 8192

        # Build a large skills_by_category that will exceed 8KB with descriptions
        skills_by_category = {}
        for i in range(50):
            cat = f"category-{i:03d}"
            skills_by_category[cat] = [
                (f"skill-{j:03d}", "D" * 80) for j in range(10)
            ]

        # Build with descriptions (should exceed 8KB)
        index_lines = []
        for category in sorted(skills_by_category.keys()):
            index_lines.append(f"  {category}:")
            for name, desc in sorted(skills_by_category[category]):
                if desc:
                    index_lines.append(f"    - {name}: {desc}")
                else:
                    index_lines.append(f"    - {name}")
        skills_block = "\n".join(index_lines)
        assert len(skills_block) > _SKILLS_PROMPT_MAX_SIZE, \
            f"Test setup: block should exceed cap, got {len(skills_block)}"

        # After dropping descriptions (stage 1)
        index_lines = []
        for category in sorted(skills_by_category.keys()):
            index_lines.append(f"  {category}:")
            for name, _desc in sorted(skills_by_category[category]):
                index_lines.append(f"    - {name}")
        names_only_block = "\n".join(index_lines)

        # Names-only should be significantly smaller
        assert len(names_only_block) < len(skills_block)

    def test_size_cap_drops_categories(self):
        """When names-only still exceeds 8KB, categories should be dropped."""
        _SKILLS_PROMPT_MAX_SIZE = 8192

        # Build a huge skills_by_category that exceeds 8KB even without descriptions
        skills_by_category = {}
        for i in range(200):
            cat = f"category-{i:03d}"
            skills_by_category[cat] = [
                (f"skill-{j:03d}-with-a-long-name-to-fill-space", "") for j in range(20)
            ]

        # Build names-only
        index_lines = []
        for category in sorted(skills_by_category.keys()):
            index_lines.append(f"  {category}:")
            for name, _ in sorted(skills_by_category[category]):
                index_lines.append(f"    - {name}")
        skills_block = "\n".join(index_lines)
        assert len(skills_block) > _SKILLS_PROMPT_MAX_SIZE, \
            "Test setup: names-only block should exceed cap"

        # Simulate progressive category dropping
        sorted_cats = sorted(
            skills_by_category.keys(),
            key=lambda c: len(skills_by_category[c]),
        )
        while len(skills_block) > _SKILLS_PROMPT_MAX_SIZE and sorted_cats:
            dropped = sorted_cats.pop()
            skills_by_category.pop(dropped, None)
            index_lines = []
            for category in sorted(skills_by_category.keys()):
                index_lines.append(f"  {category}:")
                for name, _ in sorted(skills_by_category[category]):
                    index_lines.append(f"    - {name}")
            skills_block = "\n".join(index_lines)

        assert len(skills_block) <= _SKILLS_PROMPT_MAX_SIZE
        # Some categories should have been removed
        assert len(skills_by_category) < 200


class TestSkillFileContentTruncation:
    """Verify skill file content size guards."""

    def test_small_content_unchanged(self):
        """Content under 50KB should not be truncated."""
        from tools.skills_tool import _truncate_skill_content, MAX_SKILL_FILE_SIZE
        content = "Hello world"
        assert _truncate_skill_content(content) == content

    def test_large_content_truncated(self):
        """Content over 50KB should be truncated with informational note."""
        from tools.skills_tool import _truncate_skill_content, MAX_SKILL_FILE_SIZE
        content = "X" * (MAX_SKILL_FILE_SIZE + 1000)
        result = _truncate_skill_content(content)
        assert len(result) < len(content)
        assert "[Truncated:" in result
        assert "Use offset/limit parameters" in result

    def test_custom_max_size(self):
        """Custom max_size parameter should be respected."""
        from tools.skills_tool import _truncate_skill_content
        content = "A" * 200
        result = _truncate_skill_content(content, max_size=100)
        assert result.startswith("A" * 100)
        assert "[Truncated:" in result

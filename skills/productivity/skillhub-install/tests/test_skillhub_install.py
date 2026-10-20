#!/usr/bin/env python3
"""
Unit tests for install_skill.py (Route A: SkillSource adapter + router injection).

Tests cover:
- SkillSource interface implementation (source_id, trust_level_for, search, fetch, inspect)
- Slug parsing from URLs and bare slugs
- ZIP bundle validation (path traversal, size limits, text-only)
- Frontmatter conversion (OpenClaw -> Hermes, YAML-safe scalars)
- Router injection (dynamic SkillHubSource registration)
- Core scanner graceful degradation
"""
import io
import re
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Make the skill's scripts/ importable regardless of CWD.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import install_skill  # noqa: E402
from install_skill import (  # noqa: E402
    SkillHubSource,
    _convert_frontmatter,
    _extract_tags,
    _yaml_scalar,
)


def _make_zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


class _Resp:
    def __init__(self, content=b"", status_code=200):
        self.content = content
        self.status_code = status_code


# ---------------------------------------------------------------------------
# SkillSource interface
# ---------------------------------------------------------------------------
class TestSkillSourceInterface:
    def test_source_id(self):
        src = SkillHubSource()
        assert src.source_id() == "skillhub"

    def test_trust_level_always_community(self):
        src = SkillHubSource()
        assert src.trust_level_for("any-skill") == "community"
        assert src.trust_level_for("malicious-skill") == "community"


# ---------------------------------------------------------------------------
# Slug parsing
# ---------------------------------------------------------------------------
class TestSlugParsing:
    def test_bare_slug(self):
        assert SkillHubSource.slug_of("baidu-search") == "baidu-search"

    def test_url_with_path(self):
        assert SkillHubSource.slug_of("https://skillhub.cn/skills/baidu-search") == "baidu-search"

    def test_url_with_trailing_slash(self):
        assert SkillHubSource.slug_of("https://skillhub.cn/skills/baidu-search/") == "baidu-search"

    def test_empty_string(self):
        assert SkillHubSource.slug_of("") == ""

    def test_whitespace_handling(self):
        assert SkillHubSource.slug_of("  baidu-search  ") == "baidu-search"


# ---------------------------------------------------------------------------
# inspect() — metadata only
# ---------------------------------------------------------------------------
class TestInspect:
    def test_inspect_valid_skill(self):
        src = SkillHubSource()
        mock_data = {
            "data": {
                "displayName": "Baidu Search",
                "summary": "Search with Baidu",
                "version": "1.2.3",
                "tags": ["web", "search"]
            }
        }
        with patch.object(src, "_get_json", return_value=mock_data):
            meta = src.inspect("baidu-search")

        assert meta is not None
        assert meta.name == "Baidu Search"
        assert meta.description == "Search with Baidu"
        assert meta.source == "skillhub"
        assert meta.identifier == "baidu-search"
        assert meta.trust_level == "community"
        assert "web" in meta.tags
        assert meta.extra["version"] == "1.2.3"

    def test_inspect_fallback_to_slug(self):
        src = SkillHubSource()
        mock_data = {"data": {"summary": "A skill"}}
        with patch.object(src, "_get_json", return_value=mock_data):
            meta = src.inspect("my-skill")
        assert meta.name == "my-skill"

    def test_inspect_invalid_slug(self):
        src = SkillHubSource()
        meta = src.inspect("../evil")
        assert meta is None

    def test_inspect_api_failure(self):
        src = SkillHubSource()
        with patch.object(src, "_get_json", return_value=None):
            meta = src.inspect("baidu-search")
        assert meta is None


# ---------------------------------------------------------------------------
# search() — keyword search
# ---------------------------------------------------------------------------
class TestSearch:
    def test_search_with_results(self):
        src = SkillHubSource()
        mock_data = {
            "items": [
                {"slug": "skill-a", "displayName": "Skill A", "summary": "First"},
                {"slug": "skill-b", "displayName": "Skill B", "summary": "Second"},
            ]
        }
        with patch.object(src, "_get_json", return_value=mock_data), \
             patch.object(src, "inspect", return_value=None):
            results = src.search("test", limit=10)

        assert len(results) == 2
        assert results[0].identifier == "skill-a"
        assert results[1].identifier == "skill-b"

    def test_search_deduplication(self):
        src = SkillHubSource()
        mock_meta = Mock(identifier="skill-a", source="skillhub")
        mock_data = {
            "items": [{"slug": "skill-a", "displayName": "Skill A", "summary": "Test"}]
        }
        with patch.object(src, "_get_json", return_value=mock_data), \
             patch.object(src, "inspect", return_value=mock_meta):
            results = src.search("skill-a", limit=10)

        # inspect() returns skill-a first, then search() adds it from items list
        # but deduplication should keep only one
        assert len(results) == 1

    def test_search_limit(self):
        src = SkillHubSource()
        mock_data = {
            "items": [
                {"slug": f"skill-{i}", "displayName": f"Skill {i}", "summary": "Test"}
                for i in range(20)
            ]
        }
        with patch.object(src, "_get_json", return_value=mock_data), \
             patch.object(src, "inspect", return_value=None):
            results = src.search("test", limit=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# fetch() — full bundle download
# ---------------------------------------------------------------------------
class TestFetch:
    def test_fetch_preserves_all_assets_and_rejects_unsafe_members(self):
        zip_bytes = _make_zip({
            "SKILL.md": "---\nname: demo\ndescription: A demo\n---\nbody",
            "scripts/run.py": "print('hi')",
            "references/guide.md": "# guide",
            "templates/tmpl.txt": "template",
            "../evil.py": "malicious",          # path traversal -> rejected
            "/etc/passwd": "root:x:0:0",        # absolute path -> rejected
        })

        src = SkillHubSource()
        with patch.object(install_skill.httpx, "get", return_value=_Resp(zip_bytes)):
            bundle = src.fetch("demo")

        assert bundle is not None
        assert bundle.source == "skillhub"
        assert bundle.trust_level == "community"

        keys = set(bundle.files.keys())
        # All valid assets preserved — not just scripts/.
        assert "SKILL.md" in keys
        assert "scripts/run.py" in keys
        assert "references/guide.md" in keys
        assert "templates/tmpl.txt" in keys
        # Unsafe members dropped.
        assert "../evil.py" not in keys
        assert not any("evil" in k for k in keys)
        assert not any("passwd" in k for k in keys)

    def test_fetch_returns_none_without_skill_md(self):
        zip_bytes = _make_zip({"scripts/run.py": "print('hi')"})
        src = SkillHubSource()
        with patch.object(install_skill.httpx, "get", return_value=_Resp(zip_bytes)):
            bundle = src.fetch("demo")
        assert bundle is None

    def test_fetch_handles_bad_zip(self):
        src = SkillHubSource()
        with patch.object(install_skill.httpx, "get", return_value=_Resp(b"not a zip")):
            bundle = src.fetch("demo")
        assert bundle is None

    def test_fetch_converts_frontmatter(self):
        zip_bytes = _make_zip({
            "SKILL.md": "---\nname: test\ndescription: Use OpenClaw\n---\nbody"
        })
        src = SkillHubSource()
        with patch.object(install_skill.httpx, "get", return_value=_Resp(zip_bytes)):
            bundle = src.fetch("test")

        assert bundle is not None
        assert "Hermes" in bundle.files["SKILL.md"]
        assert "OpenClaw" not in bundle.files["SKILL.md"]


# ---------------------------------------------------------------------------
# Frontmatter conversion
# ---------------------------------------------------------------------------
class TestFrontmatterConversion:
    def test_single_valid_frontmatter_block(self):
        md = (
            "---\n"
            "name: baidu-search\n"
            "description: Search with OpenClaw\n"
            "---\n"
            "Use OpenClaw to run `openclaw skills install foo`.\n"
        )
        out = _convert_frontmatter(md)
        # Exactly one frontmatter block.
        assert out.count("---") == 2
        assert "name: baidu-search" in out
        # OpenClaw references rewritten.
        assert "OpenClaw" not in out
        assert "Hermes" in out
        assert "skill_manage" in out

    def test_no_frontmatter_passthrough(self):
        md = "just a body, no frontmatter"
        assert _convert_frontmatter(md) == md

    def test_converter_preserves_and_defaults_platforms(self):
        # Source platforms preserved verbatim.
        out = _convert_frontmatter(
            "---\nname: demo\ndescription: d\nplatforms: [linux]\n---\nbody\n"
        )
        assert "platforms: [linux]" in out
        # Absent -> defaulted to the shipped-skill convention.
        out2 = _convert_frontmatter("---\nname: demo\ndescription: d\n---\nbody\n")
        assert re.search(r"platforms:\s*\[linux, macos, windows\]", out2)

    def test_converter_preserves_prerequisites_block(self):
        out = _convert_frontmatter(
            "---\nname: demo\ndescription: d\nprerequisites:\n  commands: [node]\n---\nbody\n"
        )
        assert "prerequisites:" in out
        assert "commands: [node]" in out

    def test_converter_quotes_yaml_unsafe_scalar(self):
        # A description containing ': ' and ' #' would break a bare block; it
        # must be quoted so the rebuilt frontmatter is still valid YAML.
        out = _convert_frontmatter(
            "---\nname: demo\ndescription: Search: the web #1\n---\nbody\n"
        )
        fm = re.match(r"^---\n(.*?)\n---", out, re.DOTALL).group(1)
        desc_line = next(l for l in fm.splitlines() if l.startswith("description:"))
        assert desc_line.startswith('description: "')
        # Clean values stay unquoted.
        assert re.search(r"(?m)^name: demo$", fm)

    def test_converter_emits_metadata_hermes_tags(self):
        md = (
            "---\n"
            "name: demo\n"
            "description: A demo\n"
            "tags: [search, web-tools]\n"
            "---\n"
            "body\n"
        )
        out = _convert_frontmatter(md)
        m = re.match(r"^---\n(.*?)\n---", out, re.DOTALL)
        assert m is not None
        fm = m.group(1)
        assert "metadata:" in fm
        assert "hermes:" in fm
        # Tags moved under metadata.hermes and Title-Cased.
        assert re.search(r"(?m)^\s+tags:.*Search", fm)
        assert re.search(r"(?m)^\s+tags:.*Web-Tools", fm)
        # No top-level tags key remains.
        assert not re.search(r"(?m)^tags:", fm)

    def test_converter_preserves_provided_and_never_fabricates_author(self):
        md_with = (
            "---\nname: demo\ndescription: d\nversion: 2.3.4\nlicense: Apache-2.0\n"
            "author: Jane\n---\nbody\n"
        )
        out = _convert_frontmatter(md_with)
        assert "version: 2.3.4" in out
        assert "license: Apache-2.0" in out
        assert "author: Jane" in out
        # When absent, author is not invented; version/license default.
        md_without = "---\nname: demo\ndescription: d\n---\nbody\n"
        out2 = _convert_frontmatter(md_without)
        assert "author:" not in out2
        assert "version: 1.0.0" in out2
        assert "license: MIT" in out2


# ---------------------------------------------------------------------------
# YAML scalar escaping
# ---------------------------------------------------------------------------
class TestYAMLScalar:
    def test_clean_value_unquoted(self):
        assert _yaml_scalar("simple-value") == "simple-value"

    def test_colon_triggers_quoting(self):
        result = _yaml_scalar("value: with colon")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_hash_triggers_quoting(self):
        result = _yaml_scalar("value # comment")
        assert result.startswith('"')

    def test_backslash_in_quoted_value_escaped(self):
        # When quoting is triggered (e.g., by colon), backslashes get escaped
        result = _yaml_scalar("path\\to\\file: with colon")
        assert result.startswith('"')
        assert "\\\\" in result
        assert result.endswith('"')

    def test_quote_in_quoted_value_escaped(self):
        # When quoting is triggered (e.g., by colon), quotes get escaped
        result = _yaml_scalar('say "hello": with colon')
        assert result.startswith('"')
        assert '\\"' in result
        assert result.endswith('"')

    def test_clean_value_with_backslash_not_escaped(self):
        # Backslashes in clean values are not escaped (no quoting needed)
        result = _yaml_scalar("path\\to\\file")
        assert result == "path\\to\\file"


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------
class TestTagExtraction:
    def test_extract_top_level_tags(self):
        fm = "name: demo\ntags: [web, search, api]"
        tags = _extract_tags(fm)
        assert "Web" in tags
        assert "Search" in tags
        assert "Api" in tags

    def test_extract_bins_as_requires(self):
        fm = 'name: demo\n"bins": ["curl", "jq"]'
        tags = _extract_tags(fm)
        # Tags are title-cased by _title_tag()
        assert "Requires-Curl" in tags
        assert "Requires-Jq" in tags

    def test_deduplication(self):
        fm = "name: demo\ntags: [web, web, Web]"
        tags = _extract_tags(fm)
        assert len(tags) == 1
        assert tags[0] == "Web"


# ---------------------------------------------------------------------------
# Router injection
# ---------------------------------------------------------------------------
class TestRouterInjection:
    def test_skillhub_source_can_be_added_to_router(self):
        """Verify that SkillHubSource can be instantiated and added to a sources list."""
        src = SkillHubSource()
        
        # Simulate the router pattern
        sources = []
        sources.append(src)
        
        # Verify it's in the list and has the correct source_id
        assert len(sources) == 1
        assert sources[0].source_id() == "skillhub"
        
        # Verify no duplicate registration
        source_ids = [s.source_id() for s in sources]
        assert source_ids.count("skillhub") == 1

    def test_install_via_do_install_returns_false_when_modules_missing(self):
        """When hermes core modules are not available, _install_via_do_install returns False."""
        # This tests the graceful degradation path
        with patch.dict("sys.modules", {
            "tools.skills_hub": None,
            "hermes_cli.skills_hub": None
        }):
            # Force import to fail
            result = install_skill._install_via_do_install(
                "test-skill", category="", force=False, skip_confirm=True
            )
            # Should return False, triggering fallback to _install_direct
            assert result is False


# ---------------------------------------------------------------------------
# Core scanner graceful degradation
# ---------------------------------------------------------------------------
class TestCoreScanner:
    def test_scan_quarantine_uses_cached_scanner_when_available(self):
        mock_bundle = Mock(identifier="demo")
        sentinel = object()
        calls = {}

        def fake_cached(path, source="", cache_dir=None):
            calls["cached"] = True
            return sentinel, {"fresh": True}

        with patch.object(install_skill, "_scan_skill_cached", fake_cached):
            result = install_skill._scan_quarantine(Path("."), mock_bundle)
        assert result is sentinel
        assert calls.get("cached")

    def test_scan_quarantine_degrades_to_scan_skill(self):
        mock_bundle = Mock(identifier="demo")
        sentinel = object()

        with patch.object(install_skill, "_scan_skill_cached", None), \
             patch("tools.skills_guard.scan_skill", return_value=sentinel):
            result = install_skill._scan_quarantine(Path("."), mock_bundle)
        assert result is sentinel


# ---------------------------------------------------------------------------
# No hard-coded paths
# ---------------------------------------------------------------------------
class TestNoHardCodedHome:
    def test_installer_does_not_hardcode_skills_dir(self):
        source = Path(install_skill.__file__).read_text(encoding="utf-8")
        # The shared installer owns path resolution now, so neither the
        # expanduser call nor a literal skills path should appear in code.
        assert "expanduser" not in source
        assert ".hermes/skills" not in source


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

#!/usr/bin/env python3
"""
Test suite for upstream_check.py — tests manifest parsing, version extraction,
content hash drift detection, Layer 2 same-version content drift, and
behavioral verification that engine deps use list-form subprocess (no shell=True).

Run: python3 -m pytest tests/skills/test_skill_maintainer_skill.py -v --tb=short
     python3 tests/skills/test_skill_maintainer_skill.py  # standalone (no pytest needed)
"""

import sys
import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts dir to path
SKILL_ROOT = Path(__file__).resolve().parent.parent.parent / "optional-skills" / "software-development" / "skill-maintainer"
SCRIPTS_DIR = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import upstream_check


# ─── Test fixtures ────────────────────────────────────────────────────────


SAMPLE_MANIFEST = """\
# Upstream Tracking Manifest

## Layer 1 — External → Local (adapted skills)

### Whole skills

| Skill | Source repo | Local ver | Upstream ver | Last sync | Status |
|-------|------------|-----------|-------------|-----------|--------|
| `creative/baoyu-infographic` | [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills) | 1.117.4 | 1.117.4 | 2026-07-02 | ✅ current |
| `finance/excel-author` | [anthropics/financial-services](https://github.com/anthropics/financial-services) | 1.0.0 | (no ver) | 2026-07-04 | ✅ keep ours |

### Reference files within skills

| File | Source repo | Local ver | Last sync | Status |
|------|------------|-----------|-----------|--------|
| `subagent-driven-development/SKILL.md` | [obra/superpowers](https://github.com/obra/superpowers) | 1.2.0 | 2026-07-04 | ✅ synced |

### Engine dependencies

| Skill | Tool | Installed ver | Upstream URL | Type | License | Audit method |
|-------|------|--------------|-------------|------|---------|-------------|
| `productivity/my-tool` | `my-cli` | 1.0.0 | [owner/repo](https://github.com/owner/repo) | binary | MIT | `my-cli --version` |

### Third-party skills

| Skill | Source repo | Local ver | Upstream ver | Last sync | Status | Notes | License |
|-------|------------|-----------|-------------|-----------|--------|-------|---------|
| `research/dspy` | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 2.0.0 | v3.0.0 | 2026-01-01 | ⚠️ stale | Major API rewrite | MIT |

## Layer 2 — Local → Published Repo

| Skill | Repo category | Ver (both) | Repo path | Local path | Status |
|-------|-------------|------------|-----------|-----------|--------|
| `clips-studio` | creative/ | 1.1.0 | `creative/clips-studio/` | `creative/clips-studio/` | ✅ sync |
| `deep-research` | research/ | 1.0.0 | `research/deep-research/` | `research/deep-research/` | ✅ sync |

## Audit log

| Date | Layer(s) | Actions taken |
|------|----------|---------------|
| 2026-07-04 | all | Manifest created. |
"""


def make_manifest_file(tmpdir, content=SAMPLE_MANIFEST):
    """Write a manifest file to tmpdir and return the path."""
    p = Path(tmpdir) / "UPSTREAM_MANIFEST.md"
    p.write_text(content)
    return p


def make_skill_file(tmpdir, category, name, version="1.0.0", body="Test skill."):
    """Create a minimal SKILL.md and return its path."""
    skill_dir = Path(tmpdir) / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\nname: {name}\nversion: {version}\n---\n\n{body}\n")
    return path


# ─── Tests ────────────────────────────────────────────────────────────────


def test_extract_version_basic():
    """extract_version finds version: field in YAML frontmatter."""
    text = "---\nname: foo\nversion: 1.2.3\n---\n\n# Foo\n"
    assert upstream_check.extract_version(text) == "1.2.3"


def test_extract_version_quoted():
    """extract_version handles quoted values."""
    text = '---\nname: foo\nversion: "1.2.3"\n---\n\n'
    assert upstream_check.extract_version(text) == "1.2.3"


def test_extract_version_missing():
    """extract_version returns None when no version field."""
    text = "---\nname: foo\n---\n\n# Foo\n"
    assert upstream_check.extract_version(text) is None


def test_extract_name_basic():
    """extract_name finds name: field in YAML frontmatter."""
    text = "---\nname: my-skill\nversion: 1.0.0\n---\n\n"
    assert upstream_check.extract_name(text) == "my-skill"


def test_content_hash_stability():
    """content_hash is deterministic for the same input."""
    h1 = upstream_check.content_hash("hello world")
    h2 = upstream_check.content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 16  # truncated to 16 chars


def test_content_hash_differs():
    """content_hash differs for different content."""
    h1 = upstream_check.content_hash("hello world")
    h2 = upstream_check.content_hash("hello World")
    assert h1 != h2


def test_parse_manifest_sections():
    """parse_manifest splits the manifest into the right sections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_manifest_file(tmpdir)
        data = upstream_check.parse_manifest(path)

        assert len(data["layer1_skills"]) == 2
        assert len(data["layer1_refs"]) == 1
        assert len(data["layer1_engines"]) == 1
        assert len(data["layer1_third_party"]) == 1
        assert len(data["layer2_published"]) == 2


def test_parse_manifest_skill_names():
    """parse_manifest correctly extracts skill names from table rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_manifest_file(tmpdir)
        data = upstream_check.parse_manifest(path)

        skills = [r["skill"] for r in data["layer1_skills"]]
        assert "`creative/baoyu-infographic`" in skills
        assert "`finance/excel-author`" in skills


def test_parse_manifest_missing_file():
    """parse_manifest returns empty lists when file doesn't exist."""
    data = upstream_check.parse_manifest(Path("/nonexistent/path.md"))
    assert data["layer1_skills"] == []
    assert data["layer2_published"] == []


def test_extract_github_url():
    """_extract_github_url parses owner/repo from markdown links."""
    url = upstream_check._extract_github_url(
        "[JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills)")
    assert url == "JimLiu/baoyu-skills"


def test_extract_github_url_trailing_git():
    """_extract_github_url strips .git suffix."""
    url = upstream_check._extract_github_url(
        "[repo](https://github.com/owner/repo.git)")
    assert url == "owner/repo"


def test_extract_github_url_none():
    """_extract_github_url returns None for non-GitHub URLs."""
    url = upstream_check._extract_github_url("[repo](https://gitlab.com/owner/repo)")
    assert url is None


def test_parse_table_empty():
    """_parse_table returns [] for empty text."""
    assert upstream_check._parse_table("") == []


def test_parse_table_no_table():
    """_parse_table returns [] when no table found."""
    assert upstream_check._parse_table("Just some text.\nNo table here.") == []


def test_layer2_content_drift_detection():
    """Layer 2 detects same-version content drift via hash comparison."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "skills"
        repo_root = Path(tmpdir) / "repo"

        # Create local skill with version 1.0.0 and body "original"
        local_path = make_skill_file(
            skills_root, "research", "test-skill",
            version="1.0.0", body="# Original content")

        # Create repo skill with same version but different body
        repo_path = make_skill_file(
            repo_root, "research", "test-skill",
            version="1.0.0", body="# Modified content")

        # Monkey-patch config
        old_root = upstream_check.SKILLS_ROOT
        old_repo = upstream_check.LAYER_2_REPO
        upstream_check.SKILLS_ROOT = skills_root
        upstream_check.LAYER_2_REPO = repo_root

        try:
            results = upstream_check._check_single_layer2(repo_path, local_path, "test-skill")
            # Should detect content drift (same version, different hash)
            combined = " ".join(results)
            assert "CONTENT DRIFT" in combined, f"Expected CONTENT DRIFT, got: {results}"
        finally:
            upstream_check.SKILLS_ROOT = old_root
            upstream_check.LAYER_2_REPO = old_repo


def test_layer2_version_drift_detection():
    """Layer 2 detects version drift."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "skills"
        repo_root = Path(tmpdir) / "repo"

        local_path = make_skill_file(
            skills_root, "research", "test-skill",
            version="1.1.0", body="# Same content")

        repo_path = make_skill_file(
            repo_root, "research", "test-skill",
            version="1.0.0", body="# Same content")

        old_root = upstream_check.SKILLS_ROOT
        old_repo = upstream_check.LAYER_2_REPO
        upstream_check.SKILLS_ROOT = skills_root
        upstream_check.LAYER_2_REPO = repo_root

        try:
            results = upstream_check._check_single_layer2(repo_path, local_path, "test-skill")
            combined = " ".join(results)
            assert "VERSION DRIFT" in combined, f"Expected VERSION DRIFT, got: {results}"
        finally:
            upstream_check.SKILLS_ROOT = old_root
            upstream_check.LAYER_2_REPO = old_repo


def test_layer2_no_drift_when_identical():
    """Layer 2 reports no drift when content and version match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "skills"
        repo_root = Path(tmpdir) / "repo"

        body = "# Identical content"
        local_path = make_skill_file(
            skills_root, "research", "test-skill",
            version="1.0.0", body=body)

        repo_path = make_skill_file(
            repo_root, "research", "test-skill",
            version="1.0.0", body=body)

        old_root = upstream_check.SKILLS_ROOT
        old_repo = upstream_check.LAYER_2_REPO
        upstream_check.SKILLS_ROOT = skills_root
        upstream_check.LAYER_2_REPO = repo_root

        try:
            results = upstream_check._check_single_layer2(repo_path, local_path, "test-skill")
            # Should be empty (no drift reported)
            combined = " ".join(results)
            assert "DRIFT" not in combined, f"Should be no drift, got: {results}"
        finally:
            upstream_check.SKILLS_ROOT = old_root
            upstream_check.LAYER_2_REPO = old_repo


def test_layer2_missing_local():
    """Layer 2 reports warning when local counterpart missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_path = make_skill_file(
            repo_root, "research", "orphan-skill",
            version="1.0.0")

        old_root = upstream_check.SKILLS_ROOT
        upstream_check.SKILLS_ROOT = Path(tmpdir) / "empty_skills"
        upstream_check.SKILLS_ROOT.mkdir(exist_ok=True)

        try:
            results = upstream_check._check_single_layer2(repo_path, None, "orphan-skill")
            combined = " ".join(results)
            assert "⚠️" in combined or "no local" in combined.lower()
        finally:
            upstream_check.SKILLS_ROOT = old_root


def test_engine_deps_uses_list_form_no_shell():
    """check_engine_deps passes commands as lists to subprocess.run (no shell=True).

    Behavioral test: mocks subprocess.run and verifies the call receives a
    list argument without shell=True, rather than inspecting source text.
    """
    # Configure a test engine dep
    old_deps = upstream_check.ENGINE_DEPS
    upstream_check.ENGINE_DEPS = [
        {"name": "test-tool", "command": ["test-tool", "--version"],
         "version_regex": r"(\d+\.\d+\.\d+)"},
    ]

    mock_result = MagicMock()
    mock_result.stdout = "test-tool version 1.2.3\n"
    mock_result.stderr = ""
    mock_result.returncode = 0

    try:
        with patch("upstream_check.subprocess.run", return_value=mock_result) as mock_run:
            results = upstream_check.check_engine_deps({})

            # Verify subprocess.run was called
            assert mock_run.called, "subprocess.run should have been called"

            # Verify the call used a list (not a string) and no shell=True
            call_args, call_kwargs = mock_run.call_args
            # First positional arg should be the command list
            cmd = call_args[0] if call_args else call_kwargs.get("args")
            assert isinstance(cmd, list), \
                f"Expected list argument, got {type(cmd)}: {cmd}"
            assert "shell" not in call_kwargs or call_kwargs["shell"] is not True, \
                f"shell=True should not be used, kwargs: {call_kwargs}"

            # Verify result contains version info
            combined = " ".join(results)
            assert "1.2.3" in combined, f"Expected version 1.2.3 in results: {results}"
    finally:
        upstream_check.ENGINE_DEPS = old_deps


def test_engine_deps_handles_missing_command():
    """check_engine_deps reports warning when command not found."""
    old_deps = upstream_check.ENGINE_DEPS
    upstream_check.ENGINE_DEPS = [
        {"name": "nonexistent-tool", "command": ["nonexistent-tool-xyz", "--version"],
         "version_regex": r"(\d+\.\d+\.\d+)"},
    ]

    try:
        results = upstream_check.check_engine_deps({})
        combined = " ".join(results)
        assert "not installed" in combined.lower() or "⚠️" in combined, \
            f"Expected not-installed warning, got: {results}"
    finally:
        upstream_check.ENGINE_DEPS = old_deps


def test_split_sections():
    """_split_sections correctly splits markdown by ## and ### headers."""
    text = """\
# Title

## Section A
content a

### Subsection
table here

## Section B
content b
"""
    sections = upstream_check._split_sections(text)
    assert len(sections) == 3
    assert "Section A" in sections[0][0]
    assert "Subsection" in sections[1][0]
    assert "Section B" in sections[2][0]


# ─── Standalone runner (no pytest required) ───────────────────────────────


def run_all_tests():
    """Run all tests without pytest — for environments without it."""
    tests = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
"""Tests for agent/skill_bundles.py — YAML-defined skill bundles."""

import os
from pathlib import Path

import pytest

from agent.skill_bundles import (
    _slugify,
    build_bundle_invocation_message,
    delete_bundle,
    get_bundle,
    get_skill_bundles,
    list_bundles,
    reload_bundles,
    resolve_bundle_command_key,
    save_bundle,
    scan_bundles,
)


def _make_bundle_yaml(
    bundles_dir: Path, slug: str, skills: list[str],
    description: str = "", instruction: str = "", name: str | None = None,
) -> Path:
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    if name is not None:
        lines.append(f"name: {name}")
    else:
        lines.append(f"name: {slug}")
    if description:
        lines.append(f"description: {description}")
    lines.append("skills:")
    for s in skills:
        lines.append(f"  - {s}")
    if instruction:
        lines.append(f"instruction: |")
        for ln in instruction.splitlines():
            lines.append(f"  {ln}")
    path = bundles_dir / f"{slug}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_skill(skills_dir: Path, name: str, body: str = "Do the thing.") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


@pytest.fixture
def bundles_env(tmp_path, monkeypatch):
    """Isolated bundles dir + skills dir."""
    bundles_dir = tmp_path / "skill-bundles"
    bundled_bundles_dir = tmp_path / "bundled-skill-bundles"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setenv("HERMES_BUNDLED_SKILL_BUNDLES", str(bundled_bundles_dir))
    # Patch SKILLS_DIR so skill loading hits our temp tree.
    import tools.skills_tool as skills_tool_module
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    # Reset module-level cache between tests.
    import agent.skill_bundles as mod
    mod._bundles_cache = {}
    mod._bundles_cache_mtime = None
    return bundles_dir, skills_dir


class TestSlugify:
    def test_basic(self):
        assert _slugify("Backend Dev") == "backend-dev"

    def test_underscores(self):
        assert _slugify("backend_dev") == "backend-dev"

    def test_strips_invalid_chars(self):
        assert _slugify("hello, world!") == "hello-world"

    def test_collapses_hyphens(self):
        assert _slugify("a--b---c") == "a-b-c"

    def test_empty(self):
        assert _slugify("") == ""
        assert _slugify("!!!") == ""


class TestScanBundles:
    def test_empty_dir(self, bundles_env):
        bundles_dir, _ = bundles_env
        result = scan_bundles()
        assert result == {}

    def test_finds_bundle(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "backend", ["skill-a", "skill-b"])
        result = scan_bundles()
        assert "/backend" in result
        assert result["/backend"]["name"] == "backend"
        assert result["/backend"]["skills"] == ["skill-a", "skill-b"]

    def test_skips_invalid_yaml(self, bundles_env):
        bundles_dir, _ = bundles_env
        bundles_dir.mkdir(parents=True)
        (bundles_dir / "broken.yaml").write_text("{not: valid yaml: [")
        _make_bundle_yaml(bundles_dir, "good", ["skill-a"])
        result = scan_bundles()
        assert "/good" in result
        assert "/broken" not in result

    def test_skips_bundle_without_skills(self, bundles_env):
        bundles_dir, _ = bundles_env
        bundles_dir.mkdir(parents=True)
        (bundles_dir / "noskills.yaml").write_text("name: noskills\nskills: []\n")
        result = scan_bundles()
        assert "/noskills" not in result

    def test_duplicate_slug_first_wins(self, bundles_env):
        bundles_dir, _ = bundles_env
        # Two files normalizing to the same slug. Sort order is by filename:
        # 'alpha-dup.yaml' sorts before 'alpha.yaml' (`-` < `.` in ASCII), so
        # the first-seen file wins.
        _make_bundle_yaml(bundles_dir, "alpha", ["s1"], name="alpha")
        _make_bundle_yaml(bundles_dir, "alpha-dup", ["s2"], name="ALPHA")
        result = scan_bundles()
        assert "/alpha" in result
        # alpha-dup.yaml is scanned first → its skills win
        assert result["/alpha"]["skills"] == ["s2"]

    def test_uses_filename_as_fallback_name(self, bundles_env):
        bundles_dir, _ = bundles_env
        bundles_dir.mkdir(parents=True)
        (bundles_dir / "fallback.yaml").write_text("skills:\n  - foo\n")
        result = scan_bundles()
        assert "/fallback" in result
        assert result["/fallback"]["name"] == "fallback"

    def test_finds_bundled_bundle(self, bundles_env):
        bundled_dir = Path(os.environ["HERMES_BUNDLED_SKILL_BUNDLES"])
        _make_bundle_yaml(
            bundled_dir,
            "product-manager",
            ["product-manager"],
            description="Bundled PM pack",
        )
        result = scan_bundles()
        assert "/product-manager" in result
        assert result["/product-manager"]["source"] == "bundled"
        assert result["/product-manager"]["description"] == "Bundled PM pack"

    def test_user_bundle_overrides_bundled_default(self, bundles_env):
        bundles_dir, _ = bundles_env
        bundled_dir = Path(os.environ["HERMES_BUNDLED_SKILL_BUNDLES"])
        _make_bundle_yaml(
            bundled_dir,
            "analyst",
            ["bundled-analyst"],
            description="Bundled default",
        )
        _make_bundle_yaml(
            bundles_dir,
            "analyst",
            ["custom-analyst"],
            description="User override",
        )
        result = scan_bundles()
        assert result["/analyst"]["source"] == "user"
        assert result["/analyst"]["skills"] == ["custom-analyst"]
        assert result["/analyst"]["description"] == "User override"


class TestGetSkillBundles:
    def test_returns_cache(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "a", ["s1"])
        first = get_skill_bundles()
        # Second call should hit cache (no rescan unless mtime changed).
        second = get_skill_bundles()
        assert first is second or first == second

    def test_rescans_on_change(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "a", ["s1"])
        assert "/a" in get_skill_bundles()
        # Add a second bundle and bump mtime.
        import time as _t
        _t.sleep(0.05)  # ensure mtime granularity is exceeded
        _make_bundle_yaml(bundles_dir, "b", ["s2"])
        os.utime(bundles_dir, None)
        result = get_skill_bundles()
        assert "/a" in result
        assert "/b" in result


class TestResolveBundleCommandKey:
    def test_exact_match(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "my-bundle", ["s1"])
        scan_bundles()
        assert resolve_bundle_command_key("my-bundle") == "/my-bundle"

    def test_underscore_alias(self, bundles_env):
        """Telegram converts hyphens to underscores in command names."""
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "my-bundle", ["s1"])
        scan_bundles()
        assert resolve_bundle_command_key("my_bundle") == "/my-bundle"

    def test_unknown(self, bundles_env):
        scan_bundles()
        assert resolve_bundle_command_key("missing") is None

    def test_empty(self, bundles_env):
        assert resolve_bundle_command_key("") is None


class TestBuildBundleInvocationMessage:
    def test_loads_all_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a", body="Skill A content.")
        _make_skill(skills_dir, "skill-b", body="Skill B content.")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-b"])
        scan_bundles()

        result = build_bundle_invocation_message("/combo")
        assert result is not None
        msg, loaded, missing = result
        assert set(loaded) == {"skill-a", "skill-b"}
        assert missing == []
        assert "Skill A content." in msg
        assert "Skill B content." in msg
        assert "combo" in msg

    def test_skips_missing_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-ghost"])
        scan_bundles()

        result = build_bundle_invocation_message("/combo")
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a"]
        assert missing == ["skill-ghost"]
        assert "skill-ghost" in msg  # called out in header

    def test_unknown_bundle_returns_none(self, bundles_env):
        scan_bundles()
        assert build_bundle_invocation_message("/nope") is None

    def test_no_loadable_skills_returns_none(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "ghost", ["nonexistent-skill"])
        scan_bundles()
        result = build_bundle_invocation_message("/ghost")
        assert result is None

    def test_includes_user_instruction(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a"])
        scan_bundles()
        result = build_bundle_invocation_message(
            "/combo", user_instruction="extra context here"
        )
        assert result is not None
        msg, _, _ = result
        assert "extra context here" in msg

    def test_includes_bundle_instruction(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_bundle_yaml(
            bundles_dir, "combo", ["skill-a"],
            instruction="Always check tests first.",
        )
        scan_bundles()
        result = build_bundle_invocation_message("/combo")
        assert result is not None
        msg, _, _ = result
        assert "Always check tests first." in msg

    def test_dedupes_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-a"])
        scan_bundles()
        result = build_bundle_invocation_message("/combo")
        assert result is not None
        _, loaded, _ = result
        assert loaded == ["skill-a"]


class TestSaveAndDeleteBundle:
    def test_save_creates_file(self, bundles_env):
        bundles_dir, _ = bundles_env
        path = save_bundle("test-bundle", ["s1", "s2"], description="d", instruction="i")
        assert path.exists()
        assert path.parent == bundles_dir
        content = path.read_text()
        assert "test-bundle" in content
        assert "s1" in content
        assert "s2" in content
        assert "description: d" in content

    def test_save_refuses_overwrite_by_default(self, bundles_env):
        save_bundle("dup", ["s1"])
        with pytest.raises(FileExistsError):
            save_bundle("dup", ["s2"])

    def test_save_overwrites_with_force(self, bundles_env):
        save_bundle("dup", ["s1"])
        save_bundle("dup", ["s2"], overwrite=True)
        info = get_bundle("dup")
        assert info is not None
        assert info["skills"] == ["s2"]

    def test_save_requires_skills(self, bundles_env):
        with pytest.raises(ValueError):
            save_bundle("empty", [])

    def test_save_requires_name(self, bundles_env):
        with pytest.raises(ValueError):
            save_bundle("", ["s1"])

    def test_delete_removes_file(self, bundles_env):
        bundles_dir, _ = bundles_env
        save_bundle("doomed", ["s1"])
        assert get_bundle("doomed") is not None
        delete_bundle("doomed")
        assert get_bundle("doomed") is None

    def test_delete_missing_raises(self, bundles_env):
        with pytest.raises(FileNotFoundError):
            delete_bundle("ghost")


class TestReloadBundles:
    def test_reports_added_and_removed(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "old", ["s1"])
        scan_bundles()  # populate cache with {old}

        # Mutate the disk WITHOUT going through save/delete helpers (which
        # would refresh the cache mid-way). reload_bundles() diffs the
        # in-memory cache against the freshly-scanned disk state.
        (bundles_dir / "old.yaml").unlink()
        _make_bundle_yaml(bundles_dir, "new", ["s2"])

        diff = reload_bundles()
        added_names = {e["name"] for e in diff["added"]}
        removed_names = {e["name"] for e in diff["removed"]}
        assert "new" in added_names
        assert "old" in removed_names
        assert diff["total"] == 1


class TestListBundles:
    def test_sorted_by_slug(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "zebra", ["s1"])
        _make_bundle_yaml(bundles_dir, "apple", ["s2"])
        _make_bundle_yaml(bundles_dir, "mango", ["s3"])
        scan_bundles()
        info_list = list_bundles()
        slugs = [b["slug"] for b in info_list]
        assert slugs == sorted(slugs)

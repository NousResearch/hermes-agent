"""Tests for agent/skill_utils.py."""

from unittest.mock import patch

from agent import skill_utils
from agent.skill_utils import (
    _detect_environment,
    _normalize_string_set,
    _resolve_dotpath,
    discover_all_skill_config_vars,
    extract_skill_config_vars,
    extract_skill_conditions,
    extract_skill_description,
    get_disabled_skill_names,
    get_external_skills_dirs,
    is_excluded_skill_path,
    is_external_skill_path,
    is_skill_support_path,
    is_valid_namespace,
    iter_skill_index_files,
    parse_frontmatter,
    parse_qualified_name,
    resolve_skill_config_values,
    skill_matches_environment,
    skill_matches_platform,
    skill_matches_platform_list,
)


def test_is_excluded_skill_path_accepts_paths_and_strings(tmp_path):
    assert is_excluded_skill_path(tmp_path / "skill" / "SKILL.md") is False
    assert is_excluded_skill_path(tmp_path / ".git" / "hooks" / "SKILL.md") is True
    assert is_excluded_skill_path("pkg/node_modules/dep/SKILL.md") is True


def test_parse_frontmatter_full_yaml_and_missing_frontmatter():
    content = (
        "---\n"
        "name: demo\n"
        "description: Test skill\n"
        "metadata:\n"
        "  hermes:\n"
        "    requires_tools:\n"
        "      - read_file\n"
        "---\n"
        "# Body\n"
    )

    frontmatter, body = parse_frontmatter(content)

    assert frontmatter["name"] == "demo"
    assert frontmatter["metadata"]["hermes"]["requires_tools"] == ["read_file"]
    assert body == "# Body\n"
    assert parse_frontmatter("# No frontmatter") == ({}, "# No frontmatter")
    assert parse_frontmatter("---\nname: demo\n") == ({}, "---\nname: demo\n")


def test_parse_frontmatter_falls_back_for_malformed_yaml(monkeypatch):
    monkeypatch.setattr(
        skill_utils,
        "yaml_load",
        lambda _content: (_ for _ in ()).throw(ValueError("bad yaml")),
    )

    frontmatter, body = parse_frontmatter(
        "---\n"
        "name: demo\n"
        "description: fallback parser\n"
        "not-a-pair\n"
        "---\n"
        "body"
    )

    assert frontmatter == {"name": "demo", "description": "fallback parser"}
    assert body == "body"


def test_metadata_as_dict_with_hermes():
    """Normal case: metadata is a dict containing hermes keys."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": ["toolset_a"],
                "requires_toolsets": ["toolset_b"],
                "fallback_for_tools": ["tool_x"],
                "requires_tools": ["tool_y"],
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result["fallback_for_toolsets"] == ["toolset_a"]
    assert result["requires_toolsets"] == ["toolset_b"]
    assert result["fallback_for_tools"] == ["tool_x"]
    assert result["requires_tools"] == ["tool_y"]


def test_metadata_as_string_does_not_crash():
    """Bug case: metadata is a non-dict truthy value (e.g. a YAML string)."""
    frontmatter = {"metadata": "some text"}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_as_none():
    """metadata key is present but set to null/None."""
    frontmatter = {"metadata": None}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_metadata_missing_entirely():
    """metadata key is absent from frontmatter."""
    frontmatter = {"name": "my-skill", "description": "Does stuff."}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_iter_skill_index_files_prunes_dependency_dirs(tmp_path):
    real = tmp_path / "real-skill"
    real.mkdir()
    (real / "SKILL.md").write_text("---\nname: real-skill\n---\n", encoding="utf-8")

    nested = (
        tmp_path
        / "bring"
        / "scripts"
        / ".venv"
        / "lib"
        / "python3.13"
        / "site-packages"
        / "typer"
        / ".agents"
        / "skills"
        / "typer"
    )
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text("---\nname: typer\n---\n", encoding="utf-8")

    node_module = (
        tmp_path
        / "web-skill"
        / "node_modules"
        / "dep"
        / ".agents"
        / "skills"
        / "dep"
    )
    node_module.mkdir(parents=True)
    (node_module / "SKILL.md").write_text("---\nname: dep\n---\n", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [real / "SKILL.md"]


def test_skill_config_helpers_share_raw_config_parse_cache(tmp_path, monkeypatch):
    """Repeated skill config helpers should parse config.yaml only once."""
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    external = tmp_path / "external-skills"
    external.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        f"""
skills:
  disabled:
    - hidden-skill
  external_dirs:
    - {external}
  config:
    wiki:
      path: ~/wiki
""".strip(),
        encoding="utf-8",
    )
    parse_count = 0
    real_yaml_load = skill_utils.yaml_load

    def counting_yaml_load(text):
        nonlocal parse_count
        parse_count += 1
        return real_yaml_load(text)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()
    getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()
    monkeypatch.setattr(skill_utils, "yaml_load", counting_yaml_load)

    assert get_disabled_skill_names() == {"hidden-skill"}
    assert get_external_skills_dirs() == [external.resolve()]
    assert resolve_skill_config_values([
        {"key": "wiki.path", "description": "Wiki path"}
    ])["wiki.path"].endswith("/wiki")
    assert parse_count == 1


def test_skill_config_raw_cache_invalidates_on_config_edit(tmp_path, monkeypatch):
    """Editing config.yaml should invalidate the shared raw config cache."""
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text("skills:\n  disabled: [old-skill]\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()
    assert get_disabled_skill_names() == {"old-skill"}

    config_path.write_text("skills:\n  disabled: [new-skill]\n", encoding="utf-8")
    import os
    os.utime(config_path, None)

    assert get_disabled_skill_names() == {"new-skill"}


def test_is_external_skill_path_matches_configured_external_dir(tmp_path, monkeypatch):
    from agent import skill_utils

    hermes_home = tmp_path / ".hermes"
    local_skills = hermes_home / "skills"
    external = tmp_path / "external-skills"
    local_skills.mkdir(parents=True)
    external.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()

    assert is_external_skill_path(external / "team-skill" / "SKILL.md") is True
    assert is_external_skill_path(local_skills / "local-skill" / "SKILL.md") is False


def test_iter_skill_index_files_prunes_skill_support_dirs(tmp_path):
    """Archived package SKILL.md files under support dirs are not active skills."""
    real = tmp_path / "umbrella"
    real.mkdir()
    (real / "SKILL.md").write_text("---\nname: umbrella\n---\n", encoding="utf-8")

    package = real / "references" / "old-skill-package"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("---\nname: old-skill\n---\n", encoding="utf-8")
    (package / "DESCRIPTION.md").write_text(
        "---\ndescription: archived package\n---\n", encoding="utf-8"
    )

    script_package = real / "scripts" / "helper-skill"
    script_package.mkdir(parents=True)
    (script_package / "SKILL.md").write_text("---\nname: helper\n---\n", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))
    desc_found = list(iter_skill_index_files(tmp_path, "DESCRIPTION.md"))

    assert found == [real / "SKILL.md"]
    assert desc_found == []
    assert is_skill_support_path(package / "SKILL.md") is True
    assert is_excluded_skill_path(package / "SKILL.md") is True


def test_iter_skill_index_files_keeps_support_named_categories(tmp_path):
    """A category named scripts/templates/assets/references is still valid."""
    scripts_skill = tmp_path / "scripts" / "bash-helper"
    scripts_skill.mkdir(parents=True)
    (scripts_skill / "SKILL.md").write_text(
        "---\nname: bash-helper\n---\n", encoding="utf-8"
    )

    templates_skill = tmp_path / "templates" / "deck-template"
    templates_skill.mkdir(parents=True)
    (templates_skill / "SKILL.md").write_text(
        "---\nname: deck-template\n---\n", encoding="utf-8"
    )

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [scripts_skill / "SKILL.md", templates_skill / "SKILL.md"]
    assert is_skill_support_path(scripts_skill / "SKILL.md") is False
    assert is_excluded_skill_path(scripts_skill / "SKILL.md") is False


def test_environment_matching_defaults_unknowns_and_empty_tags(monkeypatch):
    assert skill_matches_environment({}) is True
    assert skill_matches_environment({"environments": "unknown-runtime"}) is True

    calls: list[str] = []

    def fake_detect(env):
        calls.append(env)
        return env == "docker"

    monkeypatch.setattr(skill_utils, "_detect_environment", fake_detect)

    assert skill_matches_environment({"environments": ["", "s6"]}) is False
    assert calls == ["s6"]
    assert skill_matches_environment({"environments": ["kanban", "docker"]}) is True
    assert calls[-2:] == ["kanban", "docker"]


def test_detect_environment_uses_cache_and_runtime_markers(monkeypatch):
    skill_utils._ENV_DETECT_CACHE.clear()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    assert _detect_environment("kanban") is True
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _detect_environment("kanban") is True

    skill_utils._ENV_DETECT_CACHE.clear()
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
    assert _detect_environment("docker") is True

    skill_utils._ENV_DETECT_CACHE.clear()
    monkeypatch.setattr(
        skill_utils.os.path,
        "isdir",
        lambda path: path == "/package/admin/s6-overlay",
    )
    assert _detect_environment("s6") is True


def test_normalize_string_set_handles_strings_lists_and_empty_values():
    assert _normalize_string_set(None) == set()
    assert _normalize_string_set(" one ") == {"one"}
    assert _normalize_string_set([" one ", "", None, 2]) == {"one", "None", "2"}


def test_get_disabled_skill_names_handles_invalid_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    monkeypatch.setattr(skill_utils, "get_config_path", lambda: config)

    assert skill_utils.get_disabled_skill_names() == set()

    config.write_text("not: [valid", encoding="utf-8")
    assert skill_utils.get_disabled_skill_names() == set()

    config.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert skill_utils.get_disabled_skill_names() == set()

    config.write_text("skills: disabled\n", encoding="utf-8")
    assert skill_utils.get_disabled_skill_names() == set()


def test_get_disabled_skill_names_prefers_platform_specific_config(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "skills:\n"
        "  disabled:\n"
        "    - global-skill\n"
        "  platform_disabled:\n"
        "    telegram:\n"
        "      - telegram-skill\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_utils, "get_config_path", lambda: config)
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)

    assert skill_utils.get_disabled_skill_names(platform="telegram") == {
        "global-skill",
        "telegram-skill",
    }


def test_extract_skill_config_vars_normalizes_and_deduplicates():
    frontmatter = {
        "metadata": {
            "hermes": {
                "config": [
                    {
                        "key": "wiki.path",
                        "description": " Wiki directory ",
                        "default": "~/wiki",
                        "prompt": " Choose wiki ",
                    },
                    {"key": "wiki.path", "description": "duplicate"},
                    {"key": "", "description": "missing key"},
                    {"key": "bad.description"},
                    "not-a-dict",
                ]
            }
        }
    }

    assert extract_skill_config_vars(frontmatter) == [
        {
            "key": "wiki.path",
            "description": "Wiki directory",
            "default": "~/wiki",
            "prompt": "Choose wiki",
        }
    ]
    assert extract_skill_config_vars({"metadata": "bad"}) == []
    assert extract_skill_config_vars({"metadata": {"hermes": "bad"}}) == []
    assert extract_skill_config_vars({"metadata": {"hermes": {"config": "bad"}}}) == []


def test_discover_all_skill_config_vars_filters_and_deduplicates(tmp_path, monkeypatch):
    enabled = tmp_path / "a-enabled"
    enabled.mkdir()
    (enabled / "SKILL.md").write_text(
        "---\n"
        "name: enabled-skill\n"
        "metadata:\n"
        "  hermes:\n"
        "    config:\n"
        "      - key: wiki.path\n"
        "        description: Wiki path\n"
        "---\n",
        encoding="utf-8",
    )
    duplicate = tmp_path / "b-duplicate"
    duplicate.mkdir()
    (duplicate / "SKILL.md").write_text(
        "---\n"
        "name: duplicate-skill\n"
        "metadata:\n"
        "  hermes:\n"
        "    config:\n"
        "      - key: wiki.path\n"
        "        description: Duplicate path\n"
        "---\n",
        encoding="utf-8",
    )
    disabled = tmp_path / "disabled"
    disabled.mkdir()
    (disabled / "SKILL.md").write_text(
        "---\n"
        "name: disabled-skill\n"
        "metadata:\n"
        "  hermes:\n"
        "    config:\n"
        "      - key: disabled.path\n"
        "        description: Disabled path\n"
        "---\n",
        encoding="utf-8",
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_bytes(b"\xff")
    missing = tmp_path / "missing"

    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [missing, tmp_path])
    monkeypatch.setattr(
        skill_utils,
        "get_disabled_skill_names",
        lambda: {"disabled-skill"},
    )
    monkeypatch.setattr(skill_utils, "skill_matches_platform", lambda _fm: True)

    assert discover_all_skill_config_vars() == [
        {
            "key": "wiki.path",
            "description": "Wiki path",
            "prompt": "Wiki path",
            "skill": "enabled-skill",
        }
    ]


def test_resolve_skill_config_values_prefers_config_and_expands_paths(
    tmp_path,
    monkeypatch,
):
    config = tmp_path / "config.yaml"
    config.write_text(
        "skills:\n"
        "  config:\n"
        "    wiki:\n"
        "      path: ${WIKI_ROOT}/notes\n"
        "    empty:\n"
        "      path: '   '\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_utils, "get_config_path", lambda: config)
    monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "wiki"))

    resolved = resolve_skill_config_values(
        [
            {"key": "wiki.path", "default": "~/fallback"},
            {"key": "empty.path", "default": "~/empty-default"},
            {"key": "missing.path", "default": "plain-default"},
        ]
    )

    assert resolved["wiki.path"] == str(tmp_path / "wiki" / "notes")
    assert resolved["empty.path"].endswith("/empty-default")
    assert resolved["missing.path"] == "plain-default"


def test_resolve_dotpath_and_description_and_namespace_helpers():
    config = {"skills": {"config": {"wiki": {"path": "/tmp/wiki"}}}}

    assert _resolve_dotpath(config, "skills.config.wiki.path") == "/tmp/wiki"
    assert _resolve_dotpath(config, "skills.config.missing") is None
    assert extract_skill_description({}) == ""
    assert extract_skill_description({"description": "  'short text'  "}) == "short text"
    assert extract_skill_description({"description": "x" * 80}) == ("x" * 57) + "..."
    assert parse_qualified_name("github:pull-request") == ("github", "pull-request")
    assert parse_qualified_name("local-skill") == (None, "local-skill")
    assert is_valid_namespace("github-1") is True
    assert is_valid_namespace("bad namespace") is False
    assert is_valid_namespace(None) is False


# ── skill_matches_platform on Termux ──────────────────────────────────────


class TestSkillMatchesPlatformTermux:
    """Termux is Linux userland on Android. Skills tagged platforms:[linux]
    must load there regardless of whether Python reports sys.platform as
    "linux" (pre-3.13) or "android" (3.13+). Reported by user @LikiusInik
    in May 2026 — only 3 built-in skills appeared on Termux because every
    github/productivity/mlops skill is tagged platforms:[linux,macos,windows]
    and sys.platform=="android" did not start with "linux".
    """

    def test_no_platforms_field_matches_everywhere(self):
        # Backward-compat default — skills without a platforms tag load
        # on any OS, Termux included.
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform({}) is True
            assert skill_matches_platform({"name": "foo"}) is True

    def test_linux_skill_loads_on_termux_android_platform(self):
        # Python 3.13+ on Termux reports sys.platform == "android".
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_linux_macos_windows_skill_loads_on_termux(self):
        # The common "[linux, macos, windows]" tag used by github-*,
        # productivity, mlops, etc.
        fm = {"platforms": ["linux", "macos", "windows"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_linux_skill_loads_on_termux_linux_platform(self):
        # Pre-3.13 Termux reports sys.platform == "linux" already — this
        # works without the Termux escape hatch but must still pass.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_macos_only_skill_still_excluded_on_termux(self):
        # macOS-only skills (apple-notes, imessage, ...) should NOT load
        # on Termux. The Termux fallback only widens platforms:[linux,...].
        fm = {"platforms": ["macos"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_windows_only_skill_still_excluded_on_termux(self):
        fm = {"platforms": ["windows"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_explicit_termux_or_android_tag_matches(self):
        # Skills can also opt in explicitly via platforms:[termux] or
        # platforms:[android] — both should match a Termux session.
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=True
        ):
            assert skill_matches_platform({"platforms": ["termux"]}) is True
            assert skill_matches_platform({"platforms": ["android"]}) is True
            assert skill_matches_platform_list(["termux"]) is True
            assert skill_matches_platform_list(["android"]) is True

    def test_non_termux_android_does_not_widen(self):
        # If we're somehow on a plain Android Python (not Termux), don't
        # silently load Linux skills — Termux is the supported environment.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "android"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is False
            assert skill_matches_platform_list(fm["platforms"]) is False

    def test_linux_skill_on_real_linux_unaffected(self):
        # The non-Termux Linux path must not change.
        fm = {"platforms": ["linux"]}
        with patch("agent.skill_utils.sys.platform", "linux"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True

    def test_macos_skill_on_real_macos_unaffected(self):
        fm = {"platforms": ["macos"]}
        with patch("agent.skill_utils.sys.platform", "darwin"), patch(
            "agent.skill_utils.is_termux", return_value=False
        ):
            assert skill_matches_platform(fm) is True
            assert skill_matches_platform_list(fm["platforms"]) is True


class TestNormalizeSkillLookupName:
    def test_relative_path_unchanged(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        # Relative identifiers early-return before any root lookup.
        assert normalize_skill_lookup_name("foo/bar") == "foo/bar"

    def test_absolute_under_skills_dir_becomes_relative(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "category" / "my-skill"
        skill_dir.mkdir(parents=True)
        # Patch the root skill_view() itself enforces — normalization reads
        # tools.skills_tool.SKILLS_DIR at call time so the two stay in sync.
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
        assert normalize_skill_lookup_name(str(skill_dir)) == "category/my-skill"

    def test_absolute_via_symlink_uses_lexical_relative_path(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        external = tmp_path / "external" / "my-skill"
        external.mkdir(parents=True)
        link = skills_dir / "my-skill"
        try:
            link.symlink_to(external)
        except OSError:
            pytest.skip("Symlinks not supported")
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
        assert normalize_skill_lookup_name(str(link)) == "my-skill"

    def test_untrusted_absolute_returned_unchanged(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr("agent.skill_utils.get_skills_dir", lambda: tmp_path / "skills")
        outside = str(tmp_path / "outside" / "skill")
        assert normalize_skill_lookup_name(outside) == outside

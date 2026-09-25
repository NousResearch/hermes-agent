"""Tests for agent/skill_utils.py."""


import pytest

from agent.skill_utils import (
    get_disabled_skill_names,
    get_external_skills_dirs,
    is_excluded_skill_path,
    is_skill_support_path,
    iter_skill_index_files,
    parse_config_string_list,
    parse_frontmatter,
    resolve_skill_config_values,
    skill_matches_platform,
)












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
    import utils
    parse_count = 0
    real_load = utils.fast_safe_load

    def counting_load(stream):
        nonlocal parse_count
        parse_count += 1
        return real_load(stream)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    skill_utils._external_dirs_cache_clear()
    getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()
    # The file config backend parses through utils.fast_safe_load.
    monkeypatch.setattr(utils, "fast_safe_load", counting_load)

    assert get_disabled_skill_names() == {"hidden-skill"}
    assert get_external_skills_dirs() == [external.resolve()]
    assert resolve_skill_config_values([
        {"key": "wiki.path", "description": "Wiki path"}
    ])["wiki.path"].endswith("/wiki")
    assert parse_count == 1


def _write_disabled_config(home, disabled: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"skills:\n  disabled:\n    - {disabled}\n", encoding="utf-8"
    )


class TestMultiprofileRawConfigCache:
    """#141610: a multiplexed watcher rotates through up to ~32 published profile
    scopes plus the ambient one; one profile's cache miss must not evict every
    other profile's unchanged entry."""

    @staticmethod
    def _load_under(home) -> list:
        from agent import skill_utils
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(str(home))
        try:
            return skill_utils._load_raw_config()["skills"]["disabled"]
        finally:
            reset_hermes_home_override(token)

    def test_watcher_rotation_reparses_no_unchanged_config(self, tmp_path, monkeypatch):
        from agent import skill_utils

        homes = [tmp_path / f"scope-{i:02d}" for i in range(33)]
        for i, home in enumerate(homes):
            _write_disabled_config(home, f"skill-{i:02d}")
        parse_count = 0
        real_yaml_load = skill_utils.yaml_load

        def counting_yaml_load(text):
            nonlocal parse_count
            parse_count += 1
            return real_yaml_load(text)

        monkeypatch.setattr(skill_utils, "yaml_load", counting_yaml_load)
        skill_utils._raw_config_cache_clear()

        for _ in range(3):
            for i, home in enumerate(homes):
                assert self._load_under(home) == [f"skill-{i:02d}"]

        assert parse_count == 33

    def test_editing_one_profile_reparses_only_that_profile(
        self, tmp_path, monkeypatch
    ):
        from agent import skill_utils

        first, second = tmp_path / "first", tmp_path / "second"
        _write_disabled_config(first, "old-disabled")
        _write_disabled_config(second, "second-disabled")
        parse_count = 0
        real_yaml_load = skill_utils.yaml_load

        def counting_yaml_load(text):
            nonlocal parse_count
            parse_count += 1
            return real_yaml_load(text)

        monkeypatch.setattr(skill_utils, "yaml_load", counting_yaml_load)
        skill_utils._raw_config_cache_clear()
        assert self._load_under(first) == ["old-disabled"]
        assert self._load_under(second) == ["second-disabled"]

        # A different length changes st_size, so the signature must change even on
        # filesystems whose mtime granularity misses a quick rewrite.
        (first / "config.yaml").write_text(
            "skills:\n  disabled:\n    - new-disabled-name\n", encoding="utf-8"
        )
        assert self._load_under(first) == ["new-disabled-name"]
        assert self._load_under(second) == ["second-disabled"]
        assert parse_count == 3

    def test_one_profile_parse_does_not_block_another_profile(
        self, tmp_path, monkeypatch
    ):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        from agent import skill_utils

        first, second = tmp_path / "first", tmp_path / "second"
        _write_disabled_config(first, "first-disabled")
        _write_disabled_config(second, "second-disabled")
        entered, release = Event(), Event()
        real_yaml_load = skill_utils.yaml_load

        def delayed_yaml_load(text):
            if "first-disabled" in text:
                entered.set()
                assert release.wait(5), "first profile was not released"
            return real_yaml_load(text)

        monkeypatch.setattr(skill_utils, "yaml_load", delayed_yaml_load)
        skill_utils._raw_config_cache_clear()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_result = pool.submit(self._load_under, first)
            try:
                assert entered.wait(5), "first profile parse did not start"
                second_result = pool.submit(self._load_under, second)
                assert second_result.result(timeout=3) == ["second-disabled"]
            finally:
                release.set()
            assert first_result.result(timeout=5) == ["first-disabled"]
        assert self._load_under(first) == ["first-disabled"]
        assert self._load_under(second) == ["second-disabled"]

    def test_concurrent_loads_across_evicting_homes_never_raise(
        self, tmp_path, monkeypatch
    ):
        """Tool threads share the cache: two threads evicting the same oldest key must not
        raise KeyError, nor an insert during ``next(iter(...))`` a RuntimeError. A 1 µs
        switch interval makes the unlocked interleaving reliable instead of rare."""
        import sys
        from concurrent.futures import ThreadPoolExecutor

        from agent import skill_utils

        homes = [tmp_path / f"scope-{i:02d}" for i in range(24)]
        for i, home in enumerate(homes):
            _write_disabled_config(home, f"skill-{i:02d}")
        monkeypatch.setattr(skill_utils, "_RAW_CONFIG_CACHE_MAX", 2)
        skill_utils._raw_config_cache_clear()

        def hammer(offset: int) -> None:
            for n in range(150):
                i = (offset + n) % len(homes)
                assert self._load_under(homes[i]) == [f"skill-{i:02d}"]

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(hammer, t) for t in range(16)]
                errors = [f.exception(timeout=60) for f in futures]
        finally:
            sys.setswitchinterval(old_interval)
        assert errors == [None] * 16


class TestParseConfigStringList:
    """#86661: `hermes config set` and JSON-mode editor saves store lists as
    quoted strings (e.g. '["a","b"]'). Treating such a string as a single name
    made curated disabled lists silently filter nothing."""

    def test_json_array_string_parses(self):
        assert parse_config_string_list('["skill-a","skill-b"]') == [
            "skill-a",
            "skill-b",
        ]

    def test_python_literal_array_string_parses(self):
        # `hermes config set` can persist single-quoted Python-literal forms.
        assert parse_config_string_list("['skill-a']") == ["skill-a"]

    def test_scalar_string_means_one_name(self):
        # #13026: a scalar string still names a single entry.
        assert parse_config_string_list("skill-a") == ["skill-a"]

    def test_real_list_passes_through(self):
        assert parse_config_string_list(["skill-a", "skill-b"]) == [
            "skill-a",
            "skill-b",
        ]
        assert parse_config_string_list(("skill-a",)) == ["skill-a"]

    def test_none_returns_empty(self):
        assert parse_config_string_list(None) == []

    def test_malformed_json_falls_back_to_single_name(self):
        assert parse_config_string_list('["skill-a"') == ['["skill-a"']

    def test_empty_array_string_returns_empty(self):
        assert parse_config_string_list("[]") == []


class TestDisabledSkillsJsonArrayString:
    """The skills.disabled setting must honor a JSON-array string form, not
    treat the whole string as one dead skill name (#86661)."""

    def test_get_disabled_skill_names_parses_json_array_string(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "skills:\n  disabled: '[\"skill-a\",\"skill-b\"]'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from agent import skill_utils

        getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()

        assert get_disabled_skill_names() == {"skill-a", "skill-b"}



def test_skill_config_home_vars_use_subprocess_home(tmp_path, monkeypatch):
    """``~`` / ``$HOME`` / ``${HOME}`` defaults resolve against the HOME tools receive, not the
    control process HOME; other variables keep normal expansion (#12260)."""
    from agent import skill_utils

    # A backslash in the home path must not be read as a regex-replacement escape.
    hermes_home = tmp_path / "da\\ta"
    subprocess_home = hermes_home / "home"
    subprocess_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    monkeypatch.setenv("PROJECT_ROOT", "/proj")
    monkeypatch.setenv("LEAF", "leaf")
    getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()

    resolved = resolve_skill_config_values([
        {"key": "wiki.home_var", "default": "$HOME/wiki"},
        {"key": "wiki.braced_home", "default": "${HOME}/notes"},
        {"key": "wiki.tilde", "default": "~/scratch"},
        {"key": "wiki.other_var", "default": "${PROJECT_ROOT}/cache"},
        {"key": "wiki.tilde_var", "default": "~/$LEAF"},
    ])

    assert resolved["wiki.home_var"] == str(subprocess_home / "wiki")
    assert resolved["wiki.braced_home"] == str(subprocess_home / "notes")
    assert resolved["wiki.tilde"] == str(subprocess_home / "scratch")
    assert resolved["wiki.other_var"] == "/proj/cache"
    assert resolved["wiki.tilde_var"] == str(subprocess_home / "leaf")


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


def test_skill_support_path_uses_explicit_discovery_root_not_cwd(tmp_path, monkeypatch):
    discovery_root = tmp_path / "site-packages" / "skills"
    umbrella = discovery_root / "category" / "umbrella"
    nested = umbrella / "references" / "archived" / "SKILL.md"
    nested.parent.mkdir(parents=True)
    (umbrella / "SKILL.md").write_text("---\nname: umbrella\n---\n", encoding="utf-8")
    nested.write_text("---\nname: archived\n---\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    relative = nested.relative_to(discovery_root)
    assert is_skill_support_path(relative, root=discovery_root) is True
    assert is_excluded_skill_path(relative, root=discovery_root) is True


# ── skill_matches_platform on Termux ──────────────────────────────────────





class TestNormalizeSkillLookupName:
    def test_relative_path_unchanged(self, tmp_path, monkeypatch):
        from agent.skill_utils import normalize_skill_lookup_name

        # Relative identifiers early-return before any root lookup.
        assert normalize_skill_lookup_name("foo/bar") == "foo/bar"


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



# ── parse_frontmatter: UTF-8 BOM tolerance ─────────────────────────────────


class TestParseFrontmatterBOM:
    """A UTF-8 BOM (U+FEFF) on a Windows-saved SKILL.md must not defeat
    frontmatter parsing.

    Notepad and PowerShell ``>`` prepend a BOM when saving UTF-8;
    ``read_text(encoding="utf-8")`` (what ``_parse_skill_file`` uses) keeps
    it, so the bytes handed to ``parse_frontmatter`` start with a BOM ahead of
    the ``---`` fence. Before the fix the ``startswith("---")`` check returned
    False and the whole frontmatter was silently dropped — the skill loaded
    nameless, platform gating fell open, and env-var/config setup never fired.
    """

    SKILL = (
        "---\n"
        "name: my-skill\n"
        "description: Does a thing.\n"
        "platforms: [macos]\n"
        "metadata:\n"
        "  hermes:\n"
        "    config:\n"
        "      - key: my.key\n"
        "        description: A configured value\n"
        "---\n\n"
        "# My Skill\n\nBody text.\n"
    )

    def test_bom_frontmatter_matches_plain(self):
        plain_fm, plain_body = parse_frontmatter(self.SKILL)
        bom_fm, bom_body = parse_frontmatter("\ufeff" + self.SKILL)
        assert bom_fm == plain_fm
        assert bom_body == plain_body
        assert bom_fm["name"] == "my-skill"
        assert bom_fm["description"] == "Does a thing."




    def test_bom_platform_gating_regression(self):
        # The concrete harm: a macOS-only skill must be gated identically
        # whether or not the file carries a BOM. Empty frontmatter (the bug)
        # reads as "no platform restriction" and leaks the skill everywhere,
        # i.e. it would answer True on every host. Compare against the real
        # host's verdict instead of faking Windows — the fake only stood in
        # for "some non-macOS host", which the CI host already is.
        import sys

        expected = sys.platform == "darwin"
        plain_fm, _ = parse_frontmatter(self.SKILL)
        bom_fm, _ = parse_frontmatter("\ufeff" + self.SKILL)
        assert skill_matches_platform(plain_fm) is expected
        assert skill_matches_platform(bom_fm) is expected




class TestBOMToleranceSiblingSites:
    """The BOM fix must cover every independent frontmatter parser, not just
    the canonical ``parse_frontmatter`` — several modules reimplement the
    ``---`` fence check locally."""

    SKILL = "---\nname: bom-skill\ndescription: Saved by Notepad\n---\n\n# Body\n"


    def test_prompt_builder_strips_bom_frontmatter(self):
        # A BOM'd context file (AGENTS.md etc.) must not leak raw
        # frontmatter into the system prompt.
        from agent.prompt_builder import _strip_yaml_frontmatter

        out = _strip_yaml_frontmatter("\ufeff---\nfoo: bar\n---\nBody text\n")
        assert out.strip() == "Body text"

    def test_blueprints_split_frontmatter_bom(self):
        # str.lstrip() does NOT strip U+FEFF (it is not whitespace), so the
        # pre-existing lstrip() in _split_frontmatter never covered it.
        from tools.blueprints import _split_frontmatter

        fm = _split_frontmatter("\ufeff---\nname: bp\n---\nbody")
        assert fm is not None
        assert fm.get("name") == "bp"


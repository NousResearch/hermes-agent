"""Guards for ``get_external_skills_dirs`` config loading.

``get_external_skills_dirs()`` is called once per skill during banner
construction and tool registration — on a typical install that's 120+
calls. Profile and managed loaders cache YAML parsing, while configured path
resolution and initial availability are cached separately until an explicit
refresh. These tests pin managed precedence, recovery, ownership, and
invalidation behavior.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import skill_utils
from agent.skill_utils import (
    _external_dirs_cache_clear,
    get_disabled_skill_names,
    get_external_skills_dirs,
    is_external_skill_path,
    is_project_root_trusted,
)


def _clear_skill_config_caches() -> None:
    """Reset each independently-owned cache used by these isolated fixtures."""
    from hermes_cli import config as config_mod
    from hermes_cli import managed_scope

    _external_dirs_cache_clear()
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    managed_scope.invalidate_managed_cache()


@pytest.fixture
def hermes_home_with_config(tmp_path, monkeypatch):
    """Isolated ``~/.hermes/`` with a config.yaml referencing one external dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    external = tmp_path / "external_alpha"
    external.mkdir()

    config = home / "config.yaml"
    config.write_text(
        f"skills:\n  external_dirs:\n    - {external}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _clear_skill_config_caches()
    yield home, external, config
    _clear_skill_config_caches()


@pytest.fixture
def managed_external_dirs(tmp_path, monkeypatch):
    """Isolated user and managed configs with external skill directories."""
    home = tmp_path / ".hermes"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    managed_external = tmp_path / "managed_alpha"
    managed_external.mkdir()
    user_external = tmp_path / "user_external"
    user_external.mkdir()

    (home / "config.yaml").write_text(
        "skills:\n  template_vars: true\n",
        encoding="utf-8",
    )
    managed_config = managed / "config.yaml"
    managed_config.write_text(
        f"skills:\n  external_dirs:\n    - {managed_external}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _clear_skill_config_caches()
    yield home, managed_config, managed_external, user_external
    _clear_skill_config_caches()


def test_managed_external_dirs_are_used_when_user_config_omits_them(
    managed_external_dirs,
):
    """A managed-only external_dirs value participates in skill discovery."""
    _home, _managed_config, managed_external, _user_external = managed_external_dirs

    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_managed_external_dirs_are_used_without_user_config(managed_external_dirs):
    """Managed external_dirs works when the profile config does not exist."""
    home, _managed_config, managed_external, _user_external = managed_external_dirs
    (home / "config.yaml").unlink()

    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_managed_external_dirs_override_user_config(managed_external_dirs):
    """Managed external_dirs wins over the user value at the same leaf."""
    home, _managed_config, managed_external, user_external = managed_external_dirs
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {user_external}\n",
        encoding="utf-8",
    )

    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_managed_skills_enforce_disabled_and_project_policy(
    managed_external_dirs,
    monkeypatch,
):
    """All managed skills policy leaves are enforced by behavioral readers."""
    home, managed_config, managed_external, user_external = managed_external_dirs
    (managed_external / ".git").mkdir()
    (managed_external / ".hermes" / "skills" / "project-skill").mkdir(parents=True)
    monkeypatch.chdir(managed_external)
    (home / "config.yaml").write_text(
        f"skills:\n  trusted_project_dirs:\n    - {user_external}\n",
        encoding="utf-8",
    )
    managed_config.write_text(
        "skills:\n"
        f"  external_dirs:\n    - {managed_external}\n"
        f"  trusted_project_dirs:\n    - {managed_external}\n"
        "  project_discovery: false\n"
        "  disabled: [risky-skill]\n",
        encoding="utf-8",
    )

    assert get_external_skills_dirs() == [managed_external.resolve()]
    assert get_disabled_skill_names() == {"risky-skill"}
    assert not is_project_root_trusted(user_external)
    assert is_project_root_trusted(managed_external)
    assert skill_utils.get_project_skills_dirs() == []


def test_invalid_managed_skills_shape_preserves_user_config(
    managed_external_dirs,
    caplog,
):
    """An invalid managed skills section does not discard valid user settings."""
    home, managed_config, _managed_external, user_external = managed_external_dirs
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {user_external}\n",
        encoding="utf-8",
    )
    managed_config.write_text("skills: []\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        first = get_external_skills_dirs()
        second = get_external_skills_dirs()

    warnings = [
        record
        for record in caplog.records
        if "skills must be a mapping" in record.getMessage()
    ]
    assert first == [user_external.resolve()]
    assert second == [user_external.resolve()]
    assert len(warnings) == 1


def test_managed_external_dirs_expand_environment_variables(
    managed_external_dirs,
    monkeypatch,
):
    """Managed external directories retain documented environment expansion."""
    _home, managed_config, managed_external, _user_external = managed_external_dirs
    monkeypatch.setenv("SHARED_SKILLS_ROOT", str(managed_external.parent))
    managed_config.write_text(
        'skills:\n  external_dirs:\n    - "${env:SHARED_SKILLS_ROOT}/managed_alpha"\n',
        encoding="utf-8",
    )

    assert get_external_skills_dirs() == [managed_external.resolve()]


def test_user_external_dirs_expand_prefixed_environment_variables(
    hermes_home_with_config,
    monkeypatch,
):
    """User and managed paths share the documented ${env:VAR} expansion."""
    _home, external, config = hermes_home_with_config
    monkeypatch.setenv("SHARED_SKILLS_ROOT", str(external.parent))
    config.write_text(
        'skills:\n  external_dirs:\n    - "${env:SHARED_SKILLS_ROOT}/external_alpha"\n',
        encoding="utf-8",
    )

    assert get_external_skills_dirs() == [external.resolve()]


def test_json_array_string_external_dirs_is_parsed(hermes_home_with_config):
    """Stringified list values retain config-set list compatibility."""
    _home, external, config = hermes_home_with_config
    config.write_text(
        f"skills:\n  external_dirs: '[\"{external}\"]'\n",
        encoding="utf-8",
    )

    assert get_external_skills_dirs() == [external.resolve()]


def test_malformed_platform_disabled_does_not_crash(hermes_home_with_config):
    """A malformed nested policy value is treated as absent."""
    _home, _external, config = hermes_home_with_config
    config.write_text(
        "skills:\n  disabled: [global-skill]\n  platform_disabled: broken\n",
        encoding="utf-8",
    )

    assert get_disabled_skill_names("telegram") == {"global-skill"}


@pytest.mark.parametrize("managed_value", ["[]", "null"])
def test_managed_empty_external_dirs_disable_user_value(
    managed_external_dirs,
    managed_value,
):
    """An administrator can disable user external directories wholesale."""
    home, managed_config, _managed_external, user_external = managed_external_dirs
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {user_external}\n",
        encoding="utf-8",
    )
    managed_config.write_text(
        f"skills:\n  external_dirs: {managed_value}\n",
        encoding="utf-8",
    )

    assert get_external_skills_dirs() == []


def test_managed_overlay_failure_is_fail_open(managed_external_dirs):
    """A managed loader failure preserves the usable profile config."""
    home, _managed_config, _managed_external, user_external = managed_external_dirs
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {user_external}\n",
        encoding="utf-8",
    )
    from hermes_cli import managed_scope

    with patch.object(
        managed_scope,
        "load_managed_config",
        side_effect=OSError("transient managed read failure"),
    ):
        assert get_external_skills_dirs() == [user_external.resolve()]


def test_external_cache_clear_does_not_invalidate_managed_env(
    managed_external_dirs,
):
    """Skill discovery cache resets do not touch the managed env subsystem."""
    _home, managed_config, _managed_external, _user_external = managed_external_dirs
    from hermes_cli import managed_scope

    managed_env = managed_config.parent / ".env"
    managed_env.write_text("ORG_POLICY=locked\n", encoding="utf-8")
    assert managed_scope.load_managed_env() == {"ORG_POLICY": "locked"}

    with patch.object(
        managed_scope,
        "_parse_env",
        side_effect=AssertionError("managed env cache was unexpectedly cleared"),
    ):
        _external_dirs_cache_clear()
        cached = managed_scope.load_managed_env()

    assert cached == {"ORG_POLICY": "locked"}


def test_environment_change_is_visible_without_config_edit(
    hermes_home_with_config,
    monkeypatch,
):
    """Resolved paths follow environment changes while YAML stays cached."""
    _home, external, config = hermes_home_with_config
    other = external.parent / "external_bravo"
    other.mkdir()
    config.write_text(
        "skills:\n  external_dirs:\n    - ${EXTERNAL_SKILLS_DIR}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("EXTERNAL_SKILLS_DIR", str(external))
    first = get_external_skills_dirs()
    monkeypatch.setenv("EXTERNAL_SKILLS_DIR", str(other))
    second = get_external_skills_dirs()

    assert first == [external.resolve()]
    assert second == [other.resolve()]


def test_directory_availability_changes_only_after_explicit_cache_clear(
    hermes_home_with_config,
):
    """Availability stays stable until an explicit skill-directory refresh."""
    _home, _external, config = hermes_home_with_config
    later = config.parent.parent / "external_later"
    config.write_text(
        f"skills:\n  external_dirs:\n    - {later}\n",
        encoding="utf-8",
    )

    before = get_external_skills_dirs()
    later.mkdir()
    cached = get_external_skills_dirs()
    _external_dirs_cache_clear()
    refreshed = get_external_skills_dirs()

    assert before == []
    assert cached == []
    assert refreshed == [later.resolve()]


def test_external_directory_outage_does_not_change_system_prompt(
    hermes_home_with_config,
):
    """A transient external outage does not mutate the conversation prefix."""
    home, external, _config = hermes_home_with_config
    skill_dir = external / "watch"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: watch\ndescription: Observe a process.\n---\n",
        encoding="utf-8",
    )
    from agent import prompt_builder

    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    before = prompt_builder.build_skills_system_prompt(
        skills_dir_override=home / "skills"
    )
    external.rename(external.parent / "external-offline")
    after = prompt_builder.build_skills_system_prompt(
        skills_dir_override=home / "skills"
    )

    assert "watch" in before
    assert after == before


def test_external_ownership_survives_directory_unavailability(
    hermes_home_with_config,
):
    """Configured roots remain externally owned while their storage is offline."""
    _home, external, _config = hermes_home_with_config
    skill_file = external / "watch" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("---\nname: watch\n---\n", encoding="utf-8")

    before = is_external_skill_path(skill_file)
    external.rename(external.parent / "external-offline")
    after = is_external_skill_path(skill_file)

    assert before is True
    assert after is True


def test_managed_config_mtime_invalidates_same_size_content(managed_external_dirs):
    """Managed mtime changes invalidate content whose byte size is unchanged."""
    _home, managed_config, managed_external, _user_external = managed_external_dirs
    other = managed_external.parent / "managed_bravo"
    other.mkdir()

    assert get_external_skills_dirs() == [managed_external.resolve()]

    original_size = managed_config.stat().st_size
    managed_config.write_text(
        f"skills:\n  external_dirs:\n    - {other}\n",
        encoding="utf-8",
    )
    stat = managed_config.stat()
    # This is a load-bearing precondition: equal size ensures the cache miss
    # comes from mtime rather than the size component of the signature.
    assert stat.st_size == original_size
    future_mtime = stat.st_mtime_ns + 10_000_000_000
    os.utime(managed_config, ns=(stat.st_atime_ns, future_mtime))

    assert get_external_skills_dirs() == [other.resolve()]


def test_repeated_calls_do_not_reparse_unchanged_user_config(
    managed_external_dirs,
    monkeypatch,
):
    """Profile YAML parsing stays cached across repeated calls."""
    _home, _managed_config, managed_external, _user_external = managed_external_dirs

    from hermes_cli import config as config_mod

    user_parse = config_mod.fast_safe_load
    parse_count = 0

    def count_user_parse(content):
        nonlocal parse_count
        parse_count += 1
        return user_parse(content)

    monkeypatch.setattr(config_mod, "fast_safe_load", count_user_parse)

    assert get_external_skills_dirs() == [managed_external.resolve()]
    assert get_external_skills_dirs() == [managed_external.resolve()]
    assert parse_count == 1


def test_repeated_calls_reuse_resolved_external_roots(
    hermes_home_with_config,
    monkeypatch,
):
    """Repeated discovery avoids resolving an unchanged configured root."""
    _home, external, _config = hermes_home_with_config
    expected = external.resolve()
    original_resolve = Path.resolve
    external_resolve_count = 0

    def count_external_resolve(path, *args, **kwargs):
        nonlocal external_resolve_count
        if path == external:
            external_resolve_count += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", count_external_resolve)

    first = get_external_skills_dirs()
    second = get_external_skills_dirs()

    assert first == [expected]
    assert second == [expected]
    assert external_resolve_count == 1


def test_user_config_mtime_invalidates_same_size_content(hermes_home_with_config):
    """Profile mtime changes invalidate content whose byte size is unchanged."""
    _home, external, config = hermes_home_with_config
    other = external.parent / "external_bravo"
    other.mkdir()

    # Prime cache with original contents.
    first = get_external_skills_dirs()
    assert first == [external.resolve()]

    # Rewrite config; bump mtime forward explicitly so filesystems with
    # coarse mtime granularity still register the change on fast test
    # systems.
    original_size = config.stat().st_size
    config.write_text(
        f"skills:\n  external_dirs:\n    - {other}\n",
        encoding="utf-8",
    )
    stat = config.stat()
    # This is a load-bearing precondition: equal size ensures the cache miss
    # comes from mtime rather than the size component of the signature.
    assert stat.st_size == original_size
    future_mtime = stat.st_mtime_ns + 10_000_000_000
    os.utime(config, ns=(stat.st_atime_ns, future_mtime))

    second = get_external_skills_dirs()
    assert second == [other.resolve()]


def test_profile_switch_reuses_each_profiles_resolved_external_dirs(
    tmp_path,
    monkeypatch,
):
    """A → B → A profile switching reuses both profiles' resolved roots."""
    home_a = tmp_path / "home_a" / ".hermes"
    home_a.mkdir(parents=True)
    ext_a = tmp_path / "ext_a"
    ext_a.mkdir()
    (home_a / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext_a}\n", encoding="utf-8"
    )

    home_b = tmp_path / "home_b" / ".hermes"
    home_b.mkdir(parents=True)
    ext_b = tmp_path / "ext_b"
    ext_b.mkdir()
    (home_b / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext_b}\n", encoding="utf-8"
    )

    _clear_skill_config_caches()

    expected_a = ext_a.resolve()
    expected_b = ext_b.resolve()
    original_resolve = Path.resolve
    resolve_counts = {ext_a: 0, ext_b: 0}

    def count_external_resolve(path, *args, **kwargs):
        if path in resolve_counts:
            resolve_counts[path] += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", count_external_resolve)

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert get_external_skills_dirs() == [expected_a]
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert get_external_skills_dirs() == [expected_b]

    # Switching back must reuse A, not resolve it again or leak B's entry.
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert get_external_skills_dirs() == [expected_a]
    assert resolve_counts == {ext_a: 1, ext_b: 1}

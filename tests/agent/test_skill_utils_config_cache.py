"""Tests for the in-process config.yaml cache used by skill_utils.

``get_disabled_skill_names`` and ``get_external_skills_dirs`` are called on
every ``build_skills_system_prompt`` invocation.  Each call previously read +
parsed ``~/.hermes/config.yaml`` from disk.  These tests pin the cache
behaviour: hits avoid re-reading the file, mtime/size changes invalidate.
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    return home


@pytest.fixture(autouse=True)
def _isolate_config_cache():
    from agent import skill_utils
    skill_utils._clear_config_cache()
    yield
    skill_utils._clear_config_cache()


def test_repeated_reads_avoid_reparsing(hermes_home):
    (hermes_home / "config.yaml").write_text(
        "skills:\n  disabled: [first, second]\n"
    )

    from agent import skill_utils

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        with patch.object(
            skill_utils, "yaml_load", wraps=skill_utils.yaml_load
        ) as wrapped:
            first = skill_utils.get_disabled_skill_names()
            second = skill_utils.get_disabled_skill_names()
            third = skill_utils.get_external_skills_dirs()

    assert first == {"first", "second"}
    assert second == {"first", "second"}
    assert third == []
    # First call parses; subsequent calls hit the cache for the same mtime+size.
    assert wrapped.call_count == 1


def test_cache_invalidates_on_config_change(hermes_home):
    config = hermes_home / "config.yaml"
    config.write_text("skills:\n  disabled: [alpha]\n")

    from agent import skill_utils

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        before = skill_utils.get_disabled_skill_names()
        # Bump mtime explicitly so the test is robust on filesystems whose
        # write resolution might collapse two writes onto the same nanosecond.
        config.write_text("skills:\n  disabled: [beta, gamma]\n")
        st = config.stat()
        os.utime(config, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        after = skill_utils.get_disabled_skill_names()

    assert before == {"alpha"}
    assert after == {"beta", "gamma"}


def test_missing_config_returns_empty_and_caches(hermes_home):
    # No config.yaml at all.
    from agent import skill_utils

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        with patch.object(
            skill_utils, "yaml_load", wraps=skill_utils.yaml_load
        ) as wrapped:
            assert skill_utils.get_disabled_skill_names() == set()
            assert skill_utils.get_external_skills_dirs() == []
            assert skill_utils.get_disabled_skill_names() == set()

    # No file → no parse, even on repeat.
    assert wrapped.call_count == 0


def test_malformed_config_caches_empty_dict(hermes_home):
    (hermes_home / "config.yaml").write_text("not: [valid: yaml: at: all\n")

    from agent import skill_utils

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        with patch.object(
            skill_utils, "yaml_load", wraps=skill_utils.yaml_load
        ) as wrapped:
            first = skill_utils.get_disabled_skill_names()
            second = skill_utils.get_disabled_skill_names()

    assert first == set()
    assert second == set()
    # Parse is attempted once; the empty-dict result is cached afterwards.
    assert wrapped.call_count == 1

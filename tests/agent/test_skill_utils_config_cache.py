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


def test_atomic_replace_invalidates_cache(hermes_home):
    """``atomic_yaml_write`` (temp file + os.replace) must invalidate the cache.

    The new file has a fresh inode even when content size happens to match,
    so the (mtime_ns, size, inode) signature distinguishes it from the old
    file.  This is the path used by gateway/CLI runtime config edits.
    """
    from utils import atomic_yaml_write
    from agent import skill_utils

    config = hermes_home / "config.yaml"
    atomic_yaml_write(config, {"skills": {"disabled": ["one"]}})

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        before = skill_utils.get_disabled_skill_names()
        # Replace with a payload that may match size on some YAML emitters.
        atomic_yaml_write(config, {"skills": {"disabled": ["two"]}})
        after = skill_utils.get_disabled_skill_names()

    assert before == {"one"}
    assert after == {"two"}


def test_returned_set_is_independent_of_cache(hermes_home):
    """Mutating the returned set must not corrupt subsequent reads."""
    (hermes_home / "config.yaml").write_text(
        "skills:\n  disabled: [first]\n"
    )

    from agent import skill_utils

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        first = skill_utils.get_disabled_skill_names()
        first.add("INJECTED")
        first.discard("first")
        second = skill_utils.get_disabled_skill_names()

    assert second == {"first"}
    assert "INJECTED" not in second


def test_concurrent_reads_are_safe(hermes_home):
    """The cache must serialize concurrent readers without corruption."""
    import threading

    (hermes_home / "config.yaml").write_text(
        "skills:\n  disabled: [a, b, c]\n"
    )

    from agent import skill_utils

    results = []
    errors = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            for _ in range(50):
                results.append(skill_utils.get_disabled_skill_names())
        except Exception as e:  # pragma: no cover - defensive
            errors.append(e)

    # Set HERMES_HOME once at the process level (not per-thread, since
    # ``os.environ`` is shared and ``patch.dict`` would race on exit).
    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert errors == []
    assert all(r == {"a", "b", "c"} for r in results)
    assert len(results) == 8 * 50


def test_cache_bounded_under_many_distinct_paths(tmp_path):
    """Pathological test runs with many tmpdirs must not grow the cache forever."""
    from agent import skill_utils

    for i in range(skill_utils._CONFIG_CACHE_MAX_ENTRIES + 10):
        home = tmp_path / f"home-{i}"
        home.mkdir()
        (home / "config.yaml").write_text(f"skills:\n  disabled: [s{i}]\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
            skill_utils.get_disabled_skill_names()

    with skill_utils._CONFIG_CACHE_LOCK:
        size = len(skill_utils._CONFIG_CACHE)
    assert size <= skill_utils._CONFIG_CACHE_MAX_ENTRIES


def test_external_dirs_independent_of_cached_dict(hermes_home, tmp_path):
    """``get_external_skills_dirs`` must return a fresh list even on cache hit."""
    ext = tmp_path / "ext"
    ext.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext}\n"
    )

    from agent import skill_utils

    with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
        first = skill_utils.get_external_skills_dirs()
        first.append("INJECTED")
        second = skill_utils.get_external_skills_dirs()

    assert "INJECTED" not in second
    assert second == [ext.resolve()]

"""Regression tests for the adapter-guard cache race (empty ``ERROR:`` flake).

CI runs the suite as ~96 concurrent per-file pytest subprocesses and does NOT
install ``filelock``, so ``tests/gateway/conftest.py::pytest_configure`` runs
its cache read/write completely lock-free there. The original code published
the verdict with ``Path.write_text`` (open-truncate-write): a sibling process
reading between the truncate and the write saw an empty file and raised
``pytest.UsageError("")`` — surfacing in the flaky-retry frames as a bare
``ERROR:`` with no message on healthy ``tests/gateway/relay/*`` files
(e.g. CI runs 32930243538, 32932008269 on 2026-08-26).

These tests pin the two properties the fix relies on:

1. ``_read_guard_cache`` treats empty/vanished cache files as a miss, never
   as a violation verdict.
2. ``_write_guard_cache_atomic`` publishes via ``os.replace`` from a staging
   name that the stale-fingerprint eviction globs cannot match.
"""

import os

from tests.gateway import conftest as gw_conftest


class TestReadGuardCache:
    def test_missing_file_is_a_miss(self, tmp_path):
        assert gw_conftest._read_guard_cache(tmp_path / "absent") is None

    def test_empty_file_is_a_miss_not_a_violation(self, tmp_path):
        """A torn/empty cache must be treated as absent. Before the fix this
        exact state produced ``UsageError("")`` — the empty ``ERROR:`` flake."""
        f = tmp_path / "gw-adapter-guard-cafe"
        f.write_text("", encoding="utf-8")
        assert gw_conftest._read_guard_cache(f) is None

    def test_whitespace_only_is_a_miss(self, tmp_path):
        f = tmp_path / "gw-adapter-guard-cafe"
        f.write_text("\n  \n", encoding="utf-8")
        assert gw_conftest._read_guard_cache(f) is None

    def test_clean_and_violation_content_round_trip(self, tmp_path):
        f = tmp_path / "gw-adapter-guard-cafe"
        f.write_text("clean", encoding="utf-8")
        assert gw_conftest._read_guard_cache(f) == "clean"
        f.write_text("violation: bad import", encoding="utf-8")
        assert gw_conftest._read_guard_cache(f) == "violation: bad import"


class TestWriteGuardCacheAtomic:
    def test_publishes_content_under_final_name(self, tmp_path):
        f = tmp_path / "gw-adapter-guard-cafe"
        gw_conftest._write_guard_cache_atomic(f, "clean")
        assert f.read_text(encoding="utf-8") == "clean"

    def test_no_staging_file_left_behind(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        f = cache_dir / "gw-adapter-guard-cafe"
        gw_conftest._write_guard_cache_atomic(f, "clean")
        leftovers = [p.name for p in cache_dir.iterdir() if p.name != f.name]
        assert leftovers == []

    def test_staging_name_never_matches_eviction_globs(self, tmp_path):
        """The eviction pass in pytest_configure unlinks
        ``gw-adapter-guard-*`` and ``.gw-adapter-guard-*.lock`` entries for
        other fingerprints. The staging file must be invisible to both globs,
        or a sibling's eviction could delete it between write and replace."""
        f = tmp_path / "gw-adapter-guard-cafe"
        staging = f.with_name(f".tmp-{f.name}.{os.getpid()}")
        staging.write_text("in flight", encoding="utf-8")
        try:
            evictable = set(tmp_path.glob("gw-adapter-guard-*")) | set(
                tmp_path.glob(".gw-adapter-guard-*.lock")
            )
            assert staging not in evictable
        finally:
            staging.unlink(missing_ok=True)

    def test_overwrite_is_all_or_nothing(self, tmp_path):
        """os.replace swaps the whole file: after any successful publish the
        content is complete, never a truncated prefix of the new verdict."""
        f = tmp_path / "gw-adapter-guard-cafe"
        gw_conftest._write_guard_cache_atomic(f, "clean")
        long_msg = "violation " * 5000
        gw_conftest._write_guard_cache_atomic(f, long_msg)
        assert f.read_text(encoding="utf-8") == long_msg

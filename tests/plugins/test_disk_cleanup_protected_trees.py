"""Regression tests: disk-cleanup must never auto-delete persistent trees.

Covers the 2026-08-13 incident class: HERMES_HOME persistent subtrees
(``browser-profile``, ``vendor``, ``benchmarks``, ``plans``) are not in the
protected-top-level lists, so (a) ``guess_category`` mis-classifies files
named ``test_*``/``tmp_*`` inside them as ``test`` (deleted at session end),
(b) ``quick``'s empty-dir sweep recurses into them and rmdirs runtime state,
and (c) a stale ``tracked.json`` entry pointing into such a tree (or outside
HERMES_HOME entirely) is deleted without re-validation.

Same harness style as ``test_disk_cleanup_plugin.py``: import the real plugin
library from the repo path, isolate ``HERMES_HOME`` under ``tmp_path``.
"""
import importlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _load_lib():
    repo_root = Path(__file__).resolve().parents[2]
    lib_path = repo_root / "plugins" / "disk-cleanup" / "disk_cleanup.py"
    spec = importlib.util.spec_from_file_location(
        "disk_cleanup_under_test", lib_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _old_ts(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class TestGuessCategoryProtectedTrees:
    """Files inside persistent subtrees must never be auto-categorised."""

    @pytest.mark.parametrize("subtree", [
        "browser-profile", "vendor", "benchmarks", "plans",
    ])
    def test_test_named_file_in_persistent_subtree_not_categorised(self, _isolate_env, subtree):
        dg = _load_lib()
        p = _isolate_env / subtree / "Default" / "blob_storage" / "test_cache"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        assert dg.guess_category(p) is None

    @pytest.mark.parametrize("subtree", [
        "browser-profile", "vendor", "benchmarks", "plans",
    ])
    def test_tmp_named_file_in_persistent_subtree_not_categorised(self, _isolate_env, subtree):
        dg = _load_lib()
        p = _isolate_env / subtree / "tmp_state.log"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        assert dg.guess_category(p) is None


class TestCacheOwnership:
    """Only explicitly plugin-owned cache roots are auto-managed."""

    def test_vision_temporary_root_is_categorised_as_temp(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "cache" / "vision" / "temp_vision_images" / "capture.png"
        p.parent.mkdir(parents=True)
        p.write_text("x")

        assert dg.guess_category(p) == "temp"
    def test_video_temporary_root_is_categorised_as_temp(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "cache" / "video" / "temp_video_files" / "render.mp4"
        p.parent.mkdir(parents=True)
        p.write_text("x")

        assert dg.guess_category(p) == "temp"

    def test_unmanaged_cache_path_is_not_categorised(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "cache" / "tool" / "rtk-bin" / "rtk"
        p.parent.mkdir(parents=True)
        p.write_text("x")

        assert dg.guess_category(p) is None


class TestQuickEmptyDirSweepProtectedTrees:
    """Empty-dir sweep must not rmdir inside persistent subtrees."""

    @pytest.mark.parametrize("subtree,leaf", [
        ("browser-profile", "Default/blob_storage"),
        ("vendor", "some_pkg/empty_dir"),
        ("benchmarks", "suite/empty_dir"),
        ("plans", "draft/empty_dir"),
    ])
    def test_empty_dir_under_persistent_subtree_survives_quick(self, _isolate_env, subtree, leaf):
        dg = _load_lib()
        empty = _isolate_env / subtree / leaf
        empty.mkdir(parents=True)
        # A normal file must exist for quick() to consider the tree at all.
        dg.track(str(_isolate_env / "test_legit.py"), "test", silent=True)
        _ = dg.quick()
        assert empty.is_dir(), f"{subtree}/{leaf} must survive quick()"


class TestQuickRevalidatesTrackedEntries:
    """quick() must re-validate every candidate before deleting (fail-closed)."""

    def _seed_tracked(self, hermes_home, entry):
        tf = hermes_home / "disk-cleanup" / "tracked.json"
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(json.dumps([entry]))

    def test_stale_entry_in_protected_tree_not_deleted(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "browser-profile" / "Default" / "Service Worker" / "test_sw"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        self._seed_tracked(_isolate_env, {
            "path": str(p), "category": "test",
            "timestamp": _old_ts(1), "size": 1,
        })
        summary = dg.quick()
        assert summary["deleted"] == 0
        assert p.exists(), "persistent-tree file must survive even with stale entry"

    def test_entry_outside_hermes_home_not_deleted(self, _isolate_env, tmp_path):
        dg = _load_lib()
        # A sibling dir outside HERMES_HOME (same tmp_path, but not under .hermes)
        outside = tmp_path / "outside" 
        outside.mkdir()
        p = outside / "test_escape.py"
        p.write_text("x")
        self._seed_tracked(_isolate_env, {
            "path": str(p), "category": "test",
            "timestamp": _old_ts(1), "size": 1,
        })
        summary = dg.quick()
        assert summary["deleted"] == 0
        assert p.exists(), "path outside HERMES_HOME must never be deleted"

    def test_stale_temp_entry_in_unmanaged_cache_is_not_deleted(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "cache" / "tool" / "rtk-bin" / "rtk"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        self._seed_tracked(_isolate_env, {
            "path": str(p), "category": "temp",
            "timestamp": _old_ts(8), "size": 1,
        })

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists(), "unmanaged cache file must survive stale temp tracking"

    def test_dry_run_omits_stale_temp_entry_in_unmanaged_cache(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "cache" / "tool" / "rtk-bin" / "rtk"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        self._seed_tracked(_isolate_env, {
            "path": str(p), "category": "temp",
            "timestamp": _old_ts(8), "size": 1,
        })

        auto, prompt = dg.dry_run()

        assert auto == []
        assert prompt == []

    def test_legitimate_test_file_still_deleted(self, _isolate_env):
        """The fix must not break the feature: real test files are still cleaned."""
        dg = _load_lib()
        p = _isolate_env / "test_real.py"
        p.write_text("x")
        assert dg.track(str(p), "test", silent=True) is True
        summary = dg.quick()
        assert summary["deleted"] == 1
        assert not p.exists()

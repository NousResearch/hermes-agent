from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_pr_runtime_leaks.py"
spec = importlib.util.spec_from_file_location("check_pr_runtime_leaks", MODULE_PATH)
assert spec is not None
check_pr_runtime_leaks = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(check_pr_runtime_leaks)


def test_forbidden_runtime_and_overlay_paths_are_detected():
    paths = [
        ".hermes/state.db",
        "./.hermes/config.yaml",
        "hermes-local-overlay/patches/local.diff",
        "ops/plugins/config/private.yaml",
        "config-overlay.yaml",
        "nested/config-overlay.yaml",
        "local-only-files.txt",
        "ops/hermes-local-overlay/local-only-files.txt",
    ]

    assert check_pr_runtime_leaks.find_forbidden_paths(paths) == [
        ".hermes/config.yaml",
        ".hermes/state.db",
        "config-overlay.yaml",
        "hermes-local-overlay/patches/local.diff",
        "local-only-files.txt",
        "nested/config-overlay.yaml",
        "ops/hermes-local-overlay/local-only-files.txt",
        "ops/plugins/config/private.yaml",
    ]


def test_similar_non_runtime_paths_are_allowed():
    paths = [
        "docs/config-overlay.yaml.md",
        "docs/local-only-files.txt.md",
        "ops/plugins/configuration/README.md",
        "src/hermes-local-overlay-helper.py",
        "some/.hermes-notes.md",
    ]

    assert check_pr_runtime_leaks.find_forbidden_paths(paths) == []


def test_windows_style_paths_are_normalized():
    assert check_pr_runtime_leaks.find_forbidden_paths([r".hermes\\state.db"]) == [".hermes/state.db"]

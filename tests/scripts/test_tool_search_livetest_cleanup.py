from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "tool_search_livetest.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("tool_search_livetest_cleanup_test", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_uses_allocated_home_not_mutable_environment(tmp_path, monkeypatch):
    harness = _load_harness()
    allocated_root = tmp_path / "allocated"
    allocated_home = allocated_root / ".hermes"
    allocated_home.mkdir(parents=True)
    (allocated_home / "config.yaml").write_text("test: true\n", encoding="utf-8")

    unrelated_root = tmp_path / "unrelated"
    unrelated_home = unrelated_root / ".hermes"
    unrelated_home.mkdir(parents=True)
    sentinel = unrelated_root / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(unrelated_home))

    harness.cleanup_isolated_home(allocated_home)

    assert not allocated_root.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"

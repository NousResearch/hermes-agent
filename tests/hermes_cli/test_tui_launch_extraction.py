"""Regression tests for the wave-1 extraction of TUI-launch helpers.

Shard s1, clusters c15 + c17 (canonical godfile plan): the TUI-launch
functions moved out of ``hermes_cli.main`` into ``hermes_cli.tui_launch``
must (1) still resolve from ``hermes_cli.main`` under the same names — the
import-back contract that existing tests (``test_tui_npm_install``,
``test_tui_heap_sizing``, ``test_tui_bundled``, docker prebuilt-bundle) and
``hermes_cli/web_server.py`` rely on — and (2) keep their pure behaviour
byte-for-byte. The heap-sizing tests additionally pin the extraction seam:
``_resolve_tui_heap_mb`` must see monkeypatches applied to
``hermes_cli.main._read_cgroup_memory_limit``.
"""

import builtins
import io
from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.main as main
import hermes_cli.tui_launch as tui_launch

MOVED = [
    "_apply_tui_python_env",
    "_iter_tui_build_inputs",
    "_launch_tui",
    "_make_tui_argv",
    "_normalize_tui_toolsets",
    "_read_cgroup_memory_limit",
    "_resolve_tui_heap_mb",
    "_safe_tui_cwd",
    "_termux_workspace_install_context",
    "_tui_need_npm_install",
    "_tui_need_rebuild",
    "_workspace_root",
]

# Names hermes_cli.main must re-export (consumers: web_server.py, the
# existing TUI tests, and the _make_tui_argv / _resolve_tui_heap_mb seams).
REEXPORTED = [
    "_apply_tui_python_env",
    "_launch_tui",
    "_make_tui_argv",
    "_read_cgroup_memory_limit",
    "_resolve_tui_heap_mb",
    "_termux_workspace_install_context",
    "_tui_need_npm_install",
    "_tui_need_rebuild",
    "_workspace_root",
]


def test_moved_functions_reexported_from_main() -> None:
    """hermes_cli.main.<name> resolves to the moved function object."""
    for name in MOVED:
        assert hasattr(tui_launch, name), f"tui_launch missing {name}"
    for name in REEXPORTED:
        assert getattr(main, name) is getattr(tui_launch, name), name


def test_moved_constants_travel_with_the_module() -> None:
    assert tui_launch._NPM_LOCK_RUNTIME_KEYS == frozenset({"ideallyInert", "peer"})
    assert tui_launch._TUI_BUILD_INPUT_SUFFIXES == frozenset(
        {".cjs", ".js", ".jsx", ".json", ".mjs", ".ts", ".tsx"}
    )
    assert "package-lock.json" in tui_launch._TUI_BUILD_INPUT_FILES


def test_workspace_root_prefers_parent_with_lockfile(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "ui-tui"
    sub.mkdir(parents=True)
    (sub / "package.json").write_text("{}")
    (root / "package-lock.json").write_text("{}")
    # sub has package.json but no lockfile, parent has the lockfile => parent.
    assert tui_launch._workspace_root(sub) == root
    # standalone project: dir itself is the root.
    assert tui_launch._workspace_root(root) == root


def test_termux_workspace_install_context_scopes_to_child(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "ui-tui"
    sub.mkdir(parents=True)
    (sub / "package.json").write_text("{}")
    (root / "package-lock.json").write_text("{}")
    cwd, args = tui_launch._termux_workspace_install_context(sub)
    assert cwd == root
    assert "--workspace" in args and "ui-tui" in args
    assert args[-1] == "--include-workspace-root=false"


def test_tui_need_npm_install_prebuilt_bundle_skips(tmp_path: Path) -> None:
    """Prebuilt self-contained bundle (dist/entry.js, no lockfile) => no install."""
    root = tmp_path / "ui-tui"
    (root / "dist").mkdir(parents=True)
    (root / "dist" / "entry.js").write_text("// tui")
    (root / "package.json").write_text("{}")
    assert tui_launch._tui_need_npm_install(root) is False


def test_tui_need_npm_install_missing_ink_requires_install(tmp_path: Path) -> None:
    root = tmp_path / "ui-tui"
    root.mkdir()
    (root / "package.json").write_text("{}")
    (root / "package-lock.json").write_text("{}")
    assert tui_launch._tui_need_npm_install(root) is True


def test_tui_need_rebuild_forced_by_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_TUI_FORCE_BUILD", "1")
    assert tui_launch._tui_need_rebuild(tmp_path) is True


def test_read_cgroup_memory_limit_v2_max_is_unlimited() -> None:
    real_open = builtins.open

    def opener(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/memory.max":
            return io.StringIO("max")
        return real_open(path, *args, **kwargs)

    with mock.patch.object(builtins, "open", opener):
        assert tui_launch._read_cgroup_memory_limit() is None


def test_resolve_tui_heap_mb_sees_main_monkeypatch(monkeypatch) -> None:
    """Extraction seam: _resolve_tui_heap_mb must observe patches applied to
    hermes_cli.main._read_cgroup_memory_limit (the pre-extraction seam that
    test_tui_heap_sizing.py relies on)."""
    monkeypatch.setattr(main, "_read_cgroup_memory_limit", lambda: None)
    assert tui_launch._resolve_tui_heap_mb() == 8192

    monkeypatch.setattr(main, "_read_cgroup_memory_limit", lambda: 4 * 1024**3)
    assert tui_launch._resolve_tui_heap_mb() == 3072


def test_make_tui_argv_sees_main_monkeypatch(tmp_path: Path, monkeypatch) -> None:
    """Extraction seam: _make_tui_argv must observe patches applied to
    hermes_cli.main._ensure_tui_node / _find_bundled_tui (the seam
    test_tui_npm_install.py relies on) and still prefer the bundled TUI
    before the (missing) workspace."""
    import hermes_constants

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    # _node_bin resolves via hermes_constants.find_node_executable (lazily
    # imported at call time); pin it so the test is machine-independent.
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: "/usr/bin/node")
    monkeypatch.setattr(main, "_ensure_tui_node", lambda: None)
    bundled = tmp_path / "bundled" / "entry.js"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("// bundled TUI")
    monkeypatch.setattr(main, "_find_bundled_tui", lambda: bundled)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("bundled path must not spawn subprocesses")

    monkeypatch.setattr(main.subprocess, "run", fail_run)

    tui_dir = tmp_path / "ui-tui"
    argv, cwd = tui_launch._make_tui_argv(tui_dir, tui_dev=False)
    assert argv == ["/usr/bin/node", "--expose-gc", str(bundled)]
    assert cwd == bundled.parent

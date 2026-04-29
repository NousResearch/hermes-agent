"""`hermes update` skips `npm ci` when node_modules is already in sync (#17268).

`_npm_install_in_sync(root)` decides whether the committed `package-lock.json`
already matches `node_modules/.package-lock.json` (npm's hidden lockfile);
when it does, `_update_node_dependencies()` must not call `npm ci` at all,
because `npm ci` deletes and re-installs `node_modules` from scratch even
on a no-op update — the slow path on metered or restricted networks.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as m

    return m


def _write_lock(root: Path, packages: dict) -> None:
    (root / "package-lock.json").write_text(json.dumps({"packages": packages}))


def _write_marker(root: Path, packages: dict) -> None:
    nm = root / "node_modules"
    nm.mkdir(parents=True, exist_ok=True)
    (nm / ".package-lock.json").write_text(json.dumps({"packages": packages}))


# ---------------------------------------------------------------------------
# _npm_install_in_sync — pure lockfile comparison
# ---------------------------------------------------------------------------


def test_in_sync_when_lock_and_marker_match(tmp_path: Path, main_mod) -> None:
    pkg = {"node_modules/foo": {"version": "1.0.0"}}
    _write_lock(tmp_path, pkg)
    _write_marker(tmp_path, pkg)
    assert main_mod._npm_install_in_sync(tmp_path) is True


def test_out_of_sync_when_marker_missing(tmp_path: Path, main_mod) -> None:
    _write_lock(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})
    assert main_mod._npm_install_in_sync(tmp_path) is False


def test_out_of_sync_when_lockfile_missing(tmp_path: Path, main_mod) -> None:
    _write_marker(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})
    assert main_mod._npm_install_in_sync(tmp_path) is False


def test_out_of_sync_when_required_package_missing_from_marker(tmp_path: Path, main_mod) -> None:
    _write_lock(
        tmp_path,
        {
            "node_modules/foo": {"version": "1.0.0"},
            "node_modules/bar": {"version": "1.0.0"},
        },
    )
    _write_marker(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})
    assert main_mod._npm_install_in_sync(tmp_path) is False


def test_in_sync_when_only_optional_or_peer_packages_missing(tmp_path: Path, main_mod) -> None:
    _write_lock(
        tmp_path,
        {
            "node_modules/foo": {"version": "1.0.0"},
            "node_modules/optional-only": {"version": "1.0.0", "optional": True},
            "node_modules/peer-only": {"version": "1.0.0", "peer": True},
        },
    )
    _write_marker(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})
    assert main_mod._npm_install_in_sync(tmp_path) is True


def test_in_sync_ignores_npm_runtime_annotations(tmp_path: Path, main_mod) -> None:
    _write_lock(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})
    _write_marker(
        tmp_path,
        {"node_modules/foo": {"version": "1.0.0", "ideallyInert": True}},
    )
    assert main_mod._npm_install_in_sync(tmp_path) is True


def test_out_of_sync_when_versions_differ(tmp_path: Path, main_mod) -> None:
    _write_lock(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})
    _write_marker(tmp_path, {"node_modules/foo": {"version": "1.1.0"}})
    assert main_mod._npm_install_in_sync(tmp_path) is False


def test_falls_back_to_mtime_when_unparseable(tmp_path: Path, main_mod) -> None:
    (tmp_path / "package-lock.json").write_text("not json")
    nm = tmp_path / "node_modules"
    nm.mkdir(parents=True, exist_ok=True)
    (nm / ".package-lock.json").write_text("also not json")
    # Marker newer → in sync (no install needed).
    os.utime(tmp_path / "package-lock.json", (100, 100))
    os.utime(nm / ".package-lock.json", (200, 200))
    assert main_mod._npm_install_in_sync(tmp_path) is True
    # Lock newer → out of sync.
    os.utime(tmp_path / "package-lock.json", (300, 300))
    os.utime(nm / ".package-lock.json", (200, 200))
    assert main_mod._npm_install_in_sync(tmp_path) is False


# ---------------------------------------------------------------------------
# _update_node_dependencies — does not call npm when in sync (#17268)
# ---------------------------------------------------------------------------


def _seed_repo_root_and_tui(tmp_path: Path, *, in_sync: bool) -> Path:
    """Populate tmp_path so it looks like both repo root and ui-tui/.

    `_update_node_dependencies()` consults `PROJECT_ROOT` (repo root) and
    `PROJECT_ROOT / "ui-tui"`. We only seed the repo-root path here; the
    ui-tui path is left without a `package.json` so the loop short-circuits
    on the second iteration. That keeps the test focused on the skip
    decision for one location.
    """
    (tmp_path / "package.json").write_text("{}")
    pkg = {"node_modules/foo": {"version": "1.0.0"}}
    _write_lock(tmp_path, pkg)
    if in_sync:
        _write_marker(tmp_path, pkg)
    return tmp_path


def test_update_skips_npm_call_when_in_sync(tmp_path: Path, main_mod, capsys) -> None:
    _seed_repo_root_and_tui(tmp_path, in_sync=True)

    npm_path = "/fake/npm"
    runner = MagicMock()
    with patch.object(main_mod.shutil, "which", return_value=npm_path), \
        patch.object(main_mod, "PROJECT_ROOT", tmp_path), \
        patch.object(main_mod, "_run_npm_install_deterministic", runner):
        main_mod._update_node_dependencies()

    out = capsys.readouterr().out
    assert "Updating Node.js dependencies" in out
    assert "already in sync with lockfile" in out
    runner.assert_not_called()


def test_update_calls_npm_when_marker_missing(tmp_path: Path, main_mod, capsys) -> None:
    _seed_repo_root_and_tui(tmp_path, in_sync=False)

    npm_path = "/fake/npm"
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    with patch.object(main_mod.shutil, "which", return_value=npm_path), \
        patch.object(main_mod, "PROJECT_ROOT", tmp_path), \
        patch.object(main_mod, "_run_npm_install_deterministic", runner):
        main_mod._update_node_dependencies()

    out = capsys.readouterr().out
    assert "already in sync" not in out
    runner.assert_called_once()
    args, kwargs = runner.call_args
    assert args[0] == npm_path
    assert args[1] == tmp_path


def test_update_calls_npm_when_lockfile_diverges(tmp_path: Path, main_mod) -> None:
    _seed_repo_root_and_tui(tmp_path, in_sync=True)
    # Mutate the lockfile to add a required package the marker doesn't have.
    _write_lock(
        tmp_path,
        {
            "node_modules/foo": {"version": "1.0.0"},
            "node_modules/bar": {"version": "1.0.0"},
        },
    )

    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    with patch.object(main_mod.shutil, "which", return_value="/fake/npm"), \
        patch.object(main_mod, "PROJECT_ROOT", tmp_path), \
        patch.object(main_mod, "_run_npm_install_deterministic", runner):
        main_mod._update_node_dependencies()

    runner.assert_called_once()

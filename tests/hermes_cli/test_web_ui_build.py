"""Tests for _web_ui_build_needed — staleness check for the web UI dist.

Critical invariant: the Vite build outputs to hermes_cli/web_dist/
(vite.config.ts: outDir: "../hermes_cli/web_dist"), NOT web/dist/.
The sentinel must be checked in the correct output directory or the
freshness check is a no-op and the OOM rebuild always runs.
"""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main import _web_ui_build_needed, _build_web_ui


def _touch(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if offset:
        t = time.time() + offset
        os.utime(path, (t, t))


def _make_web_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Return (web_dir, dist_dir) matching real repo layout."""
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "package.json").touch()
    dist_dir = tmp_path / "hermes_cli" / "web_dist"
    return web_dir, dist_dir


class TestWebUIBuildNeeded:

    def test_returns_true_when_dist_missing(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_false_when_vite_manifest_fresh(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(web_dir / "src" / "App.tsx", offset=-10)
        _touch(dist_dir / ".vite" / "manifest.json")
        assert _web_ui_build_needed(web_dir) is False

    def test_returns_true_when_source_newer_than_manifest(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "src" / "App.tsx")
        assert _web_ui_build_needed(web_dir) is True

    def test_falls_back_to_index_html_when_manifest_missing(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(web_dir / "src" / "main.ts", offset=-10)
        _touch(dist_dir / "index.html")
        assert _web_ui_build_needed(web_dir) is False

    def test_web_dist_dir_not_web_dist_subdir(self, tmp_path):
        """Regression: sentinel must be in hermes_cli/web_dist/, NOT web/dist/."""
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(web_dir / "src" / "App.tsx", offset=-10)
        # Place manifest in wrong location (web/dist/) — should NOT count as fresh
        wrong_dist = web_dir / "dist" / ".vite" / "manifest.json"
        _touch(wrong_dist)
        # Correct location is empty → still needs build
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_true_when_package_lock_newer_than_dist(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "package-lock.json")
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_true_when_vite_config_newer_than_dist(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "vite.config.ts")
        assert _web_ui_build_needed(web_dir) is True

    def test_ignores_node_modules(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        # package.json older than manifest; only node_modules file is newer
        _touch(web_dir / "package.json", offset=-20)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "node_modules" / "react" / "index.js")
        assert _web_ui_build_needed(web_dir) is False

    def test_ignores_dist_subdir_under_web(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        # package.json older than manifest; only web/dist file is newer
        _touch(web_dir / "package.json", offset=-20)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "dist" / "assets" / "index.js")
        assert _web_ui_build_needed(web_dir) is False


class TestBuildWebUISkipsWhenFresh:

    def test_skips_npm_when_dist_is_fresh(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json")

        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run") as mock_run:
            result = _build_web_ui(web_dir)

        assert result is True
        mock_run.assert_not_called()

    def test_runs_npm_when_dist_missing(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)

        mock_cp = __import__("subprocess").CompletedProcess([], 0, stdout=b"", stderr=b"")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=mock_cp) as mock_run:
            result = _build_web_ui(web_dir)

        assert result is True
        assert mock_run.call_count == 2  # npm install + npm run build

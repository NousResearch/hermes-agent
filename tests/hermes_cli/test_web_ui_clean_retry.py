"""Regression tests for #34312 / #34335 — `hermes update` leaves
web/node_modules in a broken state and the dashboard crash-loops.

The fix: on npm install failure with an existing node_modules tree,
wipe node_modules and retry. The clean install handles dep-version
layout shifts (lucide-react icon paths, missing tsc in .bin) that
defeat in-place reinstalls.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _has_npm():
    """Check if npm-related code paths can even be tested in this env."""
    try:
        import shutil
        return shutil.which("npm") is not None
    except Exception:
        return False


def _make_web_dir(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    web.mkdir()
    (web / "package.json").write_text('{"name": "web", "version": "0.0.0"}')
    return web


def _mock_completed(rc: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["npm", "install"], returncode=rc, stdout=stdout, stderr=stderr
    )


def test_clean_retry_runs_when_install_fails_with_node_modules_present(tmp_path):
    """The fix path: first install fails, node_modules exists, wipe + retry."""
    from hermes_cli.main import _build_web_ui

    web = _make_web_dir(tmp_path)
    node_modules = web / "node_modules"
    node_modules.mkdir()
    (node_modules / "stale.txt").write_text("from previous version")

    install_calls = []

    def fake_install(npm, path, **kwargs):
        install_calls.append((str(path), node_modules.exists()))
        # Simulate: first call fails (in-place install over stale tree),
        # second call succeeds (after wipe).
        if len(install_calls) == 1:
            return _mock_completed(1, stderr="lucide-react resolve error")
        return _mock_completed(0)

    with patch("hermes_cli.main._run_npm_install_deterministic", side_effect=fake_install), \
         patch("hermes_cli.main._run_with_idle_timeout", return_value=_mock_completed(0)), \
         patch("hermes_cli.main._web_ui_build_needed", return_value=True), \
         patch("shutil.which", return_value="/usr/bin/npm"):
        ok = _build_web_ui(web, fatal=False)

    # Two install attempts.
    assert len(install_calls) == 2
    # node_modules was wiped between attempts.
    assert install_calls[0][1] is True   # existed at first call
    assert install_calls[1][1] is False  # gone by second call
    # Build succeeded overall.
    assert ok is True


def test_no_clean_retry_when_node_modules_absent(tmp_path):
    """If node_modules doesn't exist (fresh install), don't try to wipe
    something that isn't there. Fail fast."""
    from hermes_cli.main import _build_web_ui

    web = _make_web_dir(tmp_path)
    # No node_modules dir.

    install_calls = []

    def fake_install(npm, path, **kwargs):
        install_calls.append(str(path))
        return _mock_completed(1, stderr="real failure")

    with patch("hermes_cli.main._run_npm_install_deterministic", side_effect=fake_install), \
         patch("hermes_cli.main._web_ui_build_needed", return_value=True), \
         patch("shutil.which", return_value="/usr/bin/npm"):
        ok = _build_web_ui(web, fatal=False)

    # Only ONE install attempt (no retry because nothing to wipe).
    assert len(install_calls) == 1
    assert ok is False


def test_clean_retry_skipped_when_first_install_succeeds(tmp_path):
    """The happy path: first install succeeds, no retry, no wipe."""
    from hermes_cli.main import _build_web_ui

    web = _make_web_dir(tmp_path)
    node_modules = web / "node_modules"
    node_modules.mkdir()
    (node_modules / "ok.txt").write_text("healthy")

    install_calls = []

    def fake_install(npm, path, **kwargs):
        install_calls.append(str(path))
        return _mock_completed(0)

    with patch("hermes_cli.main._run_npm_install_deterministic", side_effect=fake_install), \
         patch("hermes_cli.main._run_with_idle_timeout", return_value=_mock_completed(0)), \
         patch("hermes_cli.main._web_ui_build_needed", return_value=True), \
         patch("shutil.which", return_value="/usr/bin/npm"):
        ok = _build_web_ui(web, fatal=False)

    # Only ONE install attempt because the first one passed.
    assert len(install_calls) == 1
    # node_modules NOT wiped (still has our marker).
    assert (node_modules / "ok.txt").exists()
    assert ok is True


def test_clean_retry_failure_reports_real_error(tmp_path):
    """If both attempts fail, the user gets the real error from the second
    (clean) attempt — not a misleading message about the stale state."""
    from hermes_cli.main import _build_web_ui

    web = _make_web_dir(tmp_path)
    node_modules = web / "node_modules"
    node_modules.mkdir()

    def fake_install(npm, path, **kwargs):
        return _mock_completed(2, stderr="npm: command unavailable")

    with patch("hermes_cli.main._run_npm_install_deterministic", side_effect=fake_install), \
         patch("hermes_cli.main._web_ui_build_needed", return_value=True), \
         patch("shutil.which", return_value="/usr/bin/npm"):
        ok = _build_web_ui(web, fatal=False)

    assert ok is False

"""Regression tests for scripts/check-windows-footguns.py path handling.

The checker documents an explicit-path mode
(``python scripts/check-windows-footguns.py path/to/file.py``). ``main()``
resolves those arguments to absolute paths and feeds them through
``iter_files`` -> ``should_scan_file``. Both ``should_scan_file`` and the
``main`` output loop called ``path.relative_to(REPO_ROOT).as_posix()``
unconditionally — but ``Path.relative_to`` raises ``ValueError`` for any path
that does not live under ``REPO_ROOT`` (a sibling file, a symlinked/relocated
worktree, an agent worktree under a different prefix). That crashed the whole
checker with a traceback, silently defeating the pre-PR footgun gate it exists
to enforce. ``EXCLUDED_FILES`` only ever holds repo-relative paths, so for a
file outside the repo the correct answer is simply "not excluded" — never a
crash.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-windows-footguns.py"
_MODULE_NAME = "check_windows_footguns_test_module"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module-level @dataclass can resolve
    # ``cls.__module__`` via ``sys.modules`` during decoration.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


footguns = _load_script_module()


def test_should_scan_file_outside_repo_does_not_crash():
    """A path outside REPO_ROOT must not raise and must be in scope.

    Before the fix this raised ``ValueError`` from
    ``path.relative_to(REPO_ROOT)``.
    """
    outside = Path("/tmp/outside-repo-check/foo.py")
    # Does not raise, and a .py file outside the repo is still scannable
    # (it can never match EXCLUDED_FILES, which is repo-relative only).
    assert footguns.should_scan_file(outside) is True


def test_main_scans_explicit_path_outside_repo(tmp_path, capsys):
    """End-to-end: explicit-path mode on a file outside the repo reports the
    footgun and exits 1 instead of crashing with an uncaught ValueError.
    """
    # tmp_path is a pytest temp dir outside the repo checkout. Write a file
    # containing a real footgun the checker matches: ``os.killpg`` (no
    # post_filter, so the bare reference is always flagged).
    outside_file = tmp_path / "uses_killpg.py"
    outside_file.write_text(
        "import os\n\n\ndef stop(pgid):\n    os.killpg(pgid, 9)\n",
        encoding="utf-8",
    )

    rc = footguns.main([str(outside_file)])

    out = capsys.readouterr().out
    assert rc == 1, "explicit-path scan of a footgun file should exit 1"
    # The finding line falls back to the absolute path when the file is
    # outside the repo (no relative_to crash).
    assert str(outside_file) in out
    assert "os.killpg" in out

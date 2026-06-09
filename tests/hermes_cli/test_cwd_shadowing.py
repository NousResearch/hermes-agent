"""Regression tests for CWD package shadowing.

When a user runs ``hermes`` from a directory containing a file named
``tools.py`` (or ``utils.py``, etc.), Python's PathFinder resolves
``import tools`` to that file instead of the Hermes ``tools/`` package.
The CWD entries ('', '.', and the absolute CWD path) must be stripped
from sys.path at startup to prevent this.

The bug is compounded by virtualenv .pth hooks which resolve '' to the
absolute CWD path before user code runs, meaning a guard that only strips
'' and '.' misses the absolute path entry.

See: https://github.com/NousResearch/hermes-agent/issues/XXXX
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def shadow_dir(tmp_path: Path):
    """Create a temp dir with empty ``tools.py`` and ``utils.py``."""
    (tmp_path / "tools.py").write_text("# shadow file\n")
    (tmp_path / "utils.py").write_text("# shadow file\n")
    return tmp_path


def test_cwd_tools_py_does_not_shadow_hermes_package(
    shadow_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """An empty ``tools.py`` in CWD must not shadow the Hermes ``tools/`` package.

    Before the fix, running ``hermes`` from a directory with a ``tools.py``
    file caused every ``from tools.xxx import …`` to fail with
    ``'tools' is not a package``.
    """
    monkeypatch.chdir(shadow_dir)
    # Simulate what main.py does: insert PROJECT_ROOT then strip CWD entries.
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # The fix: strip '', '.', and the absolute CWD.
    # Virtualenv .pth hooks may resolve '' to the absolute CWD before our
    # code runs, so we must strip all three forms.
    cwd = os.getcwd()
    sys.path = [p for p in sys.path if p not in {"", ".", cwd}]

    # Clear any cached 'tools' module from previous tests.
    sys.modules.pop("tools", None)
    sys.modules.pop("tools.registry", None)

    import tools  # noqa: F811

    assert hasattr(tools, "__path__"), (
        f"'tools' resolved to a file ({tools.__file__}), not a package. "
        "CWD shadowing is not prevented."
    )
    assert "hermes-agent" in str(tools.__file__), (
        f"'tools' resolved to {tools.__file__}, expected Hermes package."
    )


def test_cwd_utils_py_does_not_shadow_hermes_utils(
    shadow_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """An empty ``utils.py`` in CWD must not shadow the Hermes ``utils`` module."""
    monkeypatch.chdir(shadow_dir)
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    cwd = os.getcwd()
    sys.path = [p for p in sys.path if p not in {"", ".", cwd}]

    sys.modules.pop("utils", None)

    import utils  # noqa: F811

    assert "hermes-agent" in str(utils.__file__), (
        f"'utils' resolved to {utils.__file__}, expected Hermes module."
    )


def test_sys_path_strips_cwd_entries():
    """Verify the guard logic removes '', '.', and absolute CWD."""
    cwd = os.getcwd()
    test_path = ["/some/path", "", "/another", ".", cwd, "/last"]
    result = [p for p in test_path if p not in {"", ".", cwd}]
    assert result == ["/some/path", "/another", "/last"]

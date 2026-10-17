"""``_apply_tui_python_env``: how the TUI child's ``HERMES_PYTHON`` is chosen."""

import sys

import pytest

from hermes_cli import main_tui_launch


@pytest.mark.parametrize("advertised", [None, "", "   "])
def test_unset_python_falls_back_without_path_scan(tmp_path, monkeypatch, advertised):
    """An empty HERMES_PYTHON falls back to this interpreter without a PATH lookup.

    ``shutil.which("")`` can never name an interpreter, yet it stats every PATH
    directory itself; on a PM-managed machine those include tool directories
    under the real Hermes home, which the suite's home I/O guard refuses.
    """
    lookups = []
    monkeypatch.setattr(main_tui_launch.shutil, "which", lambda *a, **kw: lookups.append(a))
    env = {"HERMES_CWD": str(tmp_path), "PATH": str(tmp_path)}
    if advertised is not None:
        env["HERMES_PYTHON"] = advertised

    main_tui_launch._apply_tui_python_env(env)

    assert env["HERMES_PYTHON"] == sys.executable
    assert lookups == []

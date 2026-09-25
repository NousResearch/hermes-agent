"""Windows gateway imports must follow PM's committed Python ABI."""

import os
import sys
from pathlib import Path

from gateway import run


def test_committed_generation_wins_over_legacy_virtualenv(tmp_path, monkeypatch):
    from pm import environments

    committed = tmp_path / "committed"
    legacy = tmp_path / "legacy"
    for root in (committed, legacy):
        (root / "Lib" / "site-packages").mkdir(parents=True)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(legacy))
    monkeypatch.setattr(environments, "committed_venv", lambda _root: committed)

    run._ensure_windows_gateway_venv_imports()

    assert sys.path[1] == str(committed / "Lib" / "site-packages")
    assert str(legacy / "Lib" / "site-packages") not in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(committed)

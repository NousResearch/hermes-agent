"""`hermes checkpoints clear` acts on the active profile's checkpoint folder.

One process can serve several profiles. A checkpoint folder that belongs to the profile
`tools.checkpoint_manager` was loaded under must not decide what `clear` does for a profile
that has none.
"""

from __future__ import annotations

import argparse
import importlib


def test_clear_reads_the_active_profile_not_the_one_the_module_loaded_under(tmp_path, monkeypatch, capsys):
    loaded_under, active = tmp_path / "loaded-under", tmp_path / "active"
    kept = loaded_under / "checkpoints" / "store"
    kept.mkdir(parents=True)
    active.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(loaded_under))
    from tools import checkpoint_manager
    importlib.reload(checkpoint_manager)
    monkeypatch.setenv("HERMES_HOME", str(active))

    from hermes_cli.checkpoints import cmd_clear

    assert cmd_clear(argparse.Namespace(force=True)) == 0
    assert "Nothing to clear" in capsys.readouterr().out
    assert kept.is_dir()

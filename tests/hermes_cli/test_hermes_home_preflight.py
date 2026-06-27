from pathlib import Path

import pytest


def test_ensure_hermes_home_surfaces_writable_probe_failures(tmp_path):
    from hermes_cli import config as config_mod

    home = tmp_path / ".hermes"

    with (
        pytest.MonkeyPatch.context() as mp,
        pytest.raises(RuntimeError, match="Cannot initialize HERMES_HOME"),
    ):
        mp.setattr(config_mod, "get_hermes_home", lambda: home)
        mp.setattr(
            config_mod,
            "_probe_hermes_home_writable",
            lambda _home: (_ for _ in ()).throw(PermissionError("Write denied")),
        )
        config_mod.ensure_hermes_home()


def test_ensure_hermes_home_mentions_termux_android_conflicts(tmp_path):
    from hermes_cli import config as config_mod

    home = tmp_path / ".hermes"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config_mod, "get_hermes_home", lambda: home)
        mp.setattr(config_mod, "is_termux", lambda: True)
        mp.setattr(
            config_mod,
            "_probe_hermes_home_writable",
            lambda _home: (_ for _ in ()).throw(PermissionError("Write denied")),
        )
        with pytest.raises(RuntimeError) as excinfo:
            config_mod.ensure_hermes_home()

    msg = str(excinfo.value)
    assert "Termux" in msg
    assert "Android app" in msg
    assert "HERMES_HOME" in msg

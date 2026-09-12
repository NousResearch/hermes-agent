"""Retired build entrypoints stop old in-memory updaters without side effects."""
import sys

import pytest

from hermes_cli.main_web_build import _nixos_build_env, _run_npm_install_deterministic, _run_with_idle_timeout


@pytest.mark.parametrize("helper", ["idle", "install", "nixos"])
def test_retired_build_helper_stops_before_running_any_command(tmp_path, helper, capsys):
    marker = tmp_path / "ran"
    script = tmp_path / "command.py"
    script.write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    with pytest.raises(SystemExit) as error:
        if helper == "idle":
            _run_with_idle_timeout([sys.executable, str(script)], tmp_path, idle_timeout_seconds=10)
        elif helper == "install":
            _run_npm_install_deterministic(sys.executable, tmp_path, extra_args=(str(script),))
        else:
            _nixos_build_env()
    assert error.value.code == 0
    assert not marker.exists()
    assert "run `hermes` again" in capsys.readouterr().err

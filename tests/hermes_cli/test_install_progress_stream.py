"""Installer progress must use the stream drained by the desktop updater."""

import subprocess
import sys

import pytest

from hermes_cli.main_install_repair import _run_install_with_heartbeat


@pytest.mark.parametrize("exit_code", [0, 7])
def test_installer_stderr_streams_to_stdout(tmp_path, monkeypatch, capfd, exit_code):
    import hermes_cli.main as main

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    # More than a pipe buffer of progress, from a real child on the native host.
    size = 256 * 1024
    cmd = [
        sys.executable,
        "-c",
        f"import sys; sys.stderr.write('x' * {size}); sys.stderr.flush(); sys.exit({exit_code})",
    ]
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as error:
            _run_install_with_heartbeat(cmd)
        assert error.value.returncode == exit_code
    else:
        _run_install_with_heartbeat(cmd)

    output = capfd.readouterr()
    assert output.out == "x" * size
    assert output.err == ""

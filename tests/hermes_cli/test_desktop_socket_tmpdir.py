"""The Electron child needs a socket-safe temp root, not a long CLI scratch path."""

import argparse
import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main_desktop


def _launch(tmp_path):
    with patch.object(main_desktop, "_desktop_launch_options", return_value=([], "auto", "basic", "auto")):
        return main_desktop._desktop_launch_env(argparse.Namespace(cwd=str(tmp_path)))


@pytest.mark.platforms("linux")
def test_long_scratch_root_can_host_electron_singleton_socket(monkeypatch, tmp_path):
    scratch = tmp_path / ("long-home-" * 8) / "cache" / "scratch"
    scratch.mkdir(parents=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(key, str(scratch))
    monkeypatch.setenv("HERMES_SCRATCH_DIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)

    env, _ = _launch(tmp_path)

    with tempfile.TemporaryDirectory(prefix="scoped_dir", dir=env["TMPDIR"]) as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(Path(directory) / "SingletonSocket"))
    assert os.environ["TMPDIR"] == str(scratch)
    assert env["TMP"] == env["TEMP"] == str(scratch)
    assert env["HERMES_SCRATCH_DIR"] == str(scratch)


@pytest.mark.platforms("linux")
def test_short_caller_selected_temp_root_is_preserved(monkeypatch, tmp_path):
    # Use the existing short-root resolver so this fixture also works under deep CI homes.
    from hermes_constants import socket_safe_tmpdir

    with tempfile.TemporaryDirectory(prefix="dt-", dir=socket_safe_tmpdir()) as directory:
        monkeypatch.setenv("TMPDIR", directory)
        monkeypatch.setattr(tempfile, "tempdir", None)
        env, _ = _launch(tmp_path)
        assert env["TMPDIR"] == directory


@pytest.mark.platforms("windows")
def test_windows_temp_environment_is_not_retargeted(monkeypatch, tmp_path):
    for key in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(key, str(tmp_path))
    env, _ = _launch(tmp_path)
    assert all(env[key] == str(tmp_path) for key in ("TMPDIR", "TMP", "TEMP"))

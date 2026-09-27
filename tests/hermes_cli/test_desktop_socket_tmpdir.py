"""The Electron child needs a socket-safe temp root, not a long CLI scratch path (#124688)."""

import argparse
import socket
import tempfile
from pathlib import Path

import pytest

from hermes_cli import main_desktop


def _launch_env(tmp_path) -> dict:
    env, _flags = main_desktop._desktop_launch_env(argparse.Namespace(cwd=str(tmp_path)))
    return env


@pytest.mark.platforms("linux")
def test_long_scratch_tmpdir_still_hosts_electrons_singleton_socket(monkeypatch, tmp_path):
    scratch = tmp_path / ("long-home-" * 8) / "cache" / "scratch"
    scratch.mkdir(parents=True)
    for key in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR"):
        monkeypatch.setenv(key, str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)

    env = _launch_env(tmp_path)

    # Chromium's ProcessSingleton binds $TMPDIR/scoped_dirXXXXXX/SingletonSocket.
    with tempfile.TemporaryDirectory(prefix="scoped_dir", dir=env["TMPDIR"]) as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(Path(directory) / "SingletonSocket"))


@pytest.mark.platforms("linux")
def test_short_caller_tmpdir_is_passed_through(monkeypatch, tmp_path):
    from hermes_constants import socket_safe_tmpdir

    with tempfile.TemporaryDirectory(prefix="dt-", dir=socket_safe_tmpdir()) as directory:
        monkeypatch.setenv("TMPDIR", directory)
        monkeypatch.setattr(tempfile, "tempdir", None)
        assert _launch_env(tmp_path)["TMPDIR"] == directory

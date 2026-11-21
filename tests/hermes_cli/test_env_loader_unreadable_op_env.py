"""A profile .op.env that exists but cannot be opened, as a sandbox launcher leaves it.

``load_hermes_dotenv`` treats it like a profile .env in that state: while managed scope pins
``terminal.backend`` the file loads as absent, with one warning, and without the pin the read error
still stops the load. Besides a stand-in for the refused open, the states are real ones: a file this
user is denied, a file another handle holds without sharing (Windows), and a directory under the
file's name.
"""

import builtins
import contextlib
import ctypes
import io
import logging
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import managed_scope
from hermes_cli.env_loader import load_hermes_dotenv

TOKEN = "OP_SERVICE_ACCOUNT_TOKEN"


@pytest.fixture(autouse=True)
def _no_process_token(monkeypatch):
    """.op.env is read only without a process token; what a test loads is unset at teardown."""
    for name in (TOKEN, "LAB_OP_ENV_FIRST", "LAB_OP_ENV_SECOND"):
        monkeypatch.setenv(name, "unset at teardown")
        monkeypatch.delenv(name)


@pytest.fixture
def managed_dir(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    managed_scope.invalidate_managed_cache()
    yield md
    managed_scope.invalidate_managed_cache()


@pytest.fixture
def pinned(managed_dir):
    (managed_dir / "config.yaml").write_text("terminal:\n  backend: busybox\n", encoding="utf-8")
    managed_scope.invalidate_managed_cache()
    return managed_dir


def _profile(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "hermes"
    home.mkdir()
    op_env = home / ".op.env"
    op_env.write_text(f"{TOKEN}=from-file\n", encoding="utf-8")
    return home, op_env


def _refuse_opens(monkeypatch, *paths: Path) -> None:
    """Make every open of *paths* fail the way Windows fails it for a file a sandbox is denied."""
    refused = {os.path.abspath(path) for path in paths}
    real_open = io.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and os.path.abspath(file) in refused:
            raise PermissionError(13, "Access is denied", os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)


@contextlib.contextmanager
def _denied(path: Path):
    """Refuse this user every read of *path*: a deny entry in its ACL on Windows, mode 000 elsewhere."""
    if os.name == "nt":
        whoami = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True)
        trustee = "*" + whoami.stdout.strip().rsplit(",", 1)[-1].strip('"')  # the user's SID
        subprocess.run(["icacls", str(path), "/deny", f"{trustee}:(RD)"], capture_output=True, check=True)
    else:
        path.chmod(0)
    try:
        yield
    finally:
        if os.name == "nt":
            subprocess.run(["icacls", str(path), "/remove:d", trustee], capture_output=True, check=True)
        else:
            path.chmod(0o600)


@contextlib.contextmanager
def _held_without_sharing(path: Path):
    """Hold *path* open with no share mode, as a scanner or a sync client can: other opens are refused."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                     ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)  # GENERIC_READ, OPEN_EXISTING
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)


@contextlib.contextmanager
def _replaced_by_a_directory(path: Path):
    path.unlink()
    path.mkdir()
    yield


def _loads_as_absent(home: Path, op_env: Path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        for _ in range(3):  # gateways reload the env files every turn
            assert load_hermes_dotenv(hermes_home=home) == []
    assert TOKEN not in os.environ
    warnings = [r for r in caplog.records if str(op_env) in r.getMessage()]
    assert [r.levelno for r in warnings] == [logging.WARNING]


def test_unreadable_op_env_loads_as_absent_under_a_pinned_backend(tmp_path, monkeypatch, caplog, pinned):
    home, op_env = _profile(tmp_path)
    _refuse_opens(monkeypatch, op_env)

    _loads_as_absent(home, op_env, caplog)


def test_unreadable_op_env_gets_the_warning_an_unreadable_env_gets(tmp_path, monkeypatch, caplog, pinned):
    home, op_env = _profile(tmp_path)
    env_file = home / ".env"
    env_file.write_text("LAB_OP_ENV_FIRST=from-file\n", encoding="utf-8")
    _refuse_opens(monkeypatch, env_file, op_env)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        assert load_hermes_dotenv(hermes_home=home) == []

    messages = {name: [r.getMessage().replace(str(home / name), "<file>") for r in caplog.records
                       if str(home / name) in r.getMessage()] for name in (".env", ".op.env")}
    assert len(messages[".env"]) == 1
    assert messages[".op.env"] == messages[".env"]


@pytest.mark.parametrize("managed_yaml", [None, "terminal:\n  timeout: 5\n", "terminal:\n  backend: ''\n"])
def test_unreadable_op_env_raises_without_a_pinned_backend(tmp_path, monkeypatch, managed_dir, managed_yaml):
    """As for .env: the file may be the only place the backend is set."""
    if managed_yaml is not None:
        (managed_dir / "config.yaml").write_text(managed_yaml, encoding="utf-8")
        managed_scope.invalidate_managed_cache()
    home, op_env = _profile(tmp_path)
    _refuse_opens(monkeypatch, op_env)

    with pytest.raises(PermissionError):
        load_hermes_dotenv(hermes_home=home)


def test_readable_op_env_still_only_fills_gaps(tmp_path, monkeypatch, pinned):
    home, op_env = _profile(tmp_path)
    (home / ".env").write_text("LAB_OP_ENV_FIRST=from-env\n", encoding="utf-8")
    op_env.write_text(f"LAB_OP_ENV_FIRST=from-op-env\nLAB_OP_ENV_SECOND=from-op-env\n{TOKEN}=from-file\n",
                      encoding="utf-8")
    monkeypatch.setenv("LAB_OP_ENV_SECOND", "from-process")

    assert load_hermes_dotenv(hermes_home=home) == [home / ".env"]

    assert os.environ["LAB_OP_ENV_FIRST"] == "from-env"
    assert os.environ["LAB_OP_ENV_SECOND"] == "from-process"
    assert os.environ[TOKEN] == "from-file"


_WINDOWS_STATES = pytest.mark.parametrize("state", [_denied, _held_without_sharing, _replaced_by_a_directory])


@pytest.mark.platforms("windows")
@_WINDOWS_STATES
def test_op_env_windows_cannot_open_loads_as_absent_under_a_pinned_backend(tmp_path, caplog, pinned, state):
    home, op_env = _profile(tmp_path)

    with state(op_env):
        _loads_as_absent(home, op_env, caplog)


@pytest.mark.platforms("windows")
@_WINDOWS_STATES
def test_op_env_windows_cannot_open_raises_without_a_pinned_backend(tmp_path, managed_dir, state):
    home, op_env = _profile(tmp_path)

    with state(op_env), pytest.raises(PermissionError):
        load_hermes_dotenv(hermes_home=home)


_ROOT = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")


@pytest.mark.platforms("posix")
@_ROOT
def test_op_env_without_read_permission_loads_as_absent_under_a_pinned_backend(tmp_path, caplog, pinned):
    home, op_env = _profile(tmp_path)

    with _denied(op_env):
        _loads_as_absent(home, op_env, caplog)


@pytest.mark.platforms("posix")
@_ROOT
def test_op_env_without_read_permission_raises_without_a_pinned_backend(tmp_path, managed_dir):
    home, op_env = _profile(tmp_path)

    with _denied(op_env), pytest.raises(PermissionError):
        load_hermes_dotenv(hermes_home=home)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name", [".env", ".op.env"])
def test_a_directory_under_either_name_stops_the_load_on_posix(tmp_path, pinned, name):
    """Opening a directory is not a refusal there (``IsADirectoryError``), so the pin does not cover it."""
    home, _op_env = _profile(tmp_path)
    (home / ".op.env").unlink()
    (home / name).mkdir()

    with pytest.raises(IsADirectoryError):
        load_hermes_dotenv(hermes_home=home)

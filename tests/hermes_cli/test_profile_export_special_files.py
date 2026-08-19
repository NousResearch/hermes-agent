"""Tests for special-file handling during profile export.

:func:`shutil.copytree` cannot copy Unix sockets, FIFOs, or device nodes —
it collects "[Errno 6] No such device or address" and raises
``shutil.Error`` at the end of the walk. Profiles accumulate such files in
normal operation (e.g. a stale agent-browser control socket under
``home/.agent-browser/``), so exports must skip them instead of failing
the whole archive.
"""

import os
import socket
import sys
import tarfile

import pytest

from hermes_cli.profiles import export_profile

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix sockets and FIFOs are not available on Windows",
)


def _patch_named_profile(monkeypatch, profiles_root, profile_dir):
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: profile_dir)
    monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)


def _bind_unix_socket(path):
    """Bind an AF_UNIX socket at *path*.

    Binds via a cwd-relative name so pytest's long tmp paths stay under the
    ~104-byte ``sun_path`` limit on macOS. The filesystem entry outlives the
    close — exactly the stale-socket state a real profile ends up with.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        sock.bind(path.name)
    finally:
        os.chdir(cwd)
        sock.close()


def test_named_profile_export_survives_unix_socket(tmp_path, monkeypatch):
    """Sockets and FIFOs in a named profile are skipped, not fatal."""
    profiles_root = tmp_path / "profiles"
    profile_dir = profiles_root / "sockety"
    browser_dir = profile_dir / "home" / ".agent-browser"
    browser_dir.mkdir(parents=True)

    (profile_dir / "config.yaml").write_text("model: gpt-4\n")
    (browser_dir / "state.json").write_text("{}\n")

    # One socket the *.sock suffix rule catches, one only lstat can catch,
    # and a FIFO — none of them may fail or enter the export.
    _bind_unix_socket(browser_dir / "dev-12345.sock")
    _bind_unix_socket(browser_dir / "control")
    os.mkfifo(browser_dir / "events")

    _patch_named_profile(monkeypatch, profiles_root, profile_dir)

    result = export_profile("sockety", str(tmp_path / "sockety.tar.gz"))

    with tarfile.open(result, "r:gz") as tf:
        names = tf.getnames()

    assert any("config.yaml" in n for n in names)
    assert any("state.json" in n for n in names)
    assert not any(".sock" in n for n in names)
    assert not any(n.endswith(("control", "events")) for n in names)


def test_default_profile_export_survives_unix_socket(tmp_path, monkeypatch):
    """The default-profile export skips suffixless sockets in allowed dirs."""
    profile_dir = tmp_path / "hermes_home"
    sessions_dir = profile_dir / "sessions"
    sessions_dir.mkdir(parents=True)

    (profile_dir / "config.yaml").write_text("model: gpt-4\n")
    (sessions_dir / "log.jsonl").write_text("{}\n")
    _bind_unix_socket(sessions_dir / "ipc")

    _patch_named_profile(monkeypatch, tmp_path / "profiles", profile_dir)

    result = export_profile("default", str(tmp_path / "default.tar.gz"))

    with tarfile.open(result, "r:gz") as tf:
        names = tf.getnames()

    assert any("config.yaml" in n for n in names)
    assert any("log.jsonl" in n for n in names)
    assert not any(n.endswith("ipc") for n in names)

"""A client must not take answers from a socket it does not own.

`control_socket`'s stated trust model is that filesystem ACLs are the auth
boundary. That holds for `$HERMES_HOME/gateway.sock` — the home is the
user's. It does not hold for the temp-dir fallback used when the home path
exceeds `sun_path`: on Linux `tempfile.gettempdir()` is normally the shared,
world-writable `/tmp`, and the filename is `hermes-gw-<sha256(home)[:16]>.sock`
— unsalted, so anyone who can guess the home path can compute it and bind
there first.

The server already binds under `umask(0o177)` and chmods 0600. These tests
pin the client half: anything not owned by this user, or reachable by anyone
else, is treated as absent so the caller falls back to the scan layer.
"""

import os
import shutil
import socket
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from gateway.control_socket import (
    _home_hash,
    _pointer_path,
    _is_trustworthy_socket,
    resolve_client_socket_path,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX ownership/mode check; Windows uses a per-user pipe namespace",
)


@pytest.fixture()
def shortdir():
    """A short-path scratch dir.

    pytest's tmp_path lives under /private/var/folders/... on macOS, which
    already exceeds sun_path — the very condition that makes production fall
    back to the temp dir. Bind somewhere short instead.
    """
    d = Path(tempfile.mkdtemp(prefix="hgw-", dir=tempfile.gettempdir()))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def home(shortdir):
    h = shortdir / "h"
    h.mkdir()
    return h


def _bind(path, mode=0o600):
    path.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(1)
    os.chmod(path, mode)
    return srv


class TestDirectPath:
    def test_a_socket_we_own_is_used(self, home):
        sock = home / "gateway.sock"
        srv = _bind(sock)
        try:
            assert resolve_client_socket_path(home) == sock
        finally:
            srv.close()

    @pytest.mark.parametrize("mode", [0o777, 0o660, 0o606])
    def test_a_reachable_socket_is_refused(self, home, mode):
        srv = _bind(home / "gateway.sock", mode)
        try:
            assert resolve_client_socket_path(home) is None, (
                f"mode {mode:o} lets another account answer identify/status"
            )
        finally:
            srv.close()

    def test_a_regular_file_is_not_a_socket(self, home):
        f = home / "gateway.sock"
        f.write_text("not a socket", encoding="utf-8")
        os.chmod(f, 0o600)
        assert resolve_client_socket_path(home) is None


class TestPointerFallback:
    """The temp-dir path is the one a shared /tmp exposes."""

    def test_a_decoy_at_the_predictable_path_is_refused(self, home, shortdir):
        decoy = shortdir / f"hermes-gw-{_home_hash(home)}.sock"
        srv = _bind(decoy, 0o777)
        _pointer_path(home).write_text(str(decoy), encoding="utf-8")
        try:
            assert resolve_client_socket_path(home) is None, (
                "client accepted a world-connectable socket at the "
                "unsalted, guessable fallback path"
            )
        finally:
            srv.close()

    def test_a_genuine_fallback_socket_still_resolves(self, home, shortdir):
        real = shortdir / f"hermes-gw-{_home_hash(home)}.sock"
        srv = _bind(real, 0o600)
        _pointer_path(home).write_text(str(real), encoding="utf-8")
        try:
            assert resolve_client_socket_path(home) == real
        finally:
            srv.close()

    def test_a_pointer_to_nothing_is_absent(self, home, shortdir):
        _pointer_path(home).write_text(str(shortdir / "gone.sock"), encoding="utf-8")
        assert resolve_client_socket_path(home) is None


class TestPredicate:
    def test_a_missing_path_is_not_trustworthy(self, shortdir):
        assert _is_trustworthy_socket(shortdir / "nope.sock") is False

    def test_owner_only_socket_passes(self, shortdir):
        p = shortdir / "ok.sock"
        srv = _bind(p, 0o600)
        try:
            assert _is_trustworthy_socket(p) is True
            assert stat.S_IMODE(os.stat(p).st_mode) & 0o077 == 0
        finally:
            srv.close()


class TestPeerCredentials:
    """Closes the stat-then-connect race the path check alone leaves open.

    A socket can be swapped between the `stat` and the `connect`, so the
    filesystem check is a filter, not a guarantee. `SO_PEERCRED` reports the
    process actually on the other end.

    Mocked rather than driven live: `SO_PEERCRED` is Linux-only, and that is
    also the platform carrying the exposure — macOS gives each user a private
    `gettempdir()`, so the shared-`/tmp` fallback never applies there.
    """

    def test_absent_support_reports_unknown_rather_than_failing(self, monkeypatch):
        import socket as _socket

        from gateway import control_socket as cs

        monkeypatch.delattr(_socket, "SO_PEERCRED", raising=False)
        assert cs._peer_euid(object()) is None

    def test_the_peer_uid_is_unpacked(self, monkeypatch):
        import socket as _socket
        import struct as _struct

        from gateway import control_socket as cs

        monkeypatch.setattr(_socket, "SO_PEERCRED", 17, raising=False)

        class _Sock:
            def getsockopt(self, level, opt, size):
                return _struct.pack("3i", 4242, 1000, 1000)

        assert cs._peer_euid(_Sock()) == 1000

    def test_a_refusing_kernel_is_not_an_error(self, monkeypatch):
        import socket as _socket

        from gateway import control_socket as cs

        monkeypatch.setattr(_socket, "SO_PEERCRED", 17, raising=False)

        class _Sock:
            def getsockopt(self, *a):
                raise OSError("not supported")

        assert cs._peer_euid(_Sock()) is None

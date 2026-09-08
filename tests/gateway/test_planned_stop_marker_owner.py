"""Regression tests for issue #91212.

The CLI that *stops* a gateway writes ``.gateway-planned-stop.json`` so the gateway's shutdown
handler can classify the SIGTERM as operator-initiated (exit 0, ``gateway_state=stopped``) instead
of an unexpected exit (non-zero status, shutdown forensics, failure alerts). That stopper is routinely more privileged than
the gateway: ``sudo hermes gateway stop`` for a system unit whose ``User=`` is a login account, and
``docker exec`` into the container, where s6 runs as root while the gateway runs under
``s6-setuidgid hermes``. ``atomic_json_write`` creates the marker as its writer, so it landed
root:root 0600 inside a home owned by the gateway user — unreadable by the gateway, so the stop was
misclassified and the file stayed in ``~/.hermes`` forever (the reported symptom).

Two behaviour contracts are pinned here:

1. Hand-over: a privileged (euid 0) marker writer gives the file to the owner of the HERMES_HOME
   directory it lives in; unprivileged writers and root-owned homes chown nothing, and a failed
   chown is logged rather than turned into a write failure.
2. Reader-side diagnosis: a marker that exists but cannot be read by this uid says so in the log
   exactly once per path, so a misclassified stop is explainable after the fact.
"""

import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from gateway import status

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only: uid/gid ownership semantics"
)


@pytest.fixture()
def marker_home(tmp_path, monkeypatch):
    """An isolated HERMES_HOME for the markers under test."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _clear_unreadable_marker_warnings():
    """The once-per-path warn ledger is module state; keep it from leaking between tests."""
    status._unreadable_marker_paths_warned.clear()
    yield
    status._unreadable_marker_paths_warned.clear()


def _owners(mapping):
    """Replacement for ``status._path_owner`` backed by a ``{path: (uid, gid)}`` dict.

    Patching the seam instead of ``os.stat`` keeps the fake scoped to the ownership decision — a
    global ``os.stat`` fake also reaches the atomic write itself.
    """
    return lambda path: mapping.get(str(path))


def _record_chowns(monkeypatch):
    calls = []
    monkeypatch.setattr(
        status.os, "chown", lambda path, uid, gid: calls.append((str(path), uid, gid))
    )
    return calls


def _refuse_chown(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError(f"unexpected chown{args}")

    monkeypatch.setattr(status.os, "chown", _fail)


class TestRootWrittenMarkerOwnership:
    def test_root_writer_hands_marker_to_home_owner(self, marker_home, monkeypatch):
        """euid 0 + a home owned by the gateway user => the marker follows the home's owner."""
        marker = status._get_planned_stop_marker_path()
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)
        monkeypatch.setattr(status.os, "geteuid", lambda: 0)
        monkeypatch.setattr(
            status, "_path_owner", _owners({str(marker_home): (1000, 1000), str(marker): (0, 0)})
        )
        calls = _record_chowns(monkeypatch)

        assert status.write_planned_stop_marker(target_pid=12345) is True

        assert calls == [(str(marker), 1000, 1000)], (
            "a root-written planned-stop marker must be handed to the HERMES_HOME owner, "
            "otherwise the unprivileged gateway cannot read it (#91212)"
        )
        assert json.loads(marker.read_text(encoding="utf-8"))["target_pid"] == 12345

    def test_unprivileged_writer_never_chowns(self, marker_home, monkeypatch):
        """A same-uid writer already produced a readable marker; chown would only raise EPERM."""
        marker = status._get_planned_stop_marker_path()
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)
        monkeypatch.setattr(status.os, "geteuid", lambda: 1000)
        monkeypatch.setattr(
            status, "_path_owner", _owners({str(marker_home): (1000, 1000), str(marker): (0, 0)})
        )
        _refuse_chown(monkeypatch)

        assert status.write_planned_stop_marker(target_pid=12345) is True
        assert marker.exists()

    def test_root_owned_home_is_left_alone(self, marker_home, monkeypatch):
        """A root-owned HERMES_HOME (root gateway, root stopper) has nobody to hand the marker to."""
        marker = status._get_planned_stop_marker_path()
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)
        monkeypatch.setattr(status.os, "geteuid", lambda: 0)
        monkeypatch.setattr(
            status, "_path_owner", _owners({str(marker_home): (0, 0), str(marker): (0, 0)})
        )
        _refuse_chown(monkeypatch)

        assert status.write_planned_stop_marker(target_pid=12345) is True
        assert marker.exists()

    def test_chown_failure_keeps_the_write_successful_and_warns(
        self, marker_home, monkeypatch, caplog
    ):
        """The marker WAS written; a hand-over failure is a diagnosable warning, not a failure."""
        marker = status._get_planned_stop_marker_path()
        monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)
        monkeypatch.setattr(status.os, "geteuid", lambda: 0)
        monkeypatch.setattr(
            status, "_path_owner", _owners({str(marker_home): (1000, 1000), str(marker): (0, 0)})
        )

        def _boom(*args, **kwargs):
            raise PermissionError("simulated chown failure")

        monkeypatch.setattr(status.os, "chown", _boom)

        with caplog.at_level(logging.WARNING, logger="gateway.status"):
            assert status.write_planned_stop_marker(target_pid=12345) is True

        assert marker.exists()
        warnings = [r.getMessage() for r in caplog.records if "#91212" in r.getMessage()]
        assert len(warnings) == 1, "a failed hand-over must be reported once, with the issue ref"
        assert str(marker) in warnings[0] and "1000" in warnings[0]

    def test_takeover_marker_into_another_home_is_handed_over_too(self, tmp_path, monkeypatch):
        """``--replace`` routes a takeover marker into the TARGET's home; same hand-over applies."""
        target_home = tmp_path / "target-home"
        target_home.mkdir()
        marker = status._get_takeover_marker_path(target_home)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "replacer-home"))
        monkeypatch.setattr(status.os, "geteuid", lambda: 0)
        monkeypatch.setattr(
            status, "_path_owner", _owners({str(marker.parent): (1000, 1000), str(marker): (0, 0)})
        )
        calls = _record_chowns(monkeypatch)

        assert status.write_takeover_marker(
            4242, target_home=target_home, target_start_time=7
        ) is True

        assert calls == [(str(marker), 1000, 1000)]
        assert json.loads(marker.read_text(encoding="utf-8"))["target_pid"] == 4242


class TestUnreadableMarkerDiagnostics:
    def test_unreadable_marker_warns_exactly_once_per_path(
        self, marker_home, monkeypatch, caplog
    ):
        """The planned-stop watcher polls twice a second: the diagnosis must not become a flood."""
        marker = status._get_planned_stop_marker_path()
        marker.write_text('{"target_pid": 4321}', encoding="utf-8")
        # What an EACCES read looks like from the reader's side: _read_json_file swallows it.
        monkeypatch.setattr(status, "_read_json_file", lambda path, **kwargs: None)
        real_access = os.access
        monkeypatch.setattr(
            status.os, "access",
            lambda path, mode, **kwargs: (
                False if str(path) == str(marker) else real_access(path, mode, **kwargs)
            ),
        )
        monkeypatch.setattr(status, "_path_owner", _owners({str(marker): (0, 0)}))

        with caplog.at_level(logging.WARNING, logger="gateway.status"):
            assert status.planned_stop_marker_targets_self() is False
            assert status.planned_stop_marker_targets_self() is False

        warnings = [r.getMessage() for r in caplog.records if "#91212" in r.getMessage()]
        assert len(warnings) == 1, "one warning per marker path per process, not one per poll"
        assert str(marker) in warnings[0]

    def test_readable_marker_is_never_reported_as_unreadable(self, marker_home, caplog):
        """A plain absent/stale marker is the normal case and must stay silent."""
        with caplog.at_level(logging.WARNING, logger="gateway.status"):
            assert status.planned_stop_marker_targets_self() is False

        assert not [r for r in caplog.records if "#91212" in r.getMessage()]


def _unprivileged_account():
    """``(uid, gid)`` of a real non-root account to hand a marker to, or None."""
    import pwd

    with contextlib.suppress(KeyError):
        entry = pwd.getpwnam("nobody")
        return (entry.pw_uid, entry.pw_gid)
    entry = next(
        (e for e in sorted(pwd.getpwall(), key=lambda e: e.pw_uid) if e.pw_uid >= 1000), None
    )
    return (entry.pw_uid, entry.pw_gid) if entry else None


def _target_pid_read_as(uid: int, gid: int, marker: Path) -> str:
    """Parse ``target_pid`` out of *marker* in a child process running as ``(uid, gid)``."""
    read_fd, write_fd = os.pipe()
    child = os.fork()  # windows-footgun: ok — linux_only real-uid probe
    if child == 0:  # pragma: no cover - child process
        code = 1
        try:
            os.close(read_fd)
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            os.write(write_fd, str(json.loads(marker.read_text(encoding="utf-8"))["target_pid"]).encode())
            code = 0
        except BaseException as exc:  # reported to the parent through the pipe
            with contextlib.suppress(OSError):
                os.write(write_fd, f"ERROR {type(exc).__name__}: {exc}".encode())
        finally:
            os._exit(code)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        output = pipe.read().decode()
    _, wait_status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(wait_status) == 0, f"reader child failed: {output}"
    return output


@pytest.mark.linux_only
def test_root_written_marker_is_readable_by_the_real_home_owner(monkeypatch):
    """End-to-end with real uids: root stops a gateway whose HERMES_HOME belongs to someone else.

    No patched ownership seams — the marker is written by this (root) process into a directory
    owned by an unprivileged account, and a process running AS that account reads it back. This is
    the shape of ``sudo hermes gateway stop`` and of the s6 supervisor stopping the container's
    ``s6-setuidgid hermes`` gateway.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() != 0:
        pytest.skip("needs a real root writer")
    account = _unprivileged_account()
    if account is None:
        pytest.skip("no unprivileged account available to hand the marker to")
    uid, gid = account

    # Not tmp_path: the unprivileged child must be able to traverse every parent directory, and
    # pytest's temp root is 0700 root-owned.
    workdir = Path(tempfile.mkdtemp(prefix="hermes-marker-owner-"))
    try:
        os.chmod(workdir, 0o755)
        home = workdir / "hermes-home"
        home.mkdir()
        os.chown(home, uid, gid)
        os.chmod(home, 0o700)
        monkeypatch.setenv("HERMES_HOME", str(home))
        marker = status._get_planned_stop_marker_path()

        assert status.write_planned_stop_marker(os.getpid()) is True

        marker_stat = marker.stat()
        assert (marker_stat.st_uid, marker_stat.st_gid) == (uid, gid), (
            "a root-written marker inside a user-owned HERMES_HOME must end up owned by that "
            "user, not root:root (#91212)"
        )
        assert _target_pid_read_as(uid, gid, marker) == str(os.getpid()), (
            "the account the gateway runs as must be able to parse the marker"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

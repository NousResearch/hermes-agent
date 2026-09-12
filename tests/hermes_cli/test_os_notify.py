"""Tests for hermes_cli/os_notify.py.

The tests verify the frozen interface and behaviour without spawning real OS
notifications (except for a host-native sanity check on macOS).  All process
spawning is mocked; the real ``subprocess.Popen`` is patched to capture calls and
verify the correct argv is built.
"""

import os
import shutil
import subprocess
from typing import List

import pytest

import hermes_cli.os_notify
from hermes_cli.os_notify import notifier_argv, notify, usable_kind

# ---------------------------------------------------------------------------
# notifier_argv tests
# ---------------------------------------------------------------------------

def test_notifier_argv_darwin():
    sentinel_title = "SENTINEL-TITLE-7f3a"
    sentinel_body = "SENTINEL-BODY-9c1b"
    argv = notifier_argv("darwin", sentinel_title, sentinel_body)
    expected = [
        "osascript",
        "-e",
        "on run {t, b}",
        "-e",
        "display notification b with title t",
        "-e",
        "end run",
        "--",
        sentinel_title,
        sentinel_body,
    ]
    assert argv == expected

    # Invariant: caller text is never part of the script source;
    # it travels as separate argv items after '--'.
    script_elements = argv[:8]  # flags, script lines, and '--'
    assert not any(sentinel_title in arg for arg in script_elements)
    assert not any(sentinel_body in arg for arg in script_elements)
    assert argv[-2] == sentinel_title
    assert argv[-1] == sentinel_body

    # Verify that a body containing quotes/backslashes survives as a single argument.
    test_body = 'say "hello" \'world\''
    argv2 = notifier_argv("darwin", "t", test_body)
    assert argv2[-1] == test_body
    assert not any(test_body in arg for arg in argv2[:8])


def test_notifier_argv_linux():
    sentinel_title = "SENTINEL-TITLE-7f3a"
    sentinel_body = "SENTINEL-BODY-9c1b"
    argv = notifier_argv("linux", sentinel_title, sentinel_body)
    expected = ["notify-send", "--app-name=Hermes", sentinel_title, sentinel_body]
    assert argv == expected

    # Invariant: caller text is never part of command tokens;
    # it travels as separate trailing argv items.
    command_elements = argv[:2]
    assert not any(sentinel_title in arg for arg in command_elements)
    assert not any(sentinel_body in arg for arg in command_elements)
    assert argv[-2] == sentinel_title
    assert argv[-1] == sentinel_body

    # Body with special characters should be passed as a single argument.
    test_body = "a'b'c"
    argv2 = notifier_argv("linux", "t", test_body)
    assert argv2[-1] == test_body
    assert test_body not in argv2[0]
    assert test_body not in argv2[1]


def test_notifier_argv_win32():
    # Windows delivery is not implemented; notifier_argv returns None.
    assert notifier_argv("win32", "title", "body") is None


def test_notifier_argv_unknown():
    assert notifier_argv("unknown", "title", "body") is None
    assert notifier_argv("freebsd", "title", "body") is None

# ---------------------------------------------------------------------------
# usable_kind tests
# ---------------------------------------------------------------------------

def test_usable_kind_darwin(monkeypatch):
    # Mock shutil.which to return a path for osascript.
    dummy_path = "/usr/bin/osascript"
    monkeypatch.setattr(shutil, "which", lambda cmd: dummy_path if cmd == "osascript" else None)
    kind = usable_kind(platform="darwin")
    assert kind == "darwin"

    # Missing binary should return None.
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert usable_kind(platform="darwin") is None


def test_usable_kind_linux(monkeypatch):
    dummy_path = "/usr/bin/notify-send"
    monkeypatch.setattr(shutil, "which", lambda cmd: dummy_path if cmd == "notify-send" else None)
    kind = usable_kind(platform="linux")
    assert kind == "linux"

    # Missing notify-send should return None.
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert usable_kind(platform="linux") is None


def test_usable_kind_win32(monkeypatch):
    # Windows notifier is not implemented; usable_kind returns None even when powershell exists.
    dummy_path = "C:\\Windows\\System32\\powershell.exe"
    which_calls = []

    def fake_which(cmd):
        which_calls.append(cmd)
        return dummy_path if cmd == "powershell" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    assert usable_kind(platform="win32") is None
    # No powershell probe should be executed for win32
    assert "powershell" not in which_calls


def test_usable_kind_unknown_platform():
    assert usable_kind(platform="freebsd") is None


def test_usable_kind_ssh_env(monkeypatch):
    # Simulate SSH environment variables.
    env = {"SSH_CONNECTION": "192.168.1.1 22", "PATH": "/usr/bin"}
    monkeypatch.setattr(os, "environ", env)
    # Even if binaries exist, SSH should disable notifications.
    dummy_path = "/usr/bin/osascript"
    monkeypatch.setattr(shutil, "which", lambda cmd: dummy_path if cmd == "osascript" else None)
    assert usable_kind(env=env, platform="darwin") is None


def test_usable_kind_ssh_tty(monkeypatch):
    env = {"SSH_TTY": "/dev/pts/0"}
    monkeypatch.setattr(os, "environ", env)
    dummy_path = "/usr/bin/osascript"
    monkeypatch.setattr(shutil, "which", lambda cmd: dummy_path)
    assert usable_kind(env=env, platform="darwin") is None

# ---------------------------------------------------------------------------
# notify tests (mocked subprocess)
# ---------------------------------------------------------------------------

def test_notify_success(monkeypatch):
    # Patch subprocess.Popen to capture the call.
    calls = []

    def mock_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    # Test each supported platform.
    for kind in ["darwin", "linux"]:
        # Force usable_kind to return the expected kind.
        monkeypatch.setattr(
            hermes_cli.os_notify,
            "usable_kind",
            lambda *a, **kw: kind,
        )
        # Ensure notifier_argv returns something.
        monkeypatch.setattr(
            hermes_cli.os_notify,
            "notifier_argv",
            lambda k, t, b: ["cmd", "arg1", t, b],
        )
        result = notify("title", "body")
        assert result is True
        # Verify Popen was called with the expected argv.
        (args, kwargs) = calls[-1]
        expected_argv = ["cmd", "arg1", "title", "body"]
        assert args[0] == expected_argv
        # Verify detach kwargs are applied (we only check that they were passed).
        assert "start_new_session" in kwargs or "stdin" in kwargs

    # Reset monkeypatches for the failure cases.
    monkeypatch.undo()


def test_notify_no_notifier(monkeypatch):
    # When usable_kind returns None, notify should return False and not spawn.
    calls = []

    def mock_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    monkeypatch.setattr(
        hermes_cli.os_notify,
        "usable_kind",
        lambda *a, **kw: None,
    )
    result = notify("title", "body")
    assert result is False
    assert not calls
    monkeypatch.undo()


def test_notify_popen_failure(monkeypatch):
    # If Popen raises an exception, notify should return False and not propagate.
    def mock_popen(*args, **kwargs):
        raise OSError("failed to spawn")

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    monkeypatch.setattr(
        hermes_cli.os_notify,
        "usable_kind",
        lambda *a, **kw: "darwin",
    )
    monkeypatch.setattr(
        hermes_cli.os_notify,
        "notifier_argv",
        lambda k, t, b: ["osascript", "-e", "..."],
    )
    result = notify("title", "body")
    assert result is False
    monkeypatch.undo()


def test_notify_usable_kind_exception(monkeypatch):
    # If usable_kind raises an unexpected error, notify must catch it and return False.
    def failing_usable_kind(*args, **kwargs):
        raise RuntimeError("unexpected probe failure")

    monkeypatch.setattr(
        hermes_cli.os_notify,
        "usable_kind",
        failing_usable_kind,
    )
    assert notify("title", "body") is False
    monkeypatch.undo()


def test_notify_notifier_argv_exception(monkeypatch):
    # If notifier_argv raises an unexpected error, notify must catch it and return False.
    monkeypatch.setattr(
        hermes_cli.os_notify,
        "usable_kind",
        lambda *a, **kw: "darwin",
    )

    def failing_notifier_argv(*args, **kwargs):
        raise RuntimeError("unexpected argv failure")

    monkeypatch.setattr(
        hermes_cli.os_notify,
        "notifier_argv",
        failing_notifier_argv,
    )
    assert notify("title", "body") is False
    monkeypatch.undo()


def test_notify_unsupported_argv(monkeypatch):
    # If notifier_argv returns None, notify should return False without spawning.
    calls = []

    def mock_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(
        hermes_cli.os_notify,
        "usable_kind",
        lambda *a, **kw: "freebsd",
    )
    monkeypatch.setattr(
        hermes_cli.os_notify,
        "notifier_argv",
        lambda k, t, b: None,
    )
    assert notify("title", "body") is False
    assert not calls
    monkeypatch.undo()

# ---------------------------------------------------------------------------
# Host-native sanity check (macOS only)
# ---------------------------------------------------------------------------

@pytest.mark.macos_only
def test_real_notifier_usable():
    # This test runs only on macOS and verifies that the real notifier is detected.
    # It must not actually pop a notification; we only check the internal state.
    # Ensure osascript is present (the test will fail on a machine without it).
    assert shutil.which("osascript") is not None
    # Verify that the returned kind is darwin and that notifier_argv returns the expected shape.
    kind = usable_kind()
    assert kind == "darwin"
    argv = notifier_argv(kind, "title", "body")
    expected = [
        "osascript",
        "-e",
        "on run {t, b}",
        "-e",
        "display notification b with title t",
        "-e",
        "end run",
        "--",
        "title",
        "body",
    ]
    assert argv == expected

def test_notify_win32_does_not_spawn(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: calls.append((a, kw)))
    # No Windows notifier is implemented: `usable_kind()` cannot return "win32" on a real host, and
    # even if it did, `notifier_argv` returns None — either way nothing may be spawned. (The host is
    # never faked here; `usable_kind` is the seam.)
    monkeypatch.setattr(hermes_cli.os_notify, "usable_kind", lambda *a, **kw: "win32")
    assert notify("title", "body") is False
    assert not calls


def test_notify_real_argv_darwin_and_linux(monkeypatch):
    calls = []

    class DummyProc:
        def poll(self):
            return None

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **kw: calls.append((a, kw)) or DummyProc(),
    )

    title = "Test Title"
    body = "Test Body"

    # Darwin: real notifier_argv is wired through to Popen
    monkeypatch.setattr(hermes_cli.os_notify, "usable_kind", lambda *a, **kw: "darwin")
    assert notify(title, body) is True
    args, kwargs = calls[-1]
    expected_darwin = [
        "osascript",
        "-e",
        "on run {t, b}",
        "-e",
        "display notification b with title t",
        "-e",
        "end run",
        "--",
        title,
        body,
    ]
    assert args[0] == expected_darwin
    assert "start_new_session" in kwargs or "stdin" in kwargs

    # Linux: real notifier_argv is wired through to Popen
    monkeypatch.setattr(hermes_cli.os_notify, "usable_kind", lambda *a, **kw: "linux")
    assert notify(title, body) is True
    args, kwargs = calls[-1]
    expected_linux = ["notify-send", "--app-name=Hermes", title, body]
    assert args[0] == expected_linux
    assert "start_new_session" in kwargs or "stdin" in kwargs


def test_reap():
    class FakeProc:
        def __init__(self, exit_code):
            self.exit_code = exit_code
            self.polled = False

        def poll(self):
            self.polled = True
            return self.exit_code

    live1 = FakeProc(None)
    exited1 = FakeProc(0)
    live2 = FakeProc(None)
    exited2 = FakeProc(1)

    saved_active = hermes_cli.os_notify._ACTIVE[:]
    try:
        hermes_cli.os_notify._ACTIVE = [live1, exited1, live2, exited2]
        hermes_cli.os_notify._reap()
        assert live1.polled is True
        assert exited1.polled is True
        assert live2.polled is True
        assert exited2.polled is True
        assert hermes_cli.os_notify._ACTIVE == [live1, live2]
    finally:
        hermes_cli.os_notify._ACTIVE = saved_active

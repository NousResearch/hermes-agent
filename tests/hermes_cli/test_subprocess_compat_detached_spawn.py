"""``popen_detached`` — the detached-spawn retry contract, and the /update call site.

The Windows branch is exercised on every host: ``is_windows`` is the platform as data
(see ``tests/_fixtures/platform_gating.py`` — a function that takes the platform as an
argument is not faking the host) and ``subprocess.Popen`` / the flag builders are the
mocked dependencies. Never ``sys.platform``.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from hermes_cli import _subprocess_compat as compat

# Distinct sentinels so an assertion says which attempt produced them.
BREAKAWAY_FLAGS = 0x0B000000
PLAIN_FLAGS = 0x00200000
CALLER_FLAGS = 0x00000400


def _access_denied() -> OSError:
    """The ``PermissionError`` CreateProcess raises when the job forbids breakaway."""
    exc = PermissionError(13, "Access is denied")
    exc.winerror = 5
    return exc


def _recording_popen(outcomes):
    """A ``Popen`` stand-in recording each call; *outcomes* is per-call, merged with success."""
    calls: list[tuple[list[str], dict]] = []
    results = list(outcomes)

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        outcome = results.pop(0) if results else None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return fake_popen, calls


@pytest.fixture(autouse=True)
def _flag_sentinels(monkeypatch):
    """Pin the flag builders so the assertions are host-independent."""
    monkeypatch.setattr(compat, "windows_detach_flags", lambda: BREAKAWAY_FLAGS)
    monkeypatch.setattr(compat, "windows_detach_flags_without_breakaway", lambda: PLAIN_FLAGS)


class TestPopenDetached:
    def test_windows_attempts_breakaway_first(self):
        fake, calls = _recording_popen(["child"])
        with patch.object(subprocess, "Popen", fake):
            assert compat.popen_detached(["cmd"], is_windows=True) == "child"
        assert len(calls) == 1
        assert calls[0][1]["creationflags"] == BREAKAWAY_FLAGS

    def test_winerror_five_retries_once_without_breakaway(self, caplog):
        fake, calls = _recording_popen([_access_denied(), "child"])
        with patch.object(subprocess, "Popen", fake), caplog.at_level("WARNING"):
            assert compat.popen_detached(["cmd"], is_windows=True) == "child"
        assert [call[1]["creationflags"] for call in calls] == [BREAKAWAY_FLAGS, PLAIN_FLAGS]
        assert "CREATE_BREAKAWAY_FROM_JOB" in caplog.text

    def test_other_oserror_is_not_retried(self):
        err = OSError(87, "The parameter is incorrect")
        err.winerror = 87
        fake, calls = _recording_popen([err, "child"])
        with patch.object(subprocess, "Popen", fake), pytest.raises(OSError):
            compat.popen_detached(["cmd"], is_windows=True)
        assert len(calls) == 1

    def test_second_attempt_failure_propagates(self):
        fake, calls = _recording_popen([_access_denied(), _access_denied()])
        with patch.object(subprocess, "Popen", fake), pytest.raises(OSError):
            compat.popen_detached(["cmd"], is_windows=True)
        assert len(calls) == 2

    def test_caller_creationflags_ride_both_attempts(self):
        fake, calls = _recording_popen([_access_denied(), "child"])
        with patch.object(subprocess, "Popen", fake):
            compat.popen_detached(["cmd"], is_windows=True, creationflags=CALLER_FLAGS)
        assert [call[1]["creationflags"] for call in calls] == [
            BREAKAWAY_FLAGS | CALLER_FLAGS, PLAIN_FLAGS | CALLER_FLAGS]

    def test_posix_detaches_with_start_new_session(self):
        fake, calls = _recording_popen(["child"])
        with patch.object(subprocess, "Popen", fake):
            assert compat.popen_detached(["cmd"], is_windows=False) == "child"
        assert calls[0][1] == {"start_new_session": True}


@pytest.mark.platforms("windows")
class TestUpdateSpawnUsesTheHelper:
    """The ``/update`` spawn must go through the retrying helper, not a bare Popen."""

    def test_spawn_detached_update_routes_through_popen_detached(self, tmp_path):
        from gateway.slash_commands import _spawn_detached_update

        recorded = {}

        def fake_popen_detached(args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return object()

        with patch("hermes_cli._subprocess_compat.popen_detached", fake_popen_detached):
            _spawn_detached_update(["hermes"], tmp_path / "out.txt", tmp_path / "rc")

        assert recorded["args"][-2:] == ["update", "--gateway"]
        assert recorded["kwargs"]["stdout"] is subprocess.DEVNULL
        assert recorded["kwargs"]["stderr"] is subprocess.DEVNULL
        assert sys.executable in recorded["args"]

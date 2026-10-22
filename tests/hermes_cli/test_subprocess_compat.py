"""Behavioral coverage for detached subprocess compatibility helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_cli import _subprocess_compat as compat


@pytest.mark.platforms("posix")
def test_detached_popen_starts_a_new_session_on_posix(monkeypatch):
    """Detached callers keep their existing POSIX session-separation behavior."""
    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(compat.subprocess, "Popen", popen)

    result = compat.detached_popen(["hermes", "gateway", "restart"], cwd="/tmp/hermes")

    assert result is popen.return_value
    popen.assert_called_once_with(
        ["hermes", "gateway", "restart"], cwd="/tmp/hermes", start_new_session=True
    )


@pytest.mark.platforms("windows")
def test_detached_popen_retries_access_denied_without_breakaway(monkeypatch):
    """Regression for #137621: a job denying breakaway still starts the child once."""
    calls = []
    child = MagicMock()

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            error = OSError(13, "Access is denied")
            error.winerror = 5
            raise error
        return child

    monkeypatch.setattr(compat.subprocess, "Popen", fake_popen)

    assert compat.detached_popen(["hermes", "gateway", "restart"], cwd="C:/hermes") is child

    assert [kwargs["creationflags"] for _, kwargs in calls] == [
        compat.windows_detach_flags(),
        compat.windows_detach_flags_without_breakaway(),
    ]
    assert [argv for argv, _ in calls] == [["hermes", "gateway", "restart"]] * 2
    assert [kwargs["cwd"] for _, kwargs in calls] == ["C:/hermes"] * 2


@pytest.mark.platforms("windows")
def test_detached_popen_reraises_non_breakaway_failure(monkeypatch):
    """Missing executables and unrelated spawn failures are never retried or masked."""
    popen = MagicMock(side_effect=OSError(2, "No such file"))
    monkeypatch.setattr(compat.subprocess, "Popen", popen)

    with pytest.raises(OSError, match="No such file"):
        compat.detached_popen(["missing-hermes"])

    popen.assert_called_once()

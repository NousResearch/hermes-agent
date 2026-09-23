"""``_ensure_venv_pip``: the venv pip bootstrap stays bounded and never leaves a probe behind.

Regression guard for #120169. The probe used to be a bare ``subprocess.run(..., capture_output=True)``
with no timeout: when it hung in the post-swap deps phase the abandoned ``pip --version`` child kept
the venv interpreter mapped, so every later ``hermes update`` refused on the Windows venv-holder guard
("Other Hermes processes are running from this install's venv") until that PID was killed by hand.
The probe therefore goes through ``bounded_probe_run`` (tree-kill on timeout), a timed-out probe is
skipped instead of answered with an ensurepip install, and the ensurepip fallback is bounded too.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from hermes_cli import update_cmd_deps
from hermes_cli.update_cmd import _m

_PIP = ["C:/hermes/venv/Scripts/python.exe", "-m", "pip"]
_PYTHON = "C:/hermes/venv/Scripts/python.exe"
_ENSUREPIP_TAIL = ["-m", "ensurepip", "--upgrade", "--default-pip"]


class _Result:
    """Minimal ``CompletedProcess`` stand-in: the helper only reads these three fields."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def probe(monkeypatch):
    """Install a scripted ``bounded_probe_run`` and return a call log.

    The script is consumed in call order: an exception instance is raised, ``None`` means the
    bounded runner gave up (timeout / spawn failure), anything else is returned as the result.
    """

    def _install(script):
        calls: list[tuple[list[str], float, dict]] = []

        def fake_bounded_probe_run(argv, *, timeout, **kwargs):
            calls.append((list(argv), timeout, kwargs))
            outcome = script[len(calls) - 1]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(update_cmd_deps, "bounded_probe_run", fake_bounded_probe_run)
        return calls

    return _install


def test_probe_success_does_not_touch_ensurepip(probe):
    calls = probe([_Result(0)])

    update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)

    assert len(calls) == 1
    argv, timeout, kwargs = calls[0]
    assert argv == _PIP + ["--version"]
    assert timeout == update_cmd_deps._VENV_PIP_PROBE_TIMEOUT_SECONDS
    assert kwargs["cwd"] == str(_m().PROJECT_ROOT)


def test_probe_never_uses_bare_subprocess_run(probe, monkeypatch):
    """The regression itself: a bare ``subprocess.run`` probe can hang *and* orphan its child."""

    def _boom(*_args, **_kwargs):  # pragma: no cover - only trips if the probe regresses
        raise AssertionError("the venv pip probe must not use subprocess.run (#120169)")

    monkeypatch.setattr(update_cmd_deps.subprocess, "run", _boom)
    probe([_Result(0)])

    update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)


def test_nonzero_probe_bootstraps_pip(probe):
    calls = probe([_Result(1), _Result(0)])

    update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)

    assert len(calls) == 2
    ensurepip_argv, timeout, _kwargs = calls[1]
    assert ensurepip_argv == [_PYTHON] + _ENSUREPIP_TAIL
    assert timeout == update_cmd_deps._VENV_ENSUREPIP_TIMEOUT_SECONDS


def test_probe_timeout_is_skipped_never_answered_with_ensurepip(probe, caplog):
    """A timed-out probe says nothing about whether pip is installed: don't run a heavier install
    against a venv that just proved it can hang, and don't raise — the probe is advisory."""
    calls = probe([None])

    with caplog.at_level(logging.WARNING, logger="hermes_cli.update_cmd"):
        update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)

    assert len(calls) == 1
    assert "skipping the ensurepip bootstrap" in caplog.text


def test_unspawnable_probe_bootstraps_pip(probe):
    """A spawn failure is the lost-pip case this function exists for — bootstrap it."""
    calls = probe([OSError("cannot execute"), _Result(0)])

    update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)

    assert len(calls) == 2
    assert calls[1][0] == [_PYTHON] + _ENSUREPIP_TAIL


def test_failed_ensurepip_still_raises(probe):
    probe([_Result(1), _Result(2, stderr="no ensurepip here")])

    with pytest.raises(subprocess.CalledProcessError):
        update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)


def test_ensurepip_timeout_raises_instead_of_wedging(probe):
    """The fallback is bounded too: a hung ensurepip aborts the update, it does not hold it open."""
    probe([_Result(1), None])

    with pytest.raises(subprocess.TimeoutExpired):
        update_cmd_deps._ensure_venv_pip(_PIP, _PYTHON)

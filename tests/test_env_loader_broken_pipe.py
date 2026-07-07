"""Tests for stderr-diagnostic resilience in ``hermes_cli.env_loader``.

Cron/daemon processes can end up orphaned (parent died, PPID 1) with no
reader on the far end of their stderr pipe. In that state a plain
``print(..., file=sys.stderr)`` raises ``BrokenPipeError``, which — if
uncaught — aborts whatever triggered it. On 2026-07-07 00:00 this took
down two cron jobs: ``load_hermes_dotenv()`` -> ``_apply_external_secret_sources``
-> a diagnostic ``print()`` -> ``BrokenPipeError`` -> job death.

These tests cover the fix: every stderr diagnostic in this module must be
best-effort and must never propagate a broken-pipe failure into env
loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402


class _OrphanedPipeStderr:
    """Stand-in for an orphaned stderr pipe.

    Matches what a cron/daemon process sees once its parent has died
    (PPID 1) and nothing is reading the far end of the pipe: every
    ``write()`` raises ``BrokenPipeError(32, "Broken pipe")``, exactly as
    in the 2026-07-07 00:00 scheduler traceback.
    """

    def write(self, message: str) -> int:  # noqa: ARG002 - signature match
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_warned_keys():
    """Each test starts with a clean warned-keys set.

    ``_sanitize_loaded_credentials`` only warns once per key per process
    (``_WARNED_KEYS``); without resetting, a test run earlier in the
    session could suppress the warning this test relies on triggering.
    """
    env_loader._WARNED_KEYS.clear()
    yield
    env_loader._WARNED_KEYS.clear()


def test_stderr_note_swallows_broken_pipe(monkeypatch):
    """``_stderr_note`` must not raise when stderr's pipe is orphaned."""
    monkeypatch.setattr(sys, "stderr", _OrphanedPipeStderr())
    # Must not raise BrokenPipeError.
    env_loader._stderr_note("x")


def test_stderr_note_writes_message_on_working_stderr(capsys):
    """On a healthy stderr, the message still lands as before."""
    env_loader._stderr_note("hello from hermes")
    captured = capsys.readouterr()
    assert captured.err == "hello from hermes\n"


def test_sanitize_loaded_credentials_survives_broken_pipe_stderr(monkeypatch):
    """Real failure path: the non-ASCII credential warning must not take
    down credential sanitization when stderr is an orphaned pipe.

    This reproduces the 2026-07-07 00:00 cron crash: a credential env var
    contains a Unicode lookalike character (e.g. copy-pasted from a PDF),
    which triggers the sanitize-pass warning print — and that print used
    to raise BrokenPipeError and kill the caller.
    """
    key = "TEST_BROKEN_PIPE_API_KEY"
    # 'ʋ' (U+028B, LATIN SMALL LETTER V WITH HOOK) is a classic lookalike
    # substitution for ASCII 'v' — see _sanitize_loaded_credentials docstring.
    dirty_value = "sk-lookalikeʋvalue-123"
    monkeypatch.setenv(key, dirty_value)
    monkeypatch.setattr(sys, "stderr", _OrphanedPipeStderr())

    # Must not raise BrokenPipeError.
    env_loader._sanitize_loaded_credentials()

    # The credential must still get sanitized even though the warning
    # print couldn't reach a reader — diagnostics must never block the
    # actual work.
    import os

    cleaned = os.environ[key]
    cleaned.encode("ascii")  # must not raise
    assert "ʋ" not in cleaned
    assert cleaned == "sk-lookalikevalue-123"

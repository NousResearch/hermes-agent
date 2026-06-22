"""Regression tests for the npm stderr capture + error log persistence fix.

Each test covers a specific failure vector introduced by the Risk 3 solution:

1. _install_sidecar() return code unchanged — still 0 on success, non-zero on failure
2. _install_sidecar() with no npm on PATH — still returns 1, no OSError on log write
3. _NPM_ERROR_LOG write fails (OSError / read-only fs) — silently handled, no exception
4. _NPM_ERROR_LOG read fails in check_requirements() — silently handled, returns False
5. _NPM_ERROR_LOG is empty string — not written, check_requirements() falls back gracefully
6. _NPM_ERROR_LOG from prior failed run exists when next run succeeds — cleared
7. check_requirements() with no _NPM_ERROR_LOG — debug log still emitted without error detail
8. proc.stderr is None (edge case on some platforms) — no AttributeError, no crash
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from plugins.platforms.photon import adapter as adapter_mod
from plugins.platforms.photon import cli as cli_mod

_NODE_ON_PATH = __import__("shutil").which("node") is not None
_requires_node = pytest.mark.skipif(
    not _NODE_ON_PATH, reason="requires node on PATH"
)


# ---------------------------------------------------------------------------
# 1. Return code contract unchanged
# ---------------------------------------------------------------------------

def test_regression_return_code_zero_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_install_sidecar() must still return 0 on npm success."""
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", tmp_path / ".photon-npm-error.log")
    assert cli_mod._install_sidecar() == 0


def test_regression_return_code_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_install_sidecar() must still propagate a non-zero npm exit code."""
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stderr="npm ERR! fail"),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", tmp_path / ".photon-npm-error.log")
    assert cli_mod._install_sidecar() == 1


def test_regression_return_code_when_npm_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_install_sidecar() must still return 1 when npm is not on PATH."""
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: None)
    assert cli_mod._install_sidecar() == 1


# ---------------------------------------------------------------------------
# 2. OSError on log write — silently swallowed, no crash
# ---------------------------------------------------------------------------

def test_regression_oserror_on_log_write_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If writing _NPM_ERROR_LOG raises OSError (read-only fs, permission denied),
    _install_sidecar() must NOT propagate the exception — it still returns the
    npm exit code."""
    def _bad_log_write(*args, **kwargs):
        raise OSError("read-only file system")

    error_log = tmp_path / ".photon-npm-error.log"
    # Monkey-patch write_text on the Path object via a subclass
    class _UnwritablePath(type(error_log)):
        def write_text(self, *a, **kw):
            raise OSError("read-only file system")
        def unlink(self, *a, **kw):
            raise OSError("read-only file system")
        def exists(self):
            return False

    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stderr="npm ERR!"),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", _UnwritablePath(error_log))

    rc = cli_mod._install_sidecar()
    assert rc == 1  # still returns the npm exit code


# ---------------------------------------------------------------------------
# 3. OSError on log read in check_requirements() — silently swallowed
# ---------------------------------------------------------------------------

@_requires_node
def test_regression_oserror_on_log_read_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If reading _NPM_ERROR_LOG raises OSError, check_requirements() must NOT
    propagate the exception — it returns False and emits the fallback debug log."""
    class _UnreadablePath(type(tmp_path)):
        def exists(self):
            return True
        def read_text(self, *a, **kw):
            raise OSError("permission denied")

    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(adapter_mod, "_NPM_ERROR_LOG", _UnreadablePath(tmp_path / ".err"))

    result = adapter_mod.check_requirements()
    assert result is False  # must not raise


# ---------------------------------------------------------------------------
# 4. Empty stderr — log file NOT written
# ---------------------------------------------------------------------------

def test_regression_empty_stderr_does_not_write_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If npm fails but stderr is empty (some npm versions), _NPM_ERROR_LOG must
    NOT be written — an empty file would mislead check_requirements()."""
    error_log = tmp_path / ".photon-npm-error.log"
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stderr=""),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", error_log)

    cli_mod._install_sidecar()

    assert not error_log.exists(), (
        "_NPM_ERROR_LOG must not be created when stderr is empty"
    )


# ---------------------------------------------------------------------------
# 5. proc.stderr is None — no AttributeError
# ---------------------------------------------------------------------------

def test_regression_none_stderr_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On some platforms/configurations proc.stderr can be None even with
    stderr=PIPE (e.g. encoding errors).  _install_sidecar() must handle this."""
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stderr=None),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", tmp_path / ".photon-npm-error.log")

    rc = cli_mod._install_sidecar()
    assert rc == 1  # must not raise AttributeError


# ---------------------------------------------------------------------------
# 6. Stale log cleared on success — no phantom errors after reinstall
# ---------------------------------------------------------------------------

def test_regression_stale_log_not_surfaced_after_successful_reinstall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If npm install succeeds on a retry but a stale _NPM_ERROR_LOG from the
    prior failed run still exists, check_requirements() must NOT surface the
    stale error after the successful reinstall clears it."""
    error_log = tmp_path / ".photon-npm-error.log"
    error_log.write_text("stale: npm ERR! old failure", encoding="utf-8")

    # Successful reinstall clears the log
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", error_log)
    cli_mod._install_sidecar()
    assert not error_log.exists(), "Success must clear the stale error log"

    # Now check_requirements() must not mention the old error
    # Create spectrum-ts inside node_modules/ — the content check requires it.
    (tmp_path / "node_modules" / "spectrum-ts").mkdir(parents=True)
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(adapter_mod, "_NPM_ERROR_LOG", error_log)

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.photon.adapter"):
        result = adapter_mod.check_requirements()

    assert result is True
    assert not any("stale" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. check_requirements() without error log — debug log still emitted
# ---------------------------------------------------------------------------

@_requires_node
def test_regression_debug_log_emitted_even_without_error_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When node_modules is missing and no _NPM_ERROR_LOG exists (first-time
    setup, not a failed install), check_requirements() must still emit a DEBUG
    line pointing to the sidecar path."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(adapter_mod, "_NPM_ERROR_LOG", tmp_path / ".photon-npm-error.log")
    # node_modules NOT created, error log NOT created

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.photon.adapter"):
        result = adapter_mod.check_requirements()

    assert result is False
    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(str(tmp_path) in m for m in debug_messages), (
        f"Expected DEBUG with sidecar path even without error log, got: {debug_messages}"
    )

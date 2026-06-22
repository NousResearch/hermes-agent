"""Tests for check_requirements() diagnostic logging (fix) and remaining risks.

Fixed in this file (tests PASS with fix, FAIL without):
  - check_requirements() now emits a specific logger.warning for each False
    condition so gateway logs pinpoint the exact failure reason.

Remaining risks documented here (still open — separate issues):
  Risk 2 – node_modules dir exists but EMPTY (partial/aborted npm install)
            → check_requirements() returns True (false positive)
  Risk 3 – _install_sidecar() subprocess.run calls carry no capture_output /
            stdout / stderr — npm error output is unrecoverable after the run
"""
from __future__ import annotations

import logging
import shutil
import types
from pathlib import Path

import pytest

from plugins.platforms.photon import adapter as adapter_mod
from plugins.platforms.photon import cli as cli_mod


# ---------------------------------------------------------------------------
# Helpers / shared marks
# ---------------------------------------------------------------------------

_NODE_ON_PATH = shutil.which("node") is not None

_requires_node = pytest.mark.skipif(
    not _NODE_ON_PATH,
    reason="requires node on PATH to isolate the node_modules check",
)

_requires_node_for_false_positive = pytest.mark.skipif(
    not _NODE_ON_PATH,
    reason="requires node on PATH so the false-positive path is reachable",
)


# ---------------------------------------------------------------------------
# Fix verification — each False branch now emits a specific warning
# ---------------------------------------------------------------------------


def test_fix_logs_warning_when_httpx_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When httpx is not installed, check_requirements() must log a warning
    that names the missing package so the operator knows what to install."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", False)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    (tmp_path / "node_modules").mkdir()

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.photon.adapter"):
        result = adapter_mod.check_requirements()

    assert result is False
    messages = [r.message for r in caplog.records]
    assert any("httpx" in m for m in messages), (
        f"Expected a warning mentioning 'httpx', got: {messages}"
    )


@_requires_node
def test_fix_logs_warning_when_node_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the node binary is not found, check_requirements() must log a
    warning that names the binary so the operator can diagnose PATH issues."""
    fake_bin = str(tmp_path / "_no_such_node_xyz")
    monkeypatch.setenv("PHOTON_NODE_BIN", fake_bin)
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    (tmp_path / "node_modules").mkdir()

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.photon.adapter"):
        result = adapter_mod.check_requirements()

    assert result is False
    messages = [r.message for r in caplog.records]
    assert any("node" in m.lower() for m in messages), (
        f"Expected a warning mentioning the node binary, got: {messages}"
    )


@_requires_node
def test_fix_logs_debug_with_sidecar_path_when_node_modules_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When node_modules is absent, check_requirements() must log at DEBUG
    (not WARNING) with the sidecar path and corrective command.

    DEBUG is intentional: absent node_modules is the normal pre-setup state.
    check_fn() is called from multiple hot paths in the core
    (load_gateway_config, hermes status, GET /api/status desktop polling) —
    WARNING here would spam logs on every probe when photon is not configured."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    # node_modules intentionally NOT created — simulates failed npm install

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.photon.adapter"):
        result = adapter_mod.check_requirements()

    assert result is False

    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]

    # Must emit a DEBUG line with the path and corrective command
    assert any(str(tmp_path) in m for m in debug_messages), (
        f"Expected DEBUG containing sidecar path '{tmp_path}', got debug={debug_messages}"
    )
    assert any("setup" in m.lower() for m in debug_messages), (
        f"Expected DEBUG mentioning 'setup', got debug={debug_messages}"
    )

    # Must NOT emit a WARNING — that would spam logs on every /api/status probe
    assert warning_messages == [], (
        f"Expected zero WARNING records for expected pre-setup state, "
        f"got: {warning_messages}"
    )


# ---------------------------------------------------------------------------
# Risk 2 (open) — empty node_modules directory is a false positive
# ---------------------------------------------------------------------------


@_requires_node_for_false_positive
def test_risk2_fix_empty_node_modules_no_longer_passes_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """npm may create node_modules/ before aborting (network timeout, ENOSPC,
    EACCES).  Previously an empty directory passed the only filesystem guard in
    check_requirements() — returning True with a broken sidecar installation.
    Fixed: check_requirements() now verifies node_modules/spectrum-ts exists,
    so a partial/empty node_modules/ correctly returns False."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(adapter_mod, "_NPM_ERROR_LOG", tmp_path / ".photon-npm-error.log")
    (tmp_path / "node_modules").mkdir()  # empty — spectrum-ts absent

    # Fix verified: False instead of the old false-positive True.
    assert adapter_mod.check_requirements() is False


# ---------------------------------------------------------------------------
# Risk 3 fix — npm stderr is captured, persisted, and surfaced by check_requirements
# ---------------------------------------------------------------------------


def test_fix_risk3_npm_stderr_persisted_to_error_log_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When npm fails, _install_sidecar() must write the captured stderr to
    _NPM_ERROR_LOG so check_requirements() can surface the root cause later."""
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    error_log = sidecar_dir / ".photon-npm-error.log"

    calls: list[dict] = []

    def _fake_run(cmd: list, **kwargs: object) -> types.SimpleNamespace:
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return types.SimpleNamespace(
            returncode=1,
            stderr="npm ERR! ETIMEDOUT fetch failed\nnpm ERR! network timeout",
        )

    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", error_log)

    cli_mod._install_sidecar()

    # npm stderr must be captured (stderr= kwarg present in subprocess.run calls)
    assert all("stderr" in c["kwargs"] for c in calls), (
        "subprocess.run calls must use stderr= to capture npm error output"
    )
    # Error must be persisted to the log file
    assert error_log.exists(), "_NPM_ERROR_LOG must be written on npm failure"
    content = error_log.read_text(encoding="utf-8")
    assert "ETIMEDOUT" in content, (
        f"Expected npm error in log file, got: {content!r}"
    )


def test_fix_risk3_error_log_cleared_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale error log from a previous failed install must be deleted when
    npm install succeeds, so check_requirements() doesn't report a stale error."""
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir()
    error_log = sidecar_dir / ".photon-npm-error.log"
    error_log.write_text("stale npm error from last run", encoding="utf-8")

    monkeypatch.setattr(cli_mod.shutil, "which", lambda _: "/usr/bin/npm")
    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(cli_mod, "_NPM_ERROR_LOG", error_log)

    cli_mod._install_sidecar()

    assert not error_log.exists(), (
        "_NPM_ERROR_LOG must be deleted after a successful npm install"
    )


@_requires_node
def test_fix_risk3_check_requirements_surfaces_npm_error_in_debug_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When node_modules is missing and _NPM_ERROR_LOG exists, check_requirements()
    must include the npm error detail in the DEBUG log so the root cause is
    visible when debug logging is enabled."""
    error_log = tmp_path / ".photon-npm-error.log"
    error_log.write_text("npm ERR! ENOSPC: no space left on device", encoding="utf-8")

    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(adapter_mod, "_NPM_ERROR_LOG", error_log)
    # node_modules NOT created

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.photon.adapter"):
        result = adapter_mod.check_requirements()

    assert result is False
    debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("ENOSPC" in m for m in debug_messages), (
        f"Expected npm error 'ENOSPC' in DEBUG log, got: {debug_messages}"
    )

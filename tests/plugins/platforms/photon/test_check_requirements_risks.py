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
    # NS-606: disable the connect-time self-heal branch so the guard itself
    # (empty node_modules must not read as installed) is what's under test.
    monkeypatch.setattr(adapter_mod, "_dir_writable", lambda _p: False)
    (tmp_path / "node_modules").mkdir()  # empty — spectrum-ts absent

    # Fix verified: False instead of the old false-positive True.
    assert adapter_mod.check_requirements() is False


# ---------------------------------------------------------------------------
# Risk 3 fix — npm stderr is captured, persisted, and surfaced by check_requirements
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared predicate — status / _start_sidecar / check_requirements must agree
# ---------------------------------------------------------------------------


def test_cli_status_shares_adapter_sidecar_deps_check(tmp_path: Path) -> None:
    """`hermes photon status` must use the exact same spectrum-ts check as
    check_requirements() / _start_sidecar() — not a separate node_modules-only
    existence check that would disagree on a partial/empty install."""
    assert cli_mod.sidecar_deps_installed is adapter_mod.sidecar_deps_installed


def test_sidecar_deps_installed_false_on_empty_node_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    (tmp_path / "node_modules").mkdir()  # empty — spectrum-ts absent
    assert adapter_mod.sidecar_deps_installed() is False



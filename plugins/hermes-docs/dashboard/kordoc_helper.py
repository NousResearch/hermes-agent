"""Kordoc detection and conversion preview for Hermes Docs.

Exposes two public functions consumed by plugin_api.py routes:

- ``detect_kordoc()``  — reports whether kordoc is callable on this machine.
- ``preview_conversion()`` — converts a workspace file without mutating source.

Both functions honour module-level override hooks so tests can run without a
real Node.js / kordoc installation.  Production code leaves the overrides as
``None`` and the real subprocess paths are taken.

This module intentionally has no imports from the Hermes application layer so
it can be loaded standalone (via importlib) by plugin_api.py.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test seams
# ---------------------------------------------------------------------------

# When set to a dict, detect_kordoc() returns it directly without calling
# subprocess.  Shape: {"available": bool, "version": str|None, "detail": str}
_detect_override: Optional[dict] = None

# When set to a callable, preview_conversion() uses it instead of
# subprocess.run for the conversion call.
# Signature: _subprocess_run_override(cmd, *, capture_output, text, timeout, check)
_subprocess_run_override = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Formats kordoc can convert (MCP allowlist + CLI extras)
SUPPORTED_EXTENSIONS = frozenset({".hwp", ".hwpx", ".pdf", ".xlsx", ".docx", ".xls"})

# Target formats the preview endpoint accepts
SUPPORTED_TARGET_FORMATS = frozenset({"markdown", "json"})


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_kordoc() -> dict:
    """Check whether kordoc is available on this machine.

    Uses only already-installed local executables.  It never shells out through
    ``npx`` or another package manager that could contact a registry.

    Returns:
        dict with keys ``available`` (bool), ``version`` (str | None),
        ``detail`` (str).
    """
    if _detect_override is not None:
        return dict(_detect_override)

    executable = _find_kordoc_executable()
    if not executable:
        return {
            "available": False,
            "version": None,
            "detail": "kordoc executable not found on PATH",
        }

    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:  # pragma: no cover — OS / timeout edge
        return {
            "available": False,
            "version": None,
            "detail": str(exc)[:200],
        }

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:200] or "kordoc exited non-zero"
        return {"available": False, "version": None, "detail": detail}

    version_line = result.stdout.strip().splitlines()
    version = version_line[-1].strip() if version_line else None
    return {"available": True, "version": version, "detail": "kordoc available"}


def _find_kordoc_executable() -> str | None:
    """Return an installed local Kordoc command path, if one exists."""
    configured = os.environ.get("HERMES_DOCS_KORDOC", "").strip()
    if configured and Path(configured).exists():
        return configured
    for name in ("kordoc", "kordoc-mcp"):
        found = shutil.which(name)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# Preview conversion
# ---------------------------------------------------------------------------


def preview_conversion(workspace_base: Path, rel: str, target_format: str) -> dict:
    """Convert a workspace file to *target_format* without mutating the source.

    Path traversal outside *workspace_base* raises ``fastapi.HTTPException(403)``.
    Unsupported *target_format* raises ``HTTPException(400)``.
    Missing file raises ``HTTPException(404)``.
    When kordoc is unavailable the function returns a structured stub (never
    raises a 5xx error for an availability gap).

    Args:
        workspace_base: Absolute path to the workspace source folder.
        rel: Relative path to the target file within the workspace.
        target_format: ``"markdown"`` or ``"json"``.

    Returns:
        dict with keys:
          - ``available`` (bool) — whether kordoc is callable
          - ``rel`` (str) — echoed input path
          - ``target_format`` (str) — echoed format
          - ``content`` (str | None) — converted output; ``None`` when unavailable
          - ``warnings`` (list[str]) — any warnings from kordoc
          - ``message`` (str) — human-readable status
    """
    # Local import keeps top-level import-time deps to stdlib only.
    from fastapi import HTTPException  # noqa: PLC0415

    if target_format not in SUPPORTED_TARGET_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported target format: {target_format!r}. "
                f"Accepted: {sorted(SUPPORTED_TARGET_FORMATS)}"
            ),
        )

    # --- Path traversal guard (mirrors _safe_relative_target in plugin_api) ---
    resolved_base = workspace_base.resolve()
    target = (resolved_base / rel).resolve()
    try:
        target.relative_to(resolved_base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal blocked")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found in workspace")

    # --- Availability check ---
    status = detect_kordoc()
    if not status["available"]:
        return {
            "available": False,
            "rel": rel,
            "target_format": target_format,
            "content": None,
            "warnings": [],
            "message": f"kordoc unavailable: {status['detail']}",
        }

    # --- Conversion — source file is never written ---
    executable = _find_kordoc_executable()
    if not executable:
        if _subprocess_run_override is None:
            return {
                "available": False,
                "rel": rel,
                "target_format": target_format,
                "content": None,
                "warnings": [],
                "message": "kordoc unavailable: executable not found",
            }
        executable = "kordoc"

    cmd = [executable, str(target)]
    if target_format == "json":
        cmd += ["--format", "json"]

    try:
        if _subprocess_run_override is not None:
            result = _subprocess_run_override(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
    except Exception as exc:  # pragma: no cover
        return {
            "available": True,
            "rel": rel,
            "target_format": target_format,
            "content": None,
            "warnings": [],
            "message": f"conversion failed: {exc}",
        }

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[:400]
        return {
            "available": True,
            "rel": rel,
            "target_format": target_format,
            "content": None,
            "warnings": [],
            "message": f"kordoc exit {result.returncode}: {err}",
        }

    return {
        "available": True,
        "rel": rel,
        "target_format": target_format,
        "content": result.stdout,
        "warnings": [],
        "message": "ok",
    }

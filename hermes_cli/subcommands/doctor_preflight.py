"""``hermes doctor --preflight`` — pre-activation checks for the updater's
slot activation gate.

The Rust updater runs this against a STAGED slot before committing the atomic
``current.txt`` flip. If preflight fails, the staging dir is deleted and the
current slot is left untouched.

Checks (from ``docs/plans/updater-rework/02-phase1-updater.md`` task 1.5):
1. **Core imports** — ``run_agent``, ``model_tools``, ``gateway.run``,
   ``hermes_cli.main`` import cleanly in a subprocess.
2. **Config parses** — ``load_config()`` does not raise.
3. **Config version migratable** — ``check_config_version()`` current ≤ latest.
4. **Artifact roots resolve** — ``get_artifact_root()`` succeeds and each
   accessor (``bundled_skills_dir()``, ``web_dist_dir()``, ``tui_dist_dir()``)
   points at an existing, non-empty directory, skipping any the manifest flags
   absent (e.g. ``"desktop": false``).

Design notes:
- The preflight is crash-proof: a broken venv or missing path is REPORTED, not
  raised. The function returns ``(False, report_dict)`` on failure, never raises.
- Core imports run in a subprocess to avoid polluting the current process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def preflight() -> tuple[bool, dict]:
    """Run pre-activation checks. Returns ``(ok, report)``.

    The report dict has a top-level ``ok`` boolean and one sub-dict per check:
    ``core_imports``, ``config_parse``, ``config_version``, ``artifact_roots``.

    This function never raises — all failures are captured in the report.
    """
    report: dict[str, Any] = {"ok": True}
    ok = True

    # ------------------------------------------------------------------
    # Check 1: Core imports (subprocess — don't pollute this process)
    # ------------------------------------------------------------------
    report["core_imports"] = _check_core_imports()
    if not report["core_imports"]["ok"]:
        ok = False

    # ------------------------------------------------------------------
    # Check 2: Config parses
    # ------------------------------------------------------------------
    report["config_parse"] = _check_config_parse()
    if not report["config_parse"]["ok"]:
        ok = False

    # ------------------------------------------------------------------
    # Check 3: Config version migratable
    # ------------------------------------------------------------------
    report["config_version"] = _check_config_version()
    if not report["config_version"]["ok"]:
        ok = False

    # ------------------------------------------------------------------
    # Check 4: Artifact roots resolve
    # ------------------------------------------------------------------
    report["artifact_roots"] = _check_artifact_roots()
    if not report["artifact_roots"]["ok"]:
        ok = False

    report["ok"] = ok
    return ok, report


# -- Individual checks --------------------------------------------------------


def _check_core_imports() -> dict[str, Any]:
    """Import core modules in a subprocess to avoid polluting this process.

    Runs ``<python> -c "import run_agent, model_tools, gateway.run, hermes_cli.main"``
    and checks the exit code.
    """
    result: dict[str, Any] = {"ok": True, "check": "core_imports"}
    py = sys.executable or "python3"
    try:
        proc = subprocess.run(
            [py, "-c", "import run_agent, model_tools, gateway.run, hermes_cli.main"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            result["ok"] = False
            result["error"] = (proc.stderr or proc.stdout or "").strip()
            result["returncode"] = proc.returncode
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["python"] = py
    return result


def _check_config_parse() -> dict[str, Any]:
    """``load_config()`` must not raise."""
    result: dict[str, Any] = {"ok": True, "check": "config_parse"}
    try:
        from hermes_cli.config import load_config

        load_config()
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _check_config_version() -> dict[str, Any]:
    """``check_config_version()`` current ≤ latest."""
    result: dict[str, Any] = {"ok": True, "check": "config_version"}
    try:
        from hermes_cli.config import check_config_version

        current, latest = check_config_version()
        result["current"] = current
        result["latest"] = latest
        if current > latest:
            result["ok"] = False
            result["error"] = f"config version {current} > latest {latest}"
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _check_artifact_roots() -> dict[str, Any]:
    """Artifact root + accessors resolve to existing, non-empty directories.

    Skips accessors the manifest flags absent (e.g. ``"desktop": false``).
    In a checkout (no manifest), all accessors are checked.
    """
    result: dict[str, Any] = {
        "ok": True,
        "check": "artifact_roots",
        "roots": {},
    }

    try:
        from hermes_constants import (
            get_artifact_root,
            bundled_skills_dir,
            web_dist_dir,
            tui_dist_dir,
        )

        root = get_artifact_root()
        result["root"] = str(root)
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"get_artifact_root: {type(exc).__name__}: {exc}"
        return result

    # Determine which components to skip based on the manifest.
    # The manifest lives at the artifact root in a slot; in a checkout there
    # is no manifest so all accessors are checked.
    skip: set[str] = set()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("desktop") is False:
                skip.add("tui_dist_dir")
        except Exception:
            # If we can't read the manifest, check everything — better safe.
            pass

    # Each accessor: (name, callable, human label)
    accessors = [
        ("bundled_skills_dir", bundled_skills_dir, "bundled_skills_dir"),
        ("web_dist_dir", web_dist_dir, "web_dist_dir"),
        ("tui_dist_dir", tui_dist_dir, "tui_dist_dir"),
    ]

    for name, fn, label in accessors:
        if name in skip:
            result["roots"][label] = {"skipped": True}
            continue
        try:
            path = fn()
            entry: dict[str, Any] = {"path": str(path)}
            if not path.exists():
                result["ok"] = False
                entry["ok"] = False
                entry["error"] = "directory does not exist"
            elif not path.is_dir():
                result["ok"] = False
                entry["ok"] = False
                entry["error"] = "path is not a directory"
            else:
                # Check non-empty
                try:
                    has_content = any(path.iterdir())
                except Exception:
                    has_content = False
                if not has_content:
                    result["ok"] = False
                    entry["ok"] = False
                    entry["error"] = "directory is empty"
                else:
                    entry["ok"] = True
            result["roots"][label] = entry
        except Exception as exc:
            result["ok"] = False
            entry = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            result["roots"][label] = entry

    return result


# -- CLI entrypoint -----------------------------------------------------------


def run_preflight_cli() -> int:
    """Run preflight and print the report as JSON.

    Returns exit code 0 if ok=True, 1 if ok=False.
    """
    ok, report = preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1

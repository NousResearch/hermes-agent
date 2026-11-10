"""The Windows gateway's venv overlay: make detached Windows runs see the venv packages.

Lives beside ``gateway.run`` (which re-exports it): the overlay is a self-contained
Windows bootstrap step, patched before MCP discovery so tool injection does not depend
on launchers preserving PYTHONPATH.
"""
from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_windows_gateway_venv_imports() -> None:
    """Make detached Windows gateway runs see the Hermes venv packages.

    Patched before MCP discovery so tool injection does not depend on launchers preserving PYTHONPATH."""
    if sys.platform != "win32":
        return

    project_root = Path(__file__).resolve().parents[1]
    from pm.environments import committed_venv, venv_python_version

    # A PM install's store Python was already activated onto the committed generation by
    # hermes_bootstrap; overlaying the leftover pre-PM venv loads a foreign ABI (#122183).
    if committed_venv(project_root) is not None:
        return
    candidates = [Path(v) for v in (os.environ.get("VIRTUAL_ENV"),) if v] + [project_root / "venv"]
    # VIRTUAL_ENV is environment input, not the deployment interpreter's own venv: each
    # candidate's site directory is judged against the image before addsitedir processes
    # it (.pth files execute), keeping an image venv working and refusing anything else.
    from hermes_cli.hosted_code_fence import hosted_code_refusal

    seen: set[str] = set()
    for venv_dir in candidates:
        try:
            resolved_venv = venv_dir.resolve()
        except OSError:
            resolved_venv = venv_dir
        venv_key = str(resolved_venv).lower()
        if venv_key in seen:
            continue
        seen.add(venv_key)

        site_packages = resolved_venv / "Lib" / "site-packages"
        if not site_packages.exists():
            continue
        # Built for another (or an unreadable) Python: its compiled modules crash this one (#139085).
        if (found := venv_python_version(resolved_venv)) != sys.version_info[:2]:
            logger.debug("Skipping venv overlay %s: Python %s, gateway %s", resolved_venv, found or "unknown", sys.version_info[:2])
            continue
        # Guard: a refused candidate is skipped, never processed — its .pth must not run.
        if hosted_code_refusal(site_packages):
            logger.debug("Skipping venv overlay %s: outside the deployment image", resolved_venv)
            continue
        project_entry = str(project_root)
        site_entry = str(site_packages)
        if project_entry not in sys.path:
            sys.path.insert(0, project_entry)
        # addsitedir semantics matter: pywin32 (MCP SDK on Windows) needs .pth processing for pywintypes.
        site.addsitedir(site_entry)
        if site_entry in sys.path:
            sys.path.remove(site_entry)
        insert_at = 1 if sys.path and sys.path[0] == project_entry else 0
        sys.path.insert(insert_at, site_entry)

        os.environ["VIRTUAL_ENV"] = str(resolved_venv)
        return

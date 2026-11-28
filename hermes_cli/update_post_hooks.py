"""Operator overlays inside the venv that must survive ``hermes update`` (#115667).

A dependency sync rewrites site-packages with no trace, so a patched third-party
file silently reverts to its installed version. ``updates.post_hooks`` declares
``{path, marker, command?}`` entries; after the update's tail each file must still
contain its marker, and a declared ``command`` re-applies it when the marker is gone.

Never raises: a bad hook declaration or an unreadable file must not fail an update.
"""

from __future__ import annotations

import logging
import subprocess
from contextlib import suppress
from pathlib import Path

from hermes_constants import project_venv_dir

logger = logging.getLogger(__name__)


def _update_post_hooks_config() -> list:
    """Normalized ``updates.post_hooks`` entries: ``{path, marker, command?}`` dicts.

    Operators declare site-packages files that must still contain a marker string
    after the reinstall (e.g. a patched third-party transport). Malformed config
    degrades to ``[]`` — the update must never break on a bad hook declaration.
    See #115667.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("Update post-hooks skipped (config load failed): %s", exc, exc_info=True)
        return []
    section = cfg.get("updates") if isinstance(cfg, dict) else None
    hooks = section.get("post_hooks") if isinstance(section, dict) else None
    if not isinstance(hooks, list):
        return []
    normalized = []
    for entry in hooks:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        marker = entry.get("marker")
        if not path or not marker:
            continue
        command = entry.get("command") or ""
        normalized.append({"path": str(path), "marker": str(marker), "command": str(command)})
    return normalized


def _resolve_update_post_hook_path(raw: str, project_root) -> Path:
    """Resolve a hook path: absolute as-is, else first hit under the project venv's
    site-packages dirs (POSIX ``lib/python*/site-packages`` + Windows ``Lib/site-packages``),
    then the project root; falls back to the first site-packages candidate so logs name
    the expected location. See #115667."""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    roots: list = []
    try:
        venv_root = project_venv_dir(project_root) if project_root is not None else None
    except Exception:  # health: allow BLE001 -- a path probe: no venv root just means no overlay roots
        venv_root = None
    if venv_root is not None:
        with suppress(Exception):
            roots.extend(sorted(Path(venv_root).glob("lib/python*/site-packages")))
        with suppress(Exception):
            win_sp = Path(venv_root) / "Lib" / "site-packages"
            if win_sp.is_dir():
                roots.append(win_sp)
    for base in roots:
        hit = base / raw
        with suppress(Exception):
            if hit.is_file():
                return hit
    if project_root is not None:
        with suppress(Exception):
            hit = Path(project_root) / raw
            if hit.is_file():
                return hit
    if roots:
        return roots[0] / raw
    return Path(project_root) / raw if project_root is not None else Path(raw)


def _verify_update_post_hooks(*, project_root=None) -> list:
    """Post-sync verifier for ``updates.post_hooks`` (path + marker pairs). Re-checks each
    marker against the FINAL venv state, re-runs the hook's ``command`` when declared and the
    marker is missing (a reinstall wipes overlays mid-update, after earlier phases may have
    reported OK), and logs every outcome plus a summary count. Never raises. See #115667."""
    try:
        if project_root is None:
            from hermes_cli.update_cmd import _m
            project_root = _m().PROJECT_ROOT
    except Exception as exc:
        logger.debug("Update post-hooks skipped (project root unavailable): %s", exc, exc_info=True)
        return []
    hooks = _update_post_hooks_config()
    if not hooks:
        return []

    def _marker_present(target: Path, marker: str) -> bool:
        try:
            return target.is_file() and marker in target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    results = []
    for hook in hooks:
        target = _resolve_update_post_hook_path(hook["path"], project_root)
        if _marker_present(target, hook["marker"]):
            print(f"  \u2713 Post-update hook verified: {target} (marker present)")
            results.append({"path": str(target), "marker": hook["marker"], "status": "verified"})
            continue
        if hook["command"]:
            print(f"  \u2192 Post-update hook marker missing, re-applying: {target}")
            try:
                subprocess.run(
                    hook["command"], shell=True, cwd=project_root, timeout=120, check=False,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            except Exception as exc:  # health: allow BLE001 -- a failed hook is reported as a warning, never raised
                print(f"  \u26a0 Post-update hook command failed for {target}: {exc}")
            if _marker_present(target, hook["marker"]):
                print(f"  \u2713 Post-update hook re-applied: {target}")
                results.append({"path": str(target), "marker": hook["marker"], "status": "reapplied"})
                continue
        print(f"  \u2717 Post-update hook FAILED: {target} (marker {hook['marker']!r} missing — "
              "the reinstall may have wiped an operator overlay)")
        results.append({"path": str(target), "marker": hook["marker"], "status": "failed"})
    verified = sum(1 for r in results if r["status"] == "verified")
    reapplied = sum(1 for r in results if r["status"] == "reapplied")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\u2192 Post-update hooks: {verified} verified, {reapplied} re-applied, {failed} failed")
    return results

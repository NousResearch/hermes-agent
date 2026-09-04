"""Lazy bootstrapper for non-Python runtime deps, routed through pm.

The old shape spawned install.sh/install.ps1 with an --ensure flag; the
rewritten bootstraps no longer install tools ("heavy deps are pm's job"),
so this module is now a thin adapter: each dep maps to the pm packages
that provide it, pm's lazy-install policy decides whether installing is
allowed right now, and pm's InstallError carries the remedy when not.

Deps that degrade gracefully (ffmpeg → skip conversion) are not wired
here — only hard-fail sites call ensure_dependency (TUI needs node,
browser tools need the browser stack).
"""
from __future__ import annotations

import shutil

from hermes_constants import find_node_executable

# dep name -> (availability check, pm packages that provide it)
_DEPS = {
    "node": (lambda: find_node_executable("node") is not None, ("node",)),
    "browser": (lambda: _browser_available(), ("agent-browser", "chromium")),
    "ripgrep": (lambda: shutil.which("rg") is not None, ("ripgrep",)),
}


def _browser_available() -> bool:
    from hermes_constants import agent_browser_runnable

    if agent_browser_runnable(shutil.which("agent-browser")):
        return True
    try:
        import pm

        return pm.is_installed("agent-browser")
    except Exception:
        return False


def ensure_dependency(dep: str, interactive: bool = True) -> bool:
    """Ensure a non-Python dependency is available. Returns True if available."""
    entry = _DEPS.get(dep)
    if entry is None:
        return False
    check, packages = entry
    if check():
        return True

    try:
        import pm

        for name in packages:
            pm.ensure(name)
    except Exception as exc:
        if interactive:
            print(f"  {exc}")
        return False
    return check()


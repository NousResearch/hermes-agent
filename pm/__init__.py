"""pm: the hermes package system.

Everything hermes depends on — tool binaries, the python venv, node_modules
dirs, plugins — is a package in one dependency tree. Package definitions
(pm/packages.py) say what a package IS. The lockfile (pm/lock.json,
machine-written) says exactly which versions and hashes. The installed-state
file (facts.json, per install) says what is actually on this machine.

ensure(name) makes the installed state match the lockfile and returns a
Runner with the composed environment. env_for(*names) composes already-installed
packages' env without installing anything.

The facade resolves lazily (PEP 562): ``pm.environments`` runs at process boot,
before any dependency is importable, and must not drag the downloader/ensure
machinery in with it.
"""

from __future__ import annotations

import sys

_EXPORTS = {
    "pm.install": (
        "activate", "check", "drift", "enabled_extras", "env_for", "is_installed",
        "installed_package", "lazy_installs_allowed",
    ),
    "pm.client": (
        "ensure", "sync_venv", "build_environment", "lock_project", "stage_manager_runtime",
        "ensure_environment", "ensure_project_environment", "ensure_python_tool", "venv_is_current", "check_project_lock",
        "export_requirements", "build_requirements_environment", "prune_cache", "stage_tools",
        "prepare_tools",
    ),
    "pm.operations": ("environment_python", "python_tool"),
    "pm.extras": ("available", "ensure_import", "install_hint"),
    "pm.lock": ("Facts", "Lockfile"),
    "pm.package": ("InstallError", "Package", "Runner", "compose_env"),
    "pm.registry": ("all_packages", "get_package", "register", "walk"),
    "pm.store": ("Store", "current_target"),
}
_HOME = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_HOME)


def __getattr__(name: str):
    module = _HOME.get(name)
    if module is None:
        raise AttributeError(f"module 'pm' has no attribute {name!r}")
    # Not importlib.import_module: tests patch that globally and the facade must keep resolving.
    __import__(module)
    value = getattr(sys.modules[module], name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_HOME))


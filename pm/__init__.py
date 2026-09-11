"""pm: the hermes package system.

Everything hermes depends on — tool binaries, the python venv, node_modules
dirs, plugins — is a package in one dependency tree. Package definitions
(pm/packages.py) say what a package IS. The lockfile (pm/lock.json,
machine-written) says exactly which versions and hashes. The installed-state
file (facts.json, per install) says what is actually on this machine.

ensure(name) makes the installed state match the lockfile and returns a
Runner with the composed environment. env_for(*names) composes already-installed
packages' env without installing anything.
"""

from pm.ensure import (
    activate,
    adopt,
    check,
    enabled_extras,
    env_for,
    is_installed,
    installed_package,
    lazy_installs_allowed,
)
from pm.client import (
    ensure, sync_venv, build_environment, lock_project, stage_manager_runtime,
    ensure_environment, ensure_python_tool, venv_is_current,
    check_project_lock, export_requirements, build_requirements_environment, prune_cache,
)
from pm.operations import environment_python, python_tool
from pm.extras import available, ensure_import
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Package, Runner, compose_env
from pm.registry import all_packages, get_package, register, walk
from pm.store import Store, current_target

__all__ = [
    "ensure",
    "env_for",
    "is_installed",
    "installed_package",
    "adopt",
    "check",
    "activate",
    "sync_venv",
    "available",
    "ensure_import",
    "enabled_extras",
    "lazy_installs_allowed",
    "build_environment",
    "lock_project",
    "stage_manager_runtime",
    "ensure_environment",
    "environment_python",
    "ensure_python_tool",
    "python_tool",
    "venv_is_current",
    "check_project_lock",
    "export_requirements",
    "build_requirements_environment",
    "prune_cache",
    "Facts",
    "Lockfile",
    "InstallError",
    "Package",
    "Runner",
    "compose_env",
    "all_packages",
    "get_package",
    "register",
    "walk",
    "Store",
    "current_target",
]

import pm.packages  # noqa: E402,F401  (registers the built-in definitions)

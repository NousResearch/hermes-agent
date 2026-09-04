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
    ensure,
    env_for,
    is_installed,
    lazy_installs_allowed,
    sync_venv,
    uv,
)
from pm.extras import available, ensure_import
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Package, Runner, compose_env
from pm.registry import all_packages, get_package, register, walk
from pm.store import Store, current_target

__all__ = [
    "ensure",
    "env_for",
    "is_installed",
    "adopt",
    "check",
    "activate",
    "sync_venv",
    "available",
    "ensure_import",
    "enabled_extras",
    "lazy_installs_allowed",
    # NOTE: pm.uv is importable but deliberately not in __all__ — it is the
    # marked transitional bridge for update/repair and must not spread.
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

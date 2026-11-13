"""Admission against a dependency environment PM cannot rebuild.

A Nix package's Python environment is its store closure: immutable, and built from a lockfile and
a source tree the installed package does not carry. PM therefore never builds a generation there.
A plugin or extra is admitted when the environment the application already runs from satisfies
what it declares, and refused by name otherwise, so the user knows what to add to the package.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

# The application's import path, handed to the PM worker: the worker runs on PM's own interpreter
# and cannot see what the application (and the wrapper's PYTHONPATH additions) can import.
APPLICATION_PATH_ENV = "HERMES_PM_APPLICATION_PATH"

REMEDY = ("add them to the Nix package (extraPythonPackages for plugin requirements, "
          "extraDependencyGroups for Hermes extras) and rebuild; nothing was changed")


def export_application_path(environment: dict[str, str]) -> None:
    """Record this process's import path for a PM child, unless this process is PM's own.

    PM's interpreter must pass the application's path on, not replace it with its own.
    """
    import sys
    from pm.package import InstallError
    from pm.runtime import is_runtime

    try:
        own = is_runtime()
    except InstallError:
        own = False  # A broken packaged runtime is reported where it is launched, not here.
    if not own:
        environment[APPLICATION_PATH_ENV] = os.pathsep.join(entry for entry in sys.path if entry)


def is_nix_package() -> bool:
    """True when the tree being synced is a Nix package's own.

    The package keeps its stamp outside the code (``HERMES_INSTALL_ROOT``), and only the executing
    tree may resolve through it. A PM worker's ``repo_root`` is the tree its caller named, so a
    Nix-installed Hermes that syncs some other checkout must read that checkout's stamp, not its own.
    """
    from pm.paths import install_root, repo_root

    root = repo_root()
    executing = Path(__file__).resolve().parent.parent
    owner = install_root() if Path(root).resolve() == executing else Path(root)
    try:
        stamp = json.loads((owner / "install-stamp.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return isinstance(stamp, dict) and stamp.get("distribution") == "nix"


def application_path() -> list[str] | None:
    """The application's import path, or None when no caller supplied one."""
    value = os.environ.get(APPLICATION_PATH_ENV)
    if not value:
        return None
    return [entry for entry in value.split(os.pathsep) if entry]


def _installed(path: list[str]) -> dict:
    from importlib.metadata import distributions
    from packaging.utils import canonicalize_name

    found: dict = {}
    for distribution in distributions(path=path):
        name = distribution.metadata.get("Name")
        if name:
            # The first entry on the path is the one an import resolves to.
            found.setdefault(canonicalize_name(name), distribution)
    return found


def _unmet(specs: Iterable[str], installed: dict, *, extra: str = "") -> list[str]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    missing = []
    for spec in specs:
        requirement = Requirement(spec)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": extra}):
            continue
        distribution = installed.get(canonicalize_name(requirement.name))
        if distribution is None:
            missing.append(str(requirement))
            continue
        try:
            version = Version(distribution.version)
        except InvalidVersion:
            continue  # Present; a version no specifier can be compared with is not provably wrong.
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            missing.append(f"{requirement} (found {distribution.version})")
    return missing


def _extra_requirements(extra: str, installed: dict, seen: set[str]) -> list[str] | None:
    """Requirements the installed Hermes declares for *extra*; None when it declares no such extra."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    key = canonicalize_name(extra)
    core = installed.get("hermes-agent")
    if core is None:
        return None
    declared = {canonicalize_name(name) for name in core.metadata.get_all("Provides-Extra") or ()}
    if key not in declared:
        return None
    if key in seen:
        return []
    seen.add(key)
    wanted: list[str] = []
    for spec in core.requires or ():
        requirement = Requirement(spec)
        marker = requirement.marker
        if marker is None or "extra" not in str(marker) or not marker.evaluate({"extra": key}):
            continue
        if canonicalize_name(requirement.name) == "hermes-agent":
            # ``all = ["hermes-agent[web]", ...]``: an extra made of other extras.
            for nested in requirement.extras:
                wanted.extend(_extra_requirements(nested, installed, seen) or ())
            continue
        requirement.marker = None
        wanted.append(str(requirement))
    return wanted


def missing(extras: Iterable[str] = (), members: Iterable[Path] | Mapping | None = None) -> list[str]:
    """What the package's environment lacks for *extras* and plugin *members*, as readable entries.

    A member is also reported when its ``requires-python`` excludes the running interpreter, which
    is the application's: the package builds PM's runtime and the application on one Python.
    """
    import platform
    from pm.plugin_declarations import read_python_declaration

    # A staged update maps the installed plugin (the name the user knows) to its staged source.
    sources = list(members.items() if isinstance(members, Mapping) else ((d, d) for d in members or ()))
    declarations = [(Path(identity).name, read_python_declaration(Path(directory))) for identity, directory in sources]
    path = application_path()
    if path is None:
        # Reached only when PM was started by something other than Hermes; nothing can be proven
        # either way, and admitting unchecked would report an install that did not happen.
        asked = [f"extra '{extra}'" for extra in extras or ()]
        asked += [f"plugin '{name}'" for name, declaration in declarations if declaration.install_requirements]
        return [f"{entry} (cannot be checked: the application's import path is unknown)" for entry in asked]
    installed = _installed(path)
    report: list[str] = []
    for extra in extras or ():
        wanted = _extra_requirements(extra, installed, set())
        if wanted is None:
            report.append(f"extra '{extra}' (not built into this package)")
            continue
        report.extend(f"{spec} (extra '{extra}')" for spec in _unmet(wanted, installed))
    for name, declaration in declarations:
        if reason := declaration.python_error(platform.python_version()):
            report.append(f"plugin '{name}' {reason}")
        report.extend(f"{spec} (plugin '{name}')" for spec in _unmet(declaration.install_requirements, installed))
    return report

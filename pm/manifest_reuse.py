"""No-download admission of native plugins with already-pinned Python requirements.

Only the PM worker calls this while holding its dependency publication lock.
An already recorded, current environment and the core's resolved lock remain
authoritative. No plugin Python code, .pth file or network resource is run.
"""
from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


def reusable_manifest_selection(package: Any, change: Any, fact: dict,
                                enabled: list[str], candidate_inputs: dict) -> bool:
    """True iff exactly one *new* native manifest needs no package changes.

    This does not admit pyproject members, extras, environment markers, direct
    URLs, changed existing members, changed core locks, or invalid dependency
    state. The caller still uses the normal transactional PluginSelection
    publisher and writes the new fingerprint under its held PM install lock.
    """
    from pm.environments import install_state_dir
    from pm.install import _runtime_state_matches
    from pm.plugin_declarations import read_python_declaration
    from pm.publication import PluginSelection
    from pm.workspace import build_member_dirs, member_sources

    if type(change) is not PluginSelection or not isinstance(fact, dict):
        return False
    if not isinstance(candidate_inputs.get("plugin_dirs"), (list, tuple, dict)):
        return False
    if not isinstance(fact.get("extras"), list) or sorted(fact["extras"]) != sorted(enabled):
        return False
    if any(not isinstance(fact.get(k), str) or not fact[k]
           for k in ("environment", "resolved_lock")):
        return False
    project = package.project_root()
    old = member_sources(build_member_dirs(enabled))
    new = member_sources(candidate_inputs["plugin_dirs"])
    if not set(old).issubset(new) or any(old[key] != new[key] for key in old):
        return False
    addition = set(new) - set(old)
    if len(addition) != 1:
        return False
    if not _runtime_state_matches(fact, package.expected_stamp(enabled, plugin_dirs=old)):
        return False
    identity = next(iter(addition))
    declaration = read_python_declaration(new[identity])
    if declaration.pyproject is not None or declaration.external or not declaration.installs_packages:
        return False
    specs = declaration.install_requirements
    # install_requirements filters direct-URL/self references. Such declarations
    # must never be silently treated as satisfied by a different dependency set.
    if not specs or len(specs) != len(declaration.requirements):
        return False

    environment = Path(fact.get("environment", "")).resolve()
    lock = Path(fact.get("resolved_lock", "")).resolve()
    generations = install_state_dir(project) / "environments"
    if (environment.parent.parent != generations.resolve()
            or environment.name != "venv"
            or lock != environment.parent / "workspace" / "uv.lock"
            or not lock.is_file()):
        return False
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    try:
        required = {}
        for spec in specs:
            requirement = Requirement(spec)
            # Extras can introduce transitive deps that the current environment
            # has not qualified. Markers require host-platform evaluation.
            if requirement.url or requirement.extras or requirement.marker:
                return False
            name = canonicalize_name(requirement.name)
            if name in required and required[name] != str(requirement.specifier):
                return False
            required[name] = str(requirement.specifier)
        entries = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
        matched = {}
        for row in entries:
            name = canonicalize_name(str(row.get("name", "")))
            if name not in required:
                continue
            if name in matched or not isinstance(row.get("version"), str):
                return False
            # Native plugins may reuse packages from the recorded registry
            # closure, never Git, direct URL, local paths or editables.
            if not isinstance(row.get("source"), dict) or "registry" not in row["source"]:
                return False
            matched[name] = row["version"]
        if set(matched) != set(required):
            return False
        from packaging.specifiers import SpecifierSet
        if any(not SpecifierSet(required[name]).contains(version, prereleases=True)
               for name, version in matched.items()):
            return False
        return _installed_versions_match(environment, matched)
    except (OSError, ValueError, KeyError, TypeError):
        return False


# Stdlib-only child: -I -S prevents .pth execution and importing native code.
# It reads only distribution metadata in the committed environment.
_INVENTORY = r"""
import importlib.metadata as md, json, pathlib, re, sys
base = pathlib.Path(sys.argv[1])
site = [base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        base / "Lib" / "site-packages"]
names = json.loads(sys.stdin.read())
found = {}
for item in md.distributions(path=[str(p) for p in site if p.is_dir()]):
    name = item.metadata.get("Name")
    if not isinstance(name, str):
        continue
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if normalized in names:
        if normalized in found:
            sys.exit(2)
        found[normalized] = item.version
print(json.dumps(found, sort_keys=True))
"""


def _installed_versions_match(environment: Path, matched: dict[str, str]) -> bool:
    from pm.environments import venv_python

    executable = venv_python(environment)
    if not executable.is_file():
        return False
    try:
        result = subprocess.run(
            [str(executable), "-I", "-S", "-B", "-c", _INVENTORY, str(environment)],
            input=json.dumps(sorted(matched)), capture_output=True, text=True,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=15, check=False,
        )
        if result.returncode or len(result.stdout) > 4096:
            return False
        found = json.loads(result.stdout)
        return found == matched
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False

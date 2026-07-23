#!/usr/bin/env python3
"""Generate an Android-safe Hermes requirements set from project metadata.

The regular project metadata intentionally uses desktop/server extras such as
``uvicorn[standard]`` and Telegram webhooks. On native Termux those extras pull
native packages that are either unsupported or redundant. This helper expands
the curated Termux profile, removes only those optional accelerators, and emits
lock-version constraints without adding TUR as a global package index.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError as exc:  # pragma: no cover - installer supplies packaging first
    raise SystemExit(
        "packaging must be installed before generating Termux requirements"
    ) from exc

SELF_NAME = "hermes-agent"
DROP_EXTRAS = {
    "uvicorn": {"standard"},
    "python-telegram-bot": {"webhooks"},
}


def _android_environment(python_version: str | None = None) -> dict[str, str]:
    env = default_environment()
    env.update({
        "sys_platform": "android",
        "platform_system": "Android",
        "os_name": "posix",
    })
    if python_version:
        parts = python_version.split(".")
        env["python_version"] = ".".join(parts[:2])
        env["python_full_version"] = python_version
    return env


def _render_requirement(requirement: Requirement, extras: set[str]) -> str:
    rendered = requirement.name
    if extras:
        rendered += "[" + ",".join(sorted(extras)) + "]"
    if requirement.url:
        rendered += f" @ {requirement.url}"
    else:
        rendered += str(requirement.specifier)
    if requirement.marker:
        rendered += f"; {requirement.marker}"
    return rendered


def expand_termux_requirements(
    pyproject: Path, *, profile: str = "termux-all", python_version: str | None = None
) -> list[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data["project"]
    extras_table = project.get("optional-dependencies", {})
    pending = list(project.get("dependencies", []))
    pending.extend(extras_table.get(profile, []))
    expanded_extras: set[str] = {profile}
    output: list[str] = []
    seen: set[str] = set()
    marker_env = _android_environment(python_version)

    while pending:
        raw = pending.pop(0)
        req = Requirement(raw)
        name = canonicalize_name(req.name)
        if name == SELF_NAME:
            for extra in sorted(req.extras):
                if extra in expanded_extras:
                    continue
                expanded_extras.add(extra)
                pending.extend(extras_table.get(extra, []))
            continue
        if req.marker and not req.marker.evaluate(marker_env):
            continue
        kept_extras = set(req.extras) - DROP_EXTRAS.get(name, set())
        rendered = _render_requirement(req, kept_extras)
        if rendered not in seen:
            seen.add(rendered)
            output.append(rendered)

    return sorted(output, key=lambda value: canonicalize_name(Requirement(value).name))


def lock_constraints(lockfile: Path) -> list[str]:
    data = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in data.get("package", []):
        name = package.get("name")
        version = package.get("version")
        source = package.get("source") or {}
        if not name or not version or "registry" not in source:
            continue
        versions.setdefault(canonicalize_name(name), set()).add(str(version))
    # A universal uv.lock may legitimately contain platform forks. A conflicting
    # multi-version package is left unconstrained rather than emitting an
    # impossible pair of constraints.
    return [
        f"{name}=={next(iter(found))}"
        for name, found in sorted(versions.items())
        if len(found) == 1
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--profile", default="termux-all")
    parser.add_argument("--python-version")
    args = parser.parse_args(argv)

    requirements = expand_termux_requirements(
        args.pyproject, profile=args.profile, python_version=args.python_version
    )
    constraints = lock_constraints(args.lock)
    args.requirements.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    args.constraints.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    print(
        f"Generated {len(requirements)} direct Android requirements and "
        f"{len(constraints)} lock constraints"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

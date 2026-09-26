"""Assemble a runnable agent from prepared inputs, without downloads or live state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tomllib

from scripts.build.inputs import AgentInputs, RESOURCE_ENV, project_entries, validate_frontends
from scripts.build.launchers import write_launchers


def _copy(source: Path, destination: Path, *, replace: bool = False) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"output must not be inside copied input: {source}")
    if replace and destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".git"))
    else:
        shutil.copy2(source, destination)


def plant_surfaces(repo: Path, frontends: dict[str, Path]) -> None:
    """Only explicit built products are accepted; no hunting in the source tree."""
    validate_frontends(frontends)
    if "tui" in frontends:
        tui = frontends["tui"]
        destination = repo / "hermes_cli/tui_dist"
        _copy(tui / "dist", destination, replace=True)
        _copy(tui / "package.json", destination / "package.json")
    if "web" in frontends:
        _copy(frontends["web"], repo / "hermes_cli/web_dist", replace=True)


def write_metadata(project_path: Path, repo: Path) -> None:
    """Source-layout installs have no wheel; preserve PEP 621 dependency metadata.

    This attaches the project's static declarations, not installed-environment
    guesses. Nix keeps the real wheel's dist-info untouched.
    """
    project = tomllib.loads(project_path.read_text(encoding="utf-8-sig"))["project"]
    if project.get("dynamic"):
        raise ValueError("agent assembly requires static project metadata")
    dist = re.sub(r"[-_.]+", "_", project["name"])
    info = repo / f"{dist}-{project['version']}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    lines = ["Metadata-Version: 2.3", f"Name: {project['name']}", f"Version: {project['version']}"]
    for key, header in (("description", "Summary"), ("requires-python", "Requires-Python")):
        if key in project:
            lines.append(f"{header}: {project[key]}")
    lines.extend(f"Requires-Dist: {req}" for req in project.get("dependencies", []))
    for extra, requirements in project.get("optional-dependencies", {}).items():
        extra = re.sub(r"[-_.]+", "-", extra).lower()
        lines.append(f"Provides-Extra: {extra}")
        for req in requirements:
            requirement, _, marker = req.partition(";")
            condition = f"({marker.strip()}) and " if marker else ""
            lines.append(f"Requires-Dist: {requirement.strip()}; {condition}extra == '{extra}'")
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = {**project.get("entry-points", {}), "console_scripts": project_entries(project_path)}
    if project.get("gui-scripts"):
        groups["gui_scripts"] = project["gui-scripts"]
    text = "\n".join(f"[{group}]\n" + "\n".join(f"{name} = {entry}" for name, entry in values.items())
                     for group, values in groups.items())
    (info / "entry_points.txt").write_text(text + "\n", encoding="utf-8")


def assemble(inputs: AgentInputs, out: Path) -> dict:
    out = out.absolute()
    out.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave an old completion claim behind.
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    manifest_path.unlink(missing_ok=True)
    (out / "command-map.json").unlink(missing_ok=True)
    inputs.validate(out)
    entries = project_entries(inputs.project)
    repo = out / inputs.repo
    referenced = inputs.placement == "references"
    repo.mkdir(parents=True, exist_ok=True)
    if not referenced:
        if inputs.code.resolve() != repo.resolve():
            if repo == out:
                raise ValueError("root-level code must be prepared in place")
            _copy(inputs.code, repo, replace=True)
        for name, source in inputs.resources.items():
            _copy(source, repo / name, replace=True)
        plant_surfaces(repo, inputs.frontends)
        write_metadata(inputs.project, repo)
        # Arbitrary scripts run with the prepared environment's Python too
        # (Docker's config migration, plugins), not only generated launchers.
        if not inputs.site_packages.resolve().is_relative_to(out):
            raise ValueError("source binding requires an output-owned dependency directory")
        relative_code = Path(os.path.relpath(repo, inputs.site_packages)).as_posix()
        (inputs.site_packages / "hermes-agent.pth").write_text(relative_code + "\n", encoding="utf-8")
    for source, destination in ((inputs.stamp, repo / "install-stamp.json"),
                                (inputs.features, out / "enabled-features.json")):
        if source is not None:
            _copy(source, destination)
    if inputs.placement == "contained":
        from scripts.bundles.payload import relativize_links
        relativize_links(out)
    def location(path: Path) -> str:
        return path.relative_to(out).as_posix() if path.is_relative_to(out) and not referenced else str(path)

    python, site = location(inputs.python), location(inputs.site_packages)
    tools = location(inputs.tools) if inputs.tools else "tools"
    if referenced:
        commands = {name: str(inputs.command_dir / name) for name in entries}
        for command in commands.values():
            if not Path(command).is_file():
                raise FileNotFoundError(f"prepared command missing: {command}")
        env = {**inputs.env, "HERMES_PYTHON": str(inputs.python), "HERMES_INSTALL_ROOT": str(repo)}
        links = {RESOURCE_ENV[name]: (path, repo / name) for name, path in inputs.resources.items()}
        for name, path in inputs.frontends.items():
            key, destination = {"tui": ("HERMES_TUI_DIR", out / "ui-tui"),
                                "web": ("HERMES_WEB_DIST", repo / "web_dist")}[name]
            links[key] = (path, destination)
        for key, (source, destination) in links.items():
            destination.unlink(missing_ok=True)
            destination.symlink_to(source, target_is_directory=True)
            env[key] = str(destination)
        mapping = {"commands": {name: {"source": commands[name], "destination": f"{inputs.bin_dir}/{name}",
                                      "entry": entry} for name, entry in entries.items()}, "env": env}
        (out / "command-map.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    else:
        for name, command in previous.get("runtime", {}).get("commands", {}).items():
            if name not in entries and command == f"{inputs.bin_dir}/{name}{'.exe' if inputs.target.startswith('win32') else ''}":
                (out / command).unlink(missing_ok=True)
        launch_python = f"{location(inputs.environment)}/bin/python" if inputs.target.endswith("-bionic") else python
        commands = write_launchers(out, entries, python=launch_python, repo=inputs.repo, site=site,
                                   target=inputs.target, bin_dir=inputs.bin_dir)
    manifest = {"schema": 1, "target": inputs.target, "repo": inputs.repo,
                "venv": location(inputs.environment), "store": tools,
                "launchers": list(entries), "runtime": {
                    "repoDir": str(inputs.code) if referenced else inputs.repo,
                    "toolsDir": tools, "storePython": python,
                    "sitePackages": site, "commands": commands}}
    if inputs.ref is not None:
        manifest["ref"] = inputs.ref
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = AgentInputs.from_dict(json.loads(args.inputs.read_text(encoding="utf-8-sig")))
        assemble(inputs, args.out)
    except Exception:
        (args.out / "manifest.json").unlink(missing_ok=True)
        (args.out / "command-map.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()

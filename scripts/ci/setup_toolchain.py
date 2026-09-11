"""GitHub Actions file commands around the real, stdlib-only PM bootstrap."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import sys

# The runner invokes this file before the checkout has been installed.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.lock import Lockfile
from pm.paths import lockfile_path
from pm.registry import walk
from pm.store import current_target


def packages(toolchain: str) -> list[str]:
    roots = {"python": ["python", "uv"], "node": ["npm"], "all": ["python", "uv", "npm"]}
    return sorted(package.name for package in walk(roots[toolchain]))


def file_commands(destination: str, values: dict) -> None:
    """All exports are single-line values, including JSON-encoded arrays."""
    lines = []
    for key, value in values.items():
        text = str(value)
        if any(character in key + text for character in "\r\n\0"):
            raise ValueError(f"invalid GitHub file command: {key!r}")
        lines.append(f"{key}={text}\n")
    with open(os.environ[destination], "a", encoding="utf-8") as stream:
        stream.writelines(lines)


def parse_extras(value: str) -> list[str] | None:
    if value == "":
        return None
    message = "extras must be a JSON array of extra names"
    try:
        extras = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if not isinstance(extras, list) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
        for name in extras
    ):
        raise argparse.ArgumentTypeError(message)
    return sorted(set(extras))


def prepare(args) -> None:
    home = args.home.resolve()
    lock = Lockfile(lockfile_path())
    target = current_target()
    names = packages(args.toolchain)
    values = {
        "packages": json.dumps(names), "target": target, "arch": target.split("-")[1],
        "store": str(home / "tools"),
    }
    for name in names:
        version = lock.version(name)
        if not version or not lock.artifacts(name, target):
            raise ValueError(f"{name} has no pinned artifact for {target}")
        values[f"{name}-version"] = version.partition("+")[0] if name == "python" else version
    # The OS image belongs in the uv cache identity: built wheels can link
    # against its system libraries. Unlike npm's cache, these are not just JS.
    values["os-version"] = platform.platform()
    values["bootstrap-python"] = sys.executable
    file_commands("GITHUB_OUTPUT", values)
    file_commands("GITHUB_ENV", {
        "HERMES_HOME": home,
        "HERMES_RUNTIME_DIR": home / "tools",
        "PYTHONUTF8": "1",
    })


def add_path(directories: list[str]) -> None:
    # The runner prepends each line, so write PM's dependent-first PATH in
    # reverse. npm must shadow the different npm bundled inside Node.
    with open(os.environ["GITHUB_PATH"], "a", encoding="utf-8") as stream:
        for directory in reversed(list(dict.fromkeys(directories))):
            if any(character in directory for character in "\r\n\0"):
                raise ValueError("invalid PATH directory")
            stream.write(directory + "\n")


def python3_alias(python: Path) -> None:
    if os.name == "nt":
        import shutil

        alias = python.with_name("python3.exe")
        if not alias.exists():
            shutil.copy2(python, alias)


def archive_inputs(args) -> None:
    from pm.paths import repo_root
    from pm.store import Store
    from scripts.ci.archive_inputs import Archive, pinned_inputs, stage_inputs
    from scripts.releases import r2

    pins = pinned_inputs(repo_root(), target=current_target(), packages=set(packages(args.toolchain)))
    stage_inputs(pins, archive=Archive(*r2.credentials()), store=Store(args.home.resolve() / "tools"))


def install(args) -> None:
    import subprocess

    from pm.cli import _live_progress
    from pm.ensure import ensure, env_for
    from pm.lock import Facts
    from pm.packages import uv_cache_dir
    from pm.paths import facts_path, store_root
    from pm.registry import get_package

    names = packages(args.toolchain)
    for name in names:
        ensure(name, explicit=True, progress=_live_progress(name))
    facts = Facts(facts_path())
    target = current_target()
    binaries = {
        name: get_package(name).binary(store_root() / facts.get(name)["entry"], target)
        for name in names
    }
    environment = env_for(*names)
    path = env_for(*names, base_env={})["PATH"].split(os.pathsep)
    exported = {}
    outputs = {f"{name}-path": str(binary) for name, binary in binaries.items()}
    if "python" in names:
        tool_bin = args.home.resolve() / "bin"
        # A writable command environment keeps callers' uv pip installs out
        # of the verified tool store. On Windows its redirector also supplies
        # python3.exe, which PBS does not ship.
        commands = args.home.resolve() / "python" / facts.get("python")["entry"]
        if not (commands / "pyvenv.cfg").is_file():
            subprocess.run(
                [str(binaries["uv"]), "venv", "--relocatable", "--python", str(binaries["python"]), str(commands)],
                check=True, env=environment, timeout=120,
            )
        python = commands / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python3_alias(python)
        path.insert(0, str(python.parent))
        outputs["python-path"] = str(python)
        exported.update({
            "HERMES_PYTHON": python,
            "UV_PYTHON": binaries["python"],
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_CACHE_DIR": uv_cache_dir(),
            "UV_TOOL_DIR": args.home.resolve() / "uv-tools",
            "UV_TOOL_BIN_DIR": tool_bin,
        })
        outputs["uv-cache-path"] = str(uv_cache_dir())
        path.insert(0, str(tool_bin))
    if "npm" in names:
        cache = subprocess.check_output(
            [str(binaries["npm"]), "config", "get", "cache"],
            env=environment, text=True, encoding="utf-8", timeout=60,
        ).strip()
        outputs["npm-cache-path"] = cache
    file_commands("GITHUB_ENV", exported)
    file_commands("GITHUB_OUTPUT", outputs)
    add_path(path)
    print("PM toolchain ready: " + ", ".join(f"{name} {facts.get(name)['version']}" for name in names))


def dependencies(args) -> None:
    if args.extras is None:
        return
    import subprocess
    import tomllib

    from pm.runtime import is_runtime, runtime_command, runtime_environment

    if not is_runtime():
        subprocess.run(
            runtime_command(Path(__file__).resolve(),
                            ["dependencies", "--home", str(args.home.resolve()),
                             "--toolchain", args.toolchain, "--extras", json.dumps(args.extras)]),
            env=runtime_environment(), check=True,
        )
        return

    from hermes_cli.runtime_paths import selected_venv
    from pm.ensure import sync_venv, uv
    from pm.paths import repo_root

    project = repo_root()
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8-sig"))
    unknown = set(args.extras) - metadata["project"]["optional-dependencies"].keys()
    if unknown:
        raise ValueError(f"unknown project extras: {sorted(unknown)}")
    uv_bin, environment = uv()
    environment.pop("UV_NO_CONFIG", None)  # keep project indexes and exclude-newer
    # PM's frozen sync must not turn a stale project lock into a green job.
    subprocess.run([uv_bin, "lock", "--check"], cwd=project, env=environment, check=True, timeout=1800)
    if "dev" in args.extras:
        # Test-only groups must not enter PM facts or a shipped generation.
        venv = args.home.resolve() / "test-environment"
        environment["UV_PROJECT_ENVIRONMENT"] = str(venv)
        command = [uv_bin, "sync", "--locked", "--group", "test", "--no-install-project"]
        for extra in args.extras:
            command.extend(["--extra", extra])
        subprocess.run(command, cwd=project, env=environment, check=True, timeout=1800)
    else:
        sync_venv(args.extras, explicit=True, plugin_dirs=[])
        venv = selected_venv(project)
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    python3_alias(python)
    file_commands("GITHUB_ENV", {
        "HERMES_PYTHON": python,
        "VIRTUAL_ENV": venv,
        "UV_PROJECT_ENVIRONMENT": venv,
        "PYTHONPATH": project,
    })
    file_commands("GITHUB_OUTPUT", {"python-path": python, "venv": venv})
    add_path([str(bindir)])
    print(f"PM dependencies ready: {args.extras} in {venv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {"prepare": prepare, "archive-inputs": archive_inputs, "install": install, "dependencies": dependencies}
    parser.add_argument("phase", choices=list(phases))
    parser.add_argument("--toolchain", choices=["python", "node", "all"], default="python")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--extras", type=parse_extras, default="")
    args = parser.parse_args()
    if args.toolchain == "node" and args.extras is not None:
        parser.error("extras require the python or all toolchain")
    os.environ["HERMES_HOME"] = str(args.home.resolve())
    os.environ["HERMES_RUNTIME_DIR"] = str(args.home.resolve() / "tools")
    phases[args.phase](args)


if __name__ == "__main__":
    main()

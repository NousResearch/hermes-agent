"""Wire staged bionic inputs to PM without exposing its installer executable."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def prepare_tools(source: Path, root: Path) -> None:
    from pm.paths import lockfile_path
    from scripts.bundles.payload import record_tools

    store = root / "tools"
    store.mkdir(parents=True, exist_ok=True)
    for name in ("python", "uv"):
        entry = store / name
        if entry.absolute() != (source / name).absolute():
            entry.symlink_to((source / name).resolve(), target_is_directory=True)
    record_tools(root, lockfile_path(), "linux-arm64-bionic", {name: name for name in ("python", "uv")})


def application(root: Path, requirements: Path, python: Path) -> None:
    from pm import build_requirements_environment

    build_requirements_environment(
        requirements.read_text(encoding="utf-8").splitlines(), out=root / "venv",
        python=python, wheelhouse=root / "wheelhouse", offline=True, sealed=True, explicit=True,
    )


def assemble(root: Path, requirements: Path, python: Path) -> None:
    from pm import stage_manager_runtime
    from scripts.bundles.payload import seal_pm_runtime

    prepare_tools(root / "tools", root)
    manager = stage_manager_runtime(
        python=python, destination=root / "pm-runtime", project=root / "app/pm",
        wheelhouse=root / "wheelhouse", offline=True,
    )
    # The staged manager is complete but not published. Execute the application
    # build in it before sealing removes its build-time executable links.
    subprocess.run(
        [str(manager), "-I", "-B", str(Path(__file__).resolve()), "application",
         "--root", str(root), "--requirements", str(requirements), "--python", str(python)],
        check=True,
    )
    seal_pm_runtime(root, python)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare-tools", "assemble", "application"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-tools", type=Path)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ["HERMES_RUNTIME_DIR"] = str(root / "tools")
    if args.phase == "prepare-tools":
        if args.source_tools is None:
            parser.error("prepare-tools requires --source-tools")
        prepare_tools(args.source_tools, root)
    else:
        if args.requirements is None or args.python is None:
            parser.error("environment assembly requires --requirements and --python")
        operation = assemble if args.phase == "assemble" else application
        operation(root, args.requirements.resolve(), args.python.absolute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Native/Docker dependency preparation over PM's explicit uv mechanics.

Unlike live PM, a build never discovers plugins or publishes a selection.
The source is the prepared application tree; editable installs keep pointing
at that tree. Nix and Termux retain their own dependency providers.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pm.environment import PythonEnvironment, prune_site_pth
from pm.package import InstallError


def build_python_environment(
    *, source: Path, python: Path, uv: Path, out: Path,
    env: Mapping[str, str], cache: Path | None = None,
    extras: Sequence[str] = (), all_extras: bool = False,
    no_install_project: bool = False, offline: bool = False,
) -> Path:
    """Build a fresh environment and return its Python executable after validation.

    ``out`` must not exist. Failure removes only this invocation's destination;
    a pre-existing environment is never deleted or used as a successful result.
    The cache (default: ``out.parent / ".uv-cache"``) remains available to the
    native offline-cache packager. ``no_install_project`` prepares only the
    locked dependencies, before an application source layer is available.
    """
    source, out = source.absolute(), out.absolute()
    cache = out.parent / ".uv-cache" if cache is None else cache.absolute()
    environment = PythonEnvironment(
        uv=uv.absolute(), python=python.absolute(), destination=out,
        cache=cache.absolute(), env=env, offline=offline, output=sys.stderr,
    )
    out.mkdir(parents=True)
    try:
        environment.create()
        environment.sync(source, extras=extras, all_extras=all_extras,
                         no_install_project=no_install_project)
        environment.check()
        # The sealed payload must never process uv's venv-marker or
        # editable-install .pth files (see pm.environment.prune_site_pth):
        # the launcher addsitedirs the venv, and those two would repoint
        # sys.prefix / shadow the repo snapshot with build-machine paths.
        prune_site_pth(out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return environment.executable


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="uv cache; defaults to OUT's sibling .uv-cache")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--extra", dest="extras", action="append", default=[])
    selection.add_argument("--all-extras", action="store_true")
    parser.add_argument("--no-install-project", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        executable = build_python_environment(**vars(args), env=dict(os.environ))
    except (InstallError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"python environment: {exc}", file=sys.stderr)
        return 1
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

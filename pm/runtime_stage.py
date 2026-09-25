"""One locked dependency builder for PM workers and packaged runtimes."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from pm.package import InstallError


def _locked_artifacts(lock: Path) -> dict[tuple[str, str], set[str]]:
    """Read the pinned graph and artifact digests from uv's small PM lock."""
    packages: dict[tuple[str, str], set[str]] = {}
    for block in lock.read_text(encoding="utf-8").split("[[package]]")[1:]:
        name = re.search(r'^name = "([^"]+)"$', block, re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"$', block, re.MULTILINE)
        if not name or not version:
            raise InstallError("pm-runtime", "invalid PM lock package entry")
        key = (name.group(1), version.group(1))
        if key in packages:
            raise InstallError("pm-runtime", "duplicate PM lock package entry")
        packages[key] = set(re.findall(r'\bhash = "(sha256:[0-9a-f]+)"', block))
    if not packages:
        raise InstallError("pm-runtime", "empty PM lock")
    return packages


def stage_runtime(uv: Path, python: Path, destination: Path, *,
                  project: Path | None = None, offline: bool = False,
                  wheelhouse: Path | None = None, cache: Path | None = None) -> Path:
    """Build at the final path; the caller owns publication and its marker.

    The scratch project prevents uv from discovering the application's workspace.
    No project install, application extra, or application lock enters this graph.
    """
    from pm.environment import PythonEnvironment
    from pm.packages import uv_cache_dir
    from pm.runtime import runtime_environment

    project = project or Path(__file__).resolve().parent
    destination = destination.absolute()
    env = runtime_environment()
    environment = PythonEnvironment(
        uv=uv, python=python, destination=destination,
        cache=uv_cache_dir() if cache is None else cache.absolute(), env=env,
        offline=offline or wheelhouse is not None, output=sys.stderr, no_config=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pm-project-", dir=destination.parent) as temp:
        snapshot = Path(temp)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(project / name, snapshot / name)
        environment.create()
        if wheelhouse is None:
            # uv considers registry identity part of --locked. Resolve only
            # in the disposable snapshot; retain the pinned graph and bytes.
            from pm.index_config import _UV_INDEX_KNOBS
            if any(env.get(key) for key in _UV_INDEX_KNOBS):
                original = _locked_artifacts(snapshot / "uv.lock")
                environment.lock(snapshot, timeout=600)
                mirrored = _locked_artifacts(snapshot / "uv.lock")
                if original.keys() != mirrored.keys() or any(
                    not hashes or not hashes <= original[key]
                    for key, hashes in mirrored.items() if key != ("hermes-pm-runtime", "0.0.0")
                ):
                    raise InstallError("pm-runtime", "mirror changed the pinned PM dependency graph or artifacts")
            environment.sync(snapshot, locked=True, no_default_groups=True,
                             no_install_project=True, timeout=600)
        else:
            environment.install_wheelhouse(snapshot, wheelhouse, timeout=600)
    checked = subprocess.run(
        [str(environment.executable), "-I", "-B", "-c",
         "import packaging, tomli_w, truststore; from ruamel.yaml import YAML"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if checked.returncode:
        raise InstallError("pm-runtime", f"dependency validation failed: {checked.stderr.strip()}")
    return environment.executable

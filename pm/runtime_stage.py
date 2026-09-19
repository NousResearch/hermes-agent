"""One locked dependency builder for PM workers and packaged runtimes."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from pm.package import InstallError


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

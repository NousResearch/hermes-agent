"""One locked dependency builder for PM workers and packaged runtimes."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from pm.package import InstallError

#: uv settings that move the *default* index. UV_INDEX and UV_EXTRA_INDEX_URL add
#: an index alongside it, so they cannot desync a lock that records its registry.
_DEFAULT_INDEX_KEYS = ("UV_INDEX_URL", "UV_DEFAULT_INDEX")

#: A package source as uv writes it into ``uv.lock``: ``source = { registry = "..." }``.
_REGISTRY = re.compile(r'registry = "([^"]*)"')


def _index_identity(url: str) -> tuple[str, str, int, str]:
    """Registry identity as uv compares it: scheme, host, port, path, no credentials."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return (scheme, (parts.hostname or "").lower(),
            parts.port or (443 if scheme == "https" else 80), parts.path.rstrip("/"))


def _configured_default_index(env: dict[str, str]) -> str | None:
    """The default index ``runtime_environment`` bridged into uv, if any."""
    for key in _DEFAULT_INDEX_KEYS:
        value = (env.get(key) or "").strip()
        if value:
            return value
    return None


def _staged_registry(snapshot: Path) -> str | None:
    """The registry the staged lock records, when it records exactly one.

    Read as text rather than TOML: bootstrap imports this module before PM selects
    its own Python, so ``tomllib`` is out of reach there (Docker 3.10, historical
    Windows updaters — ``tests/pm/test_bootstrap_import_closure.py``). uv writes one
    ``source = { registry = "..." }`` line per package; a lock in any other shape
    reads as no single registry and leaves staging exactly as it is today.
    """
    try:
        text = (snapshot / "uv.lock").read_text(encoding="utf-8")
    except OSError:
        return None
    registries = {match.group(1) for match in _REGISTRY.finditer(text) if match.group(1)}
    return registries.pop() if len(registries) == 1 else None


def _needs_relock(env: dict[str, str], snapshot: Path) -> bool:
    """True when the default index in force is not the one the staged lock records.

    ``runtime_environment`` bridges a configured pip mirror into ``UV_INDEX_URL`` so
    dependencies resolve behind private mirrors (#95608), but uv treats the index a
    lock was resolved from as part of its identity. The upstream lock records
    ``pypi.org`` for every entry, so against a mirror it is rejected as stale:
    ``sync --locked`` aborts before installing anything (#122112), and where only
    the mirror is reachable ``--frozen`` fails too, fetching the recorded
    ``files.pythonhosted.org`` URLs (#123132).
    """
    index = _configured_default_index(env)
    registry = _staged_registry(snapshot)
    if not index or registry is None:
        return False
    return _index_identity(index) != _index_identity(registry)


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
            if _needs_relock(env, snapshot):
                # Re-record the snapshot's sources against the index actually in
                # force — the caller-owned scratch workspace exists for exactly
                # this — and keep verifying with --locked afterwards. No
                # --upgrade, so uv keeps every locked version and hash: only
                # where the artifacts are fetched from changes (#123132).
                environment.lock(snapshot)
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

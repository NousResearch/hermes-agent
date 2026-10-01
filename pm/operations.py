"""Python dependency operations executed by the isolated package manager.

Callers describe a dependency graph and its lifetime, never a uv command.
Only the private environment engine knows how to obtain or invoke uv.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from pm.package import InstallError


def _require_install_allowed(explicit: bool) -> None:
    from pm.install import _refuse_lazy, lazy_installs_allowed

    if not explicit and not lazy_installs_allowed():
        raise _refuse_lazy("venv", "Python dependency operation requires an explicit request")


def build_environment(
    *, source: Path, out: Path, python: Path | None = None,
    cache: Path | None = None, env: Mapping[str, str] | None = None,
    extras: Sequence[str] = (), groups: Sequence[str] = (),
    all_extras: bool = False, no_install_project: bool = False,
    frozen: bool = True, sealed: bool = False, offline: bool = False,
    explicit: bool = False, timeout: int = 1800,
) -> Path:
    """Build and validate a fresh destination; never overwrite an existing tree.

    A build has no profile/plugin discovery or application selection side effects.
    Failure removes only the destination exclusively created by this invocation.
    Sealed builds prune only the .pth files that refer to build-time state.
    """
    from pm.environment import _fresh_build, managed_environment
    from pm.native_build import source_build_environment

    source, out = Path(source).absolute(), Path(out).absolute()
    if not (source / "pyproject.toml").is_file():
        raise InstallError("venv", f"project manifest is missing: {source}")
    if frozen and not (source / "uv.lock").is_file():
        raise InstallError("venv", f"frozen build requires a lock: {source / 'uv.lock'}")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"environment destination already exists: {out}")
    _require_install_allowed(explicit)
    # A caller-supplied env is already the build environment (bundle staging
    # prepares its own, shared with its Node builds).
    if env is None:
        env = source_build_environment(source)
    environment = managed_environment(
        out, python=Path(python) if python is not None else None,
        cache=Path(cache) if cache is not None else None, env=env,
        offline=offline, explicit=explicit, output=sys.stderr,
    )
    with _fresh_build(environment, sealed=sealed):
        environment.sync(source, extras=extras, groups=groups, all_extras=all_extras,
                         no_install_project=no_install_project, frozen=frozen, timeout=timeout)
    return environment.executable


def lock_project(
    source: Path, *, upgrade: bool = False, python: Path | None = None,
    env: Mapping[str, str] | None = None, cache: Path | None = None,
    offline: bool = False, explicit: bool = False,
) -> None:
    """Refresh the project's resolution without creating or selecting a venv."""
    from pm.environment import managed_environment

    source = Path(source).absolute()
    if not (source / "pyproject.toml").is_file():
        raise InstallError("venv", f"project manifest is missing: {source}")
    _require_install_allowed(explicit)
    environment = managed_environment(
        source / ".venv", python=Path(python) if python is not None else None,
        cache=Path(cache) if cache is not None else None, env=env,
        offline=offline, explicit=explicit, output=sys.stderr,
    )
    environment.lock(source, upgrade=upgrade)


def stage_manager_runtime(
    *, python: Path, destination: Path, project: Path | None = None,
    offline: bool = False, wheelhouse: Path | None = None, cache: Path | None = None,
) -> Path:
    """Stage PM's independent dependency graph using its private toolchain."""
    from pm._uv import _toolchain
    from pm.runtime_stage import stage_runtime

    destination, python = Path(destination).absolute(), Path(python).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"environment destination already exists: {destination}")
    # Tools are staged before this bootstrap: acquiring them here would need
    # the TLS dependencies in the very runtime we are constructing.
    tools = _toolchain(realize=False)
    if tools is None:
        raise InstallError("pm-runtime", "stage pinned tools before building the manager runtime")
    destination.mkdir(parents=True)
    try:
        executable = stage_runtime(tools[0], python, destination,
                                   project=Path(project) if project is not None else None,
                                   offline=offline, wheelhouse=Path(wheelhouse) if wheelhouse is not None else None,
                                   cache=Path(cache) if cache is not None else None)
        from pm.lock import _write
        _write(destination / "pm-runtime.json", {})
        return executable
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _environment_root(name: str, root: Path | None) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise ValueError("environment name must be a simple package name")
    if root is not None:
        return Path(root).absolute()
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "environments" / name


def _python(environment: Path) -> Path:
    from pm.environments import venv_python

    return venv_python(environment)


def _selection(root: Path) -> dict:
    try:
        record = json.loads((root / "active.json").read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise InstallError("venv", f"cannot read environment selection: {root}") from exc
    if (not isinstance(record, dict) or not isinstance(record.get("generation"), str)
            or not re.fullmatch(r"gen-[0-9a-f]{32}", record["generation"])):
        raise InstallError("venv", f"invalid environment selection: {root}")
    return record


def environment_python(name: str, *, root: Path | None = None) -> Path | None:
    """Read the selected interpreter without acquiring tools or writing state."""
    root = _environment_root(name, root)
    selected = _selection(root)
    if not selected:
        return None
    environment = root / selected["generation"] / "venv"
    if not environment.resolve().is_relative_to(root.resolve()):
        raise InstallError(name, "selected environment escapes its root")
    python = _python(environment)
    return python if python.is_file() and (environment / "pyvenv.cfg").is_file() else None


def _tool(python: Path, executable: str) -> Path | None:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", executable):
        raise ValueError("tool executable must be a filename, not a path")
    directory = python.parent
    suffixes = (".exe", ".cmd", "") if os.name == "nt" else ("",)
    return next((directory / (executable + suffix) for suffix in suffixes
                 if (directory / (executable + suffix)).is_file()), None)


def python_tool(name: str, executable: str, *, root: Path | None = None) -> Path | None:
    """Return a selected tool entrypoint, never an installer executable."""
    python = environment_python(name, root=root)
    return _tool(python, executable) if python is not None else None


def _requirements(requirements: Sequence[str]) -> list[str]:
    from packaging.requirements import InvalidRequirement, Requirement

    if isinstance(requirements, str) or not requirements:
        raise ValueError("an isolated environment needs a nonempty sequence of requirements")
    try:
        return sorted({str(Requirement(requirement)) for requirement in requirements})
    except (InvalidRequirement, TypeError) as exc:
        raise ValueError(f"invalid environment requirement: {exc}") from exc


def ensure_environment(
    name: str, requirements: Sequence[str], *, root: Path | None = None,
    explicit: bool = False, timeout: int = 1800,
    executable: str | None = None,
) -> Path:
    """Make an isolated dependency set current, then atomically select it.

    An optional tool entrypoint is validated before publication, not afterwards.
    """
    root = _environment_root(name, root)
    requirements = _requirements(requirements)
    if executable is not None:
        _tool(Path("unused/python"), executable)  # Validate before any write.

    def build(generation: Path, base_python: Path) -> Path:
        (generation / "pyproject.toml").write_text(
            '[project]\nname = "hermes-side-environment"\nversion = "0"\n'
            'requires-python = ">=3.11"\ndependencies = '
            + json.dumps(requirements) + '\n[tool.uv]\npackage = false\n', encoding="utf-8",
        )
        previous = _selection(root)
        if previous:
            seed = root / previous["generation"] / "uv.lock"
            if seed.is_file():
                shutil.copyfile(seed, generation / "uv.lock")
        return build_environment(source=generation, out=generation / "venv", python=base_python,
                                 frozen=False, explicit=explicit, timeout=timeout)

    return _ensure_generation(name, root, {"requirements": requirements}, build,
                              record={"requirements": requirements}, explicit=explicit,
                              executable=executable)


def ensure_project_environment(
    name: str, project: Path, *, extras: Sequence[str] = (), groups: Sequence[str] = (),
    root: Path | None = None, explicit: bool = False, timeout: int = 1800,
) -> Path:
    """Make an isolated environment of a project's LOCKED dependencies current.

    The project itself is not installed, and nothing reaches PM facts or the
    selected application generation: this serves side environments such as the
    test suite's. The identity covers the lock and manifest bytes, so any edit
    that can change the resolved set selects a fresh generation.
    """
    root = _environment_root(name, root)
    project = Path(project).absolute()
    extras, groups = sorted(set(extras)), sorted(set(groups))
    manifests = {}
    for manifest in ("pyproject.toml", "uv.lock"):
        try:
            manifests[manifest] = hashlib.sha256((project / manifest).read_bytes()).hexdigest()
        except FileNotFoundError as exc:
            raise InstallError("venv", f"locked project environment needs {project / manifest}") from exc

    def build(generation: Path, base_python: Path) -> Path:
        return build_environment(source=project, out=generation / "venv", python=base_python,
                                 extras=extras, groups=groups, no_install_project=True,
                                 frozen=True, explicit=explicit, timeout=timeout)

    return _ensure_generation(name, root, {"manifests": manifests, "extras": extras, "groups": groups},
                              build, record={"extras": extras, "groups": groups}, explicit=explicit)


def _ensure_generation(
    name: str, root: Path, inputs: dict, build: Callable[[Path, Path], Path], *, record: dict, explicit: bool,
    executable: str | None = None,
) -> Path:
    """Select the generation whose inputs match, building one only when none does.

    Build at the final path: Windows launchers and scripts embed that path.
    The prior generation survives both successful replacement and failed builds.
    """
    from pm.filesystem import lock_fd
    from pm._uv import _toolchain
    from pm.install import _refuse_lazy, lazy_installs_allowed
    from pm.lock import Lockfile, _write
    from pm import paths
    from pm.store import current_target

    lock = Lockfile(paths.lockfile_path())
    target = current_target()
    inputs = {**inputs, "python": lock.version("python"), "target": target,
              "artifacts": [item["sha256"] for item in lock.artifacts("python", target)]}

    def selected_python() -> Path | None:
        tools = _toolchain(realize=False)
        return tools[1].resolve() if tools is not None else None

    def identity(base_python: Path) -> str:
        # The same pinned artifact can live in different stores. A venv's
        # pyvenv.cfg keeps the original interpreter path, not just its version.
        return hashlib.sha256(json.dumps({**inputs, "interpreter": str(base_python)},
                                         sort_keys=True).encode()).hexdigest()

    def current(base_python: Path | None) -> Path | None:
        if base_python is None:
            return None
        selected = _selection(root)
        python = environment_python(name, root=root)
        if (selected.get("inputs") == identity(base_python) and python is not None
                and (executable is None or _tool(python, executable) is not None)):
            return python
        return None

    existing = current(selected_python())
    if existing is not None:
        return existing
    if not explicit and not lazy_installs_allowed():
        raise _refuse_lazy(name, "isolated Python environment is missing or outdated")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".install.lock").open("a+b") as mutex:
        lock_fd(mutex.fileno(), wait=True)
        base_python = selected_python()
        existing = current(base_python)
        if existing is not None:
            return existing
        if base_python is None:
            tools = _toolchain(explicit=explicit)
            if tools is None:
                raise InstallError(name, "PM's pinned toolchain is unavailable")
            base_python = tools[1].resolve()
        generation = root / f"gen-{uuid.uuid4().hex}"
        generation.mkdir()
        try:
            python = build(generation, base_python)
            if executable is not None and _tool(python, executable) is None:
                raise InstallError(name, f"installed requirements do not provide {executable!r}")
            from pm.environments import flush_before_selecting
            flush_before_selecting()
            _write(root / "active.json", {"generation": generation.name,
                                           "inputs": identity(base_python), **record})
        except BaseException:
            shutil.rmtree(generation, ignore_errors=True)
            raise
    return python


def ensure_python_tool(
    name: str, requirements: Sequence[str], executable: str, *,
    root: Path | None = None, explicit: bool = False, timeout: int = 1800,
) -> Path:
    python = ensure_environment(name, requirements, root=root, explicit=explicit,
                                timeout=timeout, executable=executable)
    tool = _tool(python, executable)
    assert tool is not None  # Validated under the publication lock above.
    return tool

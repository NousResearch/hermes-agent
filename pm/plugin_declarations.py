"""Read plugin build declarations without importing CLI configuration or plugin code."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib


def native_manifest_file(plugin_dir: Path) -> Path | None:
    for path in (plugin_dir / "plugin.yaml", plugin_dir / "plugin.yml"):
        try:
            path.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"Could not inspect plugin manifest {path}: {exc}") from exc
        return path
    return None


def read_native_manifest(path: Path) -> dict:
    import hermes_yaml as yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read plugin manifest {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Plugin manifest must be a mapping: {path}")
    return data


def manifest_version_error(manifest: dict, name: str) -> str | None:
    from hermes_cli.plugins_manifest import SUPPORTED_MANIFEST_VERSION, requires_hermes_error

    reason = requires_hermes_error(manifest)
    if reason:
        return f"Plugin '{name}' {reason}"
    version = manifest.get("manifest_version")
    if version is None:
        return None
    try:
        parsed = int(version)
    except (TypeError, ValueError):
        return f"Plugin '{name}' has invalid manifest_version '{version}' (expected an integer)."
    if parsed > SUPPORTED_MANIFEST_VERSION:
        return (f"Plugin '{name}' requires manifest_version {version}, "
                f"but this installer only supports up to {SUPPORTED_MANIFEST_VERSION}.")
    return None


@dataclass(frozen=True)
class PythonDeclaration:
    files: tuple[Path, ...]
    pyproject: Path | None
    requirements: tuple[str, ...]
    manifest: dict
    requires_python: str | None = None
    # The pyproject itself installs something: the plugin as a package, or extras/dependency groups
    # (``uv sync --all-packages`` installs a member's default groups as well as its dependencies).
    pyproject_installs: bool = False

    @property
    def external(self) -> bool:
        return str(self.manifest.get("python_runtime") or "").strip().lower() == "external"

    @property
    def install_requirements(self) -> tuple[str, ...]:
        return () if self.external else applicable_requirements(self.requirements)

    @property
    def installs_packages(self) -> bool:
        """Admission would put packages into the shared environment: what dependency consent asks
        about and what makes the plugin a workspace member. ``dependencies = []`` installs nothing,
        so like no pyproject it asks no consent and stays out of the environment stamp, which hashes
        a member's whole tree (each enable or update would build a generation for nothing)."""
        return not self.external and (bool(self.install_requirements) or self.pyproject_installs)

    def python_error(self, version: str) -> str | None:
        """Why this member cannot join a workspace built for *version*; uv intersects every
        member's requires-python, so one excluding member fails the whole lock."""
        from packaging.specifiers import SpecifierSet

        if not self.installs_packages or not self.requires_python:
            return None
        if SpecifierSet(self.requires_python).contains(version, prereleases=True):
            return None
        return f"requires Python {self.requires_python}, but Hermes runs on Python {version}"

    def __post_init__(self) -> None:
        if not self.external:
            applicable_requirements(self.requirements)


def applicable_requirements(specs: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Index requirements only; let uv evaluate markers for the target interpreter.

    The application checkout supplies Hermes itself. Installing it from an index
    would replace that checkout; direct URLs bypass the reviewed package source.
    Markers must survive snapshots built for a different Python or platform.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    return tuple(spec for spec in specs
                 if not (req := Requirement(spec)).url and canonicalize_name(req.name) != "hermes-agent")


def unsupported_requirements(specs: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    from packaging.requirements import Requirement

    return tuple(spec for spec in specs if Requirement(spec).url)


def read_python_declaration(plugin_dir: Path) -> PythonDeclaration:
    """One effective surface for consent, membership, builds and currency.

    A pyproject with a ``[project]`` table owns packaging. Tooling-only
    pyprojects (for example, a plugin's Ruff configuration) remain legacy
    manifest plugins and must not become uv workspace members. Both YAML
    suffixes and both dependency aliases have identical semantics. Reading
    never initializes a home, discovers code, or depends on the caller's CLI
    import state.
    """
    native = native_manifest_file(plugin_dir)
    files = [native] if native is not None else []
    manifest = read_native_manifest(native) if native is not None else {}
    if native is None:
        portable = plugin_dir / "plugin.json"
        try:
            portable.lstat()
        except FileNotFoundError:
            pass
        else:
            from hermes_cli.agent_plugins import read_agent_plugin_manifest

            manifest, _diagnostics = read_agent_plugin_manifest(plugin_dir)
            files.append(portable)
    project = plugin_dir / "pyproject.toml"
    try:
        text = project.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        text = None
    if text is not None:
        files.append(project)
        if "GENERATED by pm" not in text:
            document = tomllib.loads(text)
            metadata = document.get("project")
            if isinstance(metadata, Mapping):
                specs = metadata.get("dependencies", [])
                if not isinstance(specs, list) or any(not isinstance(v, str) for v in specs):
                    raise ValueError(f"invalid project.dependencies: {project}")
                requires_python = metadata.get("requires-python")
                if requires_python is not None and not isinstance(requires_python, str):
                    raise ValueError(f"invalid project.requires-python: {project}")
                uv = document.get("tool", {}).get("uv", {})
                installs = ("build-system" in document or uv.get("package") is True
                            or any(metadata.get("optional-dependencies", {}).values())
                            or any(document.get("dependency-groups", {}).values())
                            or bool(uv.get("dev-dependencies")))
                return PythonDeclaration(tuple(files), project, tuple(specs), manifest, requires_python, installs)
    specs = []
    for key in ("pip_dependencies", "python_dependencies"):
        values = manifest.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"invalid {key}: {native}")
        specs.extend(values)
    return PythonDeclaration(tuple(files), None, tuple(dict.fromkeys(specs)), manifest)


def _normalize_extra(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class CoreBaseClosure:
    """Packages and package extras guaranteed by Hermes core's base environment."""

    versions: Mapping[str, frozenset[str]]
    extras: frozenset[tuple[str, str]]
    extra_targets: Mapping[tuple[str, str], frozenset[str]] = ()  # type: ignore[assignment]


_CORE_LOCK_CLOSURE_CACHE: dict[tuple, CoreBaseClosure | None] = {}
_ROOT_SOURCES = ({"editable": "."}, {"virtual": "."}, {"directory": "."})


def _eval_marker(marker_text: str | None, env: Mapping[str, str]) -> bool:
    """Evaluate a PEP 508 marker conservatively: any parse/eval error returns False."""
    if not marker_text:
        return True
    from packaging.markers import InvalidMarker, Marker, UndefinedComparison, UndefinedEnvironmentName

    try:
        return bool(Marker(marker_text).evaluate(env))
    except (InvalidMarker, UndefinedComparison, UndefinedEnvironmentName):
        return False


def _root_project_name(project_root: Path) -> str | None:
    from packaging.utils import canonicalize_name

    try:
        doc = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    name = doc.get("project", {}).get("name")
    return canonicalize_name(name) if isinstance(name, str) and name.strip() else None


def read_core_base_closure(
    project_root: Path,
    *,
    target_env: Mapping[str, str] | None = None,
) -> CoreBaseClosure | None:
    """Resolve Hermes core's guaranteed base dependency closure from ``project_root / 'uv.lock'``.

    Walks ONLY the root package's base ``dependencies`` (never ``optional-dependencies``,
    ``dev-dependencies``, or any plugin generation), evaluating PEP 508 markers against
    *target_env*. Returns ``None`` if the lockfile is missing, unreadable, malformed, or does
    not cover *target_env*, so callers fail conservatively.
    """
    from packaging.markers import default_environment
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    lock_path = Path(project_root).resolve() / "uv.lock"
    try:
        stat = lock_path.stat()
    except OSError:
        return None
    env = dict(default_environment()) if target_env is None else dict(target_env)
    cache_key = (lock_path, stat.st_mtime_ns, stat.st_size, tuple(sorted(env.items())))
    if cache_key in _CORE_LOCK_CLOSURE_CACHE:
        return _CORE_LOCK_CLOSURE_CACHE[cache_key]

    result = _parse_core_base_closure(lock_path, Path(project_root).resolve(), env, canonicalize_name, Version, InvalidVersion)
    if len(_CORE_LOCK_CLOSURE_CACHE) >= 32:
        _CORE_LOCK_CLOSURE_CACHE.clear()
    _CORE_LOCK_CLOSURE_CACHE[cache_key] = result
    return result


def _select_lock_root(packages: list, project_root: Path, canonicalize_name) -> dict | None:
    roots = [pkg for pkg in packages if isinstance(pkg, dict) and pkg.get("source") in _ROOT_SOURCES]
    if len(roots) > 1:
        expected_name = _root_project_name(project_root)
        if expected_name:
            roots = [pkg for pkg in roots if canonicalize_name(str(pkg.get("name", ""))) == expected_name]
    return roots[0] if len(roots) == 1 else None


def _index_lock_packages(packages: list, env: Mapping[str, str], canonicalize_name) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    for pkg in packages:
        if not isinstance(pkg, dict) or not isinstance(pkg.get("name"), str):
            continue
        res_markers = pkg.get("resolution-markers")
        if isinstance(res_markers, list) and res_markers:
            if not any(_eval_marker(marker, env) for marker in res_markers if isinstance(marker, str)):
                continue
        by_name.setdefault(canonicalize_name(pkg["name"]), []).append(pkg)
    return by_name


def _match_lock_edge(
    edge: object,
    by_name: Mapping[str, list[dict]],
    env: Mapping[str, str],
    canonicalize_name,
) -> tuple[list[dict], tuple[str, ...]]:
    if not isinstance(edge, dict) or not isinstance(edge.get("name"), str):
        return [], ()
    if not _eval_marker(edge.get("marker"), env):
        return [], ()
    canon = canonicalize_name(edge["name"])
    matched = [
        candidate for candidate in by_name.get(canon, ())
        if ("version" not in edge or candidate.get("version") == edge["version"])
        and ("source" not in edge or candidate.get("source") == edge["source"])
    ]
    raw_extras = edge.get("extra") or ()
    req_extras = tuple(_normalize_extra(str(item)) for item in raw_extras) if isinstance(raw_extras, list) else ()
    return matched, req_extras


def _enqueue_edges(
    edges: Iterable[object],
    queue: list[tuple[dict, tuple[str, ...]]],
    by_name: Mapping[str, list[dict]],
    env: Mapping[str, str],
    canonicalize_name,
    targets: set[str] | None = None,
) -> None:
    for edge in edges:
        matched, req_extras = _match_lock_edge(edge, by_name, env, canonicalize_name)
        for node in matched:
            if targets is not None:
                targets.add(canonicalize_name(node["name"]))
            queue.append((node, req_extras))


def _parse_core_base_closure(
    lock_path: Path,
    project_root: Path,
    env: Mapping[str, str],
    canonicalize_name,
    Version,
    InvalidVersion,
) -> CoreBaseClosure | None:
    try:
        data = tomllib.loads(lock_path.read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    supported = data.get("supported-markers")
    if isinstance(supported, list) and supported:
        if not any(_eval_marker(marker, env) for marker in supported if isinstance(marker, str)):
            return None

    packages = data.get("package")
    if not isinstance(packages, list):
        return None
    root = _select_lock_root(packages, project_root, canonicalize_name)
    if root is None:
        return None

    by_name = _index_lock_packages(packages, env, canonicalize_name)
    versions: dict[str, set[str]] = {}
    extras: set[tuple[str, str]] = set()
    extra_targets: dict[tuple[str, str], set[str]] = {}
    visited_nodes: set[int] = set()
    visited_extras: set[tuple[int, str]] = set()

    queue: list[tuple[dict, tuple[str, ...]]] = []
    _enqueue_edges(root.get("dependencies", ()), queue, by_name, env, canonicalize_name)

    while queue:
        node, req_extras = queue.pop()
        canon = canonicalize_name(node["name"])
        version = node.get("version")
        if isinstance(version, str):
            try:
                Version(version)
            except InvalidVersion:
                pass
            else:
                versions.setdefault(canon, set()).add(version)

        node_id = id(node)
        if node_id not in visited_nodes:
            visited_nodes.add(node_id)
            _enqueue_edges(node.get("dependencies", ()), queue, by_name, env, canonicalize_name)

        opt_deps = node.get("optional-dependencies")
        opt_map = ({_normalize_extra(str(k)): v for k, v in opt_deps.items()}
                   if isinstance(opt_deps, Mapping) else {})
        for extra in req_extras:
            extras.add((canon, extra))
            if (node_id, extra) in visited_extras:
                continue
            visited_extras.add((node_id, extra))
            targets = extra_targets.setdefault((canon, extra), set())
            _enqueue_edges(opt_map.get(extra, ()), queue, by_name, env, canonicalize_name, targets=targets)

    return CoreBaseClosure(
        versions={name: frozenset(vers) for name, vers in versions.items()},
        extras=frozenset(extras),
        extra_targets={key: frozenset(vals) for key, vals in extra_targets.items()},
    )


def _read_site_packages_distributions(site_dir: Path) -> dict[str, frozenset[str]] | None:
    """Read installed distribution versions from a single ``site-packages`` directory.

    Returns ``None`` if *site_dir* is missing or unreadable so callers fail conservatively.
    """
    import importlib.metadata
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    try:
        if not site_dir.is_dir():
            return None
        found: dict[str, set[str]] = {}
        for dist in importlib.metadata.distributions(path=[str(site_dir)]):
            raw_name = dist.metadata.get("Name") if dist.metadata is not None else None
            raw_version = dist.version
            if not raw_name or not raw_version:
                continue
            try:
                Version(raw_version)
            except InvalidVersion:
                continue
            found.setdefault(canonicalize_name(raw_name), set()).add(raw_version)
    except OSError:
        return None
    return {name: frozenset(vers) for name, vers in found.items()}


def _intersect_closure_with_site(
    lock_closure: CoreBaseClosure,
    site_dists: Mapping[str, frozenset[str]],
) -> CoreBaseClosure:
    """Keep only packages proven present in BOTH core's lock base closure AND *site_dists*."""
    from packaging.version import Version

    versions: dict[str, frozenset[str]] = {}
    for name, lock_vers in lock_closure.versions.items():
        site_vers = site_dists.get(name)
        if not site_vers:
            continue
        site_parsed = {Version(v) for v in site_vers}
        matched = frozenset(v for v in lock_vers if Version(v) in site_parsed)
        if matched:
            versions[name] = matched

    extras = frozenset(
        (name, extra)
        for name, extra in lock_closure.extras
        if name in versions
        and all(dep in versions for dep in lock_closure.extra_targets.get((name, extra), ()))
    )
    return CoreBaseClosure(versions=versions, extras=extras, extra_targets=lock_closure.extra_targets)


def _resolve_base_closure(
    *,
    project_root: Path | None = None,
    base_site_packages: Path | None = None,
    target_env: Mapping[str, str],
) -> CoreBaseClosure | None:
    """Resolve the canonical core base closure for dependency satisfaction checks.

    - Explicit *base_site_packages*: read directly (and intersect with *project_root*'s lock if
      explicitly provided). Extras stay empty when checking site-packages alone (conservative).
    - Sealed payload (``payload_venv(root)`` exists): intersect ``root / 'uv.lock'`` base closure
      with ``payload_venv``'s ``site-packages`` (neither ever contains plugin-only packages, and
      ``uv.lock``'s base closure excludes optional extras).
    - Source checkout (no ``payload_venv``): use ``root / 'uv.lock'`` base closure directly, never
      a mutable plugin generation or ad-hoc developer venv.
    """
    if base_site_packages is not None:
        site_dists = _read_site_packages_distributions(Path(base_site_packages))
        if site_dists is None:
            return None
        if project_root is not None:
            lock_closure = read_core_base_closure(Path(project_root), target_env=target_env)
            if lock_closure is None:
                return None
            return _intersect_closure_with_site(lock_closure, site_dists)
        return CoreBaseClosure(versions=site_dists, extras=frozenset(), extra_targets={})

    from pm import paths
    from pm.environments import payload_venv, site_packages

    root = paths.repo_root() if project_root is None else Path(project_root)
    lock_closure = read_core_base_closure(root, target_env=target_env)
    if lock_closure is None:
        return None
    payload = payload_venv(root)
    if payload is None:
        return lock_closure
    site_dists = _read_site_packages_distributions(site_packages(payload))
    if site_dists is None:
        return None
    return _intersect_closure_with_site(lock_closure, site_dists)


def _pyproject_uv_sources(pyproject: Path | None) -> frozenset[str] | None:
    """Canonical names overridden in ``[tool.uv.sources]``; ``None`` if unreadable/malformed."""
    if pyproject is None:
        return frozenset()
    from packaging.utils import canonicalize_name

    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    sources = document.get("tool", {}).get("uv", {}).get("sources")
    if sources is None:
        return frozenset()
    if not isinstance(sources, Mapping):
        return None
    return frozenset(canonicalize_name(str(name)) for name in sources)


def _pyproject_install_triggers(pyproject: Path | None) -> tuple[str, ...]:
    """Human-readable names of the ``pyproject.toml`` features that set ``pyproject_installs``."""
    if pyproject is None:
        return ()
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError):
        return ("pyproject.toml",)
    metadata = document.get("project", {})
    uv = document.get("tool", {}).get("uv", {})
    triggers: list[str] = []
    if "build-system" in document:
        triggers.append("[build-system]")
    if uv.get("package") is True:
        triggers.append("tool.uv.package = true")
    if isinstance(metadata, Mapping) and any(metadata.get("optional-dependencies", {}).values()):
        triggers.append("[project.optional-dependencies]")
    if any(document.get("dependency-groups", {}).values()):
        triggers.append("[dependency-groups]")
    if uv.get("dev-dependencies"):
        triggers.append("tool.uv.dev-dependencies")
    return tuple(triggers)


def unsatisfied_requirements(
    declaration: PythonDeclaration,
    *,
    project_root: Path | None = None,
    base_site_packages: Path | None = None,
    target_env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return declared install requirements NOT guaranteed by Hermes core's base environment.

    Fails conservatively: unparseable specs, direct URLs, ``[tool.uv.sources]`` overrides,
    unverified extras, missing/incompatible versions, unevaluable markers, or an unreadable base
    environment all keep the requirement unsatisfied so PM still resolves it in a workspace.
    """
    if declaration.external:
        return ()
    specs = declaration.install_requirements
    if not specs:
        return ()

    from packaging.markers import UndefinedComparison, UndefinedEnvironmentName, default_environment
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    env = dict(default_environment()) if target_env is None else dict(target_env)
    closure = _resolve_base_closure(
        project_root=project_root,
        base_site_packages=base_site_packages,
        target_env=env,
    )
    if closure is None:
        return specs

    uv_sources = _pyproject_uv_sources(declaration.pyproject)
    if uv_sources is None:
        return specs

    unsatisfied: list[str] = []
    for spec in specs:
        try:
            req = Requirement(spec)
        except InvalidRequirement:
            unsatisfied.append(spec)
            continue
        if req.url:
            unsatisfied.append(spec)
            continue
        if req.marker is not None:
            try:
                if not req.marker.evaluate(env):
                    continue
            except (UndefinedComparison, UndefinedEnvironmentName):
                unsatisfied.append(spec)
                continue
        name = canonicalize_name(req.name)
        if name in uv_sources:
            unsatisfied.append(spec)
            continue
        installed_versions = closure.versions.get(name)
        if not installed_versions:
            unsatisfied.append(spec)
            continue
        if not all(req.specifier.contains(Version(ver)) for ver in installed_versions):
            unsatisfied.append(spec)
            continue
        if req.extras:
            requested_extras = {_normalize_extra(extra) for extra in req.extras}
            if not all((name, extra) in closure.extras for extra in requested_extras):
                unsatisfied.append(spec)
                continue
    return tuple(unsatisfied)


def needs_workspace_member(
    declaration: PythonDeclaration,
    *,
    project_root: Path | None = None,
    base_site_packages: Path | None = None,
    target_env: Mapping[str, str] | None = None,
) -> bool:
    """Whether *declaration* requires PM to add the plugin as a uv workspace member.

    False for external plugins and for pure-metadata plugins whose declared requirements are
    already guaranteed by Hermes core's base environment (no build-system/extras/groups,
    compatible ``requires-python``, and zero unsatisfied dependencies).
    """
    if declaration.external or not declaration.installs_packages:
        return False
    if declaration.pyproject_installs:
        return True
    if declaration.requires_python:
        from packaging.markers import default_environment
        from packaging.specifiers import InvalidSpecifier, SpecifierSet

        env = default_environment() if target_env is None else target_env
        py_version = env.get("python_full_version") or env.get("python_version")
        if not py_version:
            return True
        try:
            if not SpecifierSet(declaration.requires_python).contains(py_version, prereleases=True):
                return True
        except InvalidSpecifier:
            return True
    return bool(unsatisfied_requirements(
        declaration,
        project_root=project_root,
        base_site_packages=base_site_packages,
        target_env=target_env,
    ))


# Representative desktop targets (Python 3.14, Hermes's pinned runtime) used when linting
# default: true catalog entries on a single CI host so platform-gated deps on Windows/macOS
# are caught alongside Linux.
_DEFAULT_PLUGIN_CHECK_TARGETS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("linux", "linux-x64", {
        "implementation_name": "cpython",
        "implementation_version": "3.14.0",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "6.8.0",
        "platform_system": "Linux",
        "platform_version": "",
        "python_full_version": "3.14.0",
        "python_version": "3.14",
        "sys_platform": "linux",
    }),
    ("macos", "darwin-arm64", {
        "implementation_name": "cpython",
        "implementation_version": "3.14.0",
        "os_name": "posix",
        "platform_machine": "arm64",
        "platform_python_implementation": "CPython",
        "platform_release": "24.0.0",
        "platform_system": "Darwin",
        "platform_version": "",
        "python_full_version": "3.14.0",
        "python_version": "3.14",
        "sys_platform": "darwin",
    }),
    ("windows", "win32-x64", {
        "implementation_name": "cpython",
        "implementation_version": "3.14.0",
        "os_name": "nt",
        "platform_machine": "AMD64",
        "platform_python_implementation": "CPython",
        "platform_release": "11",
        "platform_system": "Windows",
        "platform_version": "10.0.26100",
        "python_full_version": "3.14.0",
        "python_version": "3.14",
        "sys_platform": "win32",
    }),
)

_DEFAULT_REBUILD_GUIDANCE = (
    "default: true plugins install automatically on first launch and updates; triggering a venv "
    "rebuild forces `uv lock`/`uv sync` and a ~1 GB `uv-cache` seed copy on Windows (minutes on "
    "first launch). How to avoid triggering a rebuild: "
    "(1) drop `[build-system]` / `tool.uv.package = true` if the plugin does not need wheel installation; "
    "(2) use dependencies already shipped in Hermes core (e.g. `httpx`, `PyJWT[crypto]`, "
    "`cryptography`, `fastapi`, `pydantic`) instead of introducing new third-party packages; or "
    "(3) if the dependency is genuinely required for a default plugin, add it to Hermes core's "
    "`pyproject.toml` first."
)


@dataclass(frozen=True)
class DefaultPluginRebuildCheck:
    """Result of checking whether a ``default: true`` plugin would trigger a PM venv rebuild."""

    triggers_rebuild: bool
    reason: str
    unsatisfied_requirements: list[str]
    pyproject_installs: bool
    guidance: str


def check_default_plugin_venv_rebuild(
    plugin_dir: Path,
    *,
    project_root: Path | None = None,
    base_site_packages: Path | None = None,
    target_env: Mapping[str, str] | None = None,
    platforms: Iterable[str] | None = None,
) -> DefaultPluginRebuildCheck:
    """Check whether *plugin_dir* would trigger a PM venv rebuild as a ``default: true`` plugin.

    Uses the exact same core-base-closure check as PM workspace membership
    (:func:`needs_workspace_member` / :func:`unsatisfied_requirements`). When *target_env* and
    *base_site_packages* are omitted, checks Linux, macOS, and Windows (filtered by *platforms*
    if provided) so platform-specific rebuild triggers are caught in single-OS CI.
    """
    declaration = read_python_declaration(Path(plugin_dir))
    if declaration.external or not declaration.installs_packages:
        return DefaultPluginRebuildCheck(
            triggers_rebuild=False,
            reason="",
            unsatisfied_requirements=[],
            pyproject_installs=False,
            guidance="",
        )

    if target_env is not None or base_site_packages is not None:
        from packaging.markers import default_environment

        env = dict(default_environment()) if target_env is None else dict(target_env)
        targets: list[tuple[str, Mapping[str, str]]] = [("host", env)]
    else:
        allowed = set(platforms) if platforms else None
        targets = [
            (label, env)
            for plat, label, env in _DEFAULT_PLUGIN_CHECK_TARGETS
            if not allowed or plat in allowed
        ]
        if not targets:
            targets = [(label, env) for _plat, label, env in _DEFAULT_PLUGIN_CHECK_TARGETS]

    unsat_ordered: list[str] = []
    unsat_seen: set[str] = set()
    python_errors: list[str] = []
    any_needs = False
    for _label, env in targets:
        if needs_workspace_member(
            declaration,
            project_root=project_root,
            base_site_packages=base_site_packages,
            target_env=env,
        ):
            any_needs = True
        for spec in unsatisfied_requirements(
            declaration,
            project_root=project_root,
            base_site_packages=base_site_packages,
            target_env=env,
        ):
            if spec not in unsat_seen:
                unsat_seen.add(spec)
                unsat_ordered.append(spec)
        py_ver = env.get("python_full_version") or env.get("python_version")
        if declaration.requires_python and py_ver:
            from packaging.specifiers import InvalidSpecifier, SpecifierSet

            try:
                if not SpecifierSet(declaration.requires_python).contains(py_ver, prereleases=True):
                    msg = f"requires-python {declaration.requires_python!r} excludes Python {py_ver}"
                    if msg not in python_errors:
                        python_errors.append(msg)
            except InvalidSpecifier:
                msg = f"invalid requires-python {declaration.requires_python!r}"
                if msg not in python_errors:
                    python_errors.append(msg)

    if not any_needs:
        return DefaultPluginRebuildCheck(
            triggers_rebuild=False,
            reason="",
            unsatisfied_requirements=[],
            pyproject_installs=False,
            guidance="",
        )

    reasons: list[str] = []
    if declaration.pyproject_installs:
        triggers = ", ".join(_pyproject_install_triggers(declaration.pyproject)) or "pyproject.toml"
        reasons.append(f"pyproject.toml triggers package installation ({triggers})")
    if unsat_ordered:
        reasons.append(
            "unsatisfied Python requirement(s) not in Hermes core base environment: "
            + ", ".join(unsat_ordered)
        )
    if python_errors:
        reasons.extend(python_errors)
    reason = "; ".join(reasons) or "triggers PM workspace membership"

    return DefaultPluginRebuildCheck(
        triggers_rebuild=True,
        reason=reason,
        unsatisfied_requirements=unsat_ordered,
        pyproject_installs=declaration.pyproject_installs,
        guidance=_DEFAULT_REBUILD_GUIDANCE,
    )

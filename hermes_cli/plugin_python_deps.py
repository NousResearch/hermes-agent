"""Python dependencies declared by user plugins: read, resolve against Hermes' own ranges, install,
and re-apply after ``hermes update`` rebuilds the venv.

A plugin declares deps in its ``pyproject.toml`` (``[project].dependencies``) or, without one, in
``plugin.yaml`` ``python_dependencies`` (``pip_dependencies`` is an accepted alias). ``python_runtime:
external`` in the manifest opts a plugin out: it manages its own interpreter (sidecar venv) and its tree
is never handed to the resolver.

Contract (agreed with the Mnemosyne team, Sep 2026): the union of every enabled plugin's declarations
is resolved together with Hermes' declared ranges; a candidate that has no solution is refused without
touching the live venv or disabling anything already installed. After an update, the union is
re-applied; if core moved and the union no longer resolves, non-memory plugins are dropped first and
disabled with a loud warning, because a Hermes that boots without memory reads as data loss.

Lifted in shape from ethernet8023's ``pm/plugin_declarations.py`` / ``pm/workspace.py`` (#102765).
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml
from hermes_constants import get_hermes_home
from packaging.markers import InvalidMarker, UndefinedEnvironmentName
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

logger = logging.getLogger(__name__)

_MANIFEST_FILES = ("plugin.yaml", "plugin.yml")
_DEP_KEYS = ("python_dependencies", "pip_dependencies")
EXTERNAL_RUNTIME_KEY = "python_runtime"
_CORE_DIST = "hermes-agent"
# Substrings uv/pip print only when the resolver itself proved there is no solution. Deliberately
# narrow: a fetch timeout or index outage must never be misread as a conflict.
_CONFLICT_MARKERS = (
    "no solution found",
    "conflicting requirements",
    "resolutionimpossible",
    "because only the following versions",
)


class DependencyConflict(Exception):
    """The resolver proved core + enabled plugins + candidate has no valid solution."""


class DependencyInstallError(Exception):
    """The install failed for a reason other than a proven conflict (network, build, gate)."""


@dataclass(frozen=True)
class PythonDeclaration:
    plugin_dir: Path
    name: str
    specs: tuple[str, ...]
    source: str            # "pyproject" | "manifest" | ""
    external: bool = False
    memory_provider: bool = False

    @property
    def wants_install(self) -> bool:
        return bool(self.specs) and not self.external


# ── reading ───────────────────────────────────────────────────────────────────

def _read_manifest(plugin_dir: Path) -> dict:
    for candidate in _MANIFEST_FILES:
        path = plugin_dir / candidate
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _pyproject_specs(plugin_dir: Path) -> Optional[list[str]]:
    """``[project].dependencies`` from the plugin's pyproject, or ``None`` when there is none."""
    path = plugin_dir / "pyproject.toml"
    if not path.is_file():
        return None
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    specs = document.get("project", {}).get("dependencies", [])
    if not isinstance(specs, list) or any(not isinstance(s, str) for s in specs):
        raise ValueError(f"{path}: [project].dependencies must be a list of strings")
    return specs


def _manifest_specs(manifest: dict) -> list[str]:
    specs: list[str] = []
    for key in _DEP_KEYS:
        values = manifest.get(key) or []
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"{key} must be a list of requirement strings")
        specs.extend(v.strip() for v in values if v.strip())
    return list(dict.fromkeys(specs))


def _is_memory_provider(plugin_dir: Path, manifest: dict) -> bool:
    if str(manifest.get("kind") or "").strip().lower() == "exclusive":
        return True
    init = plugin_dir / "__init__.py"
    if not init.is_file():
        return False
    head = init.read_text(encoding="utf-8", errors="replace")[:8192]
    return "register_memory_provider" in head or "MemoryProvider" in head


def read_declaration(plugin_dir: Path) -> PythonDeclaration:
    """One effective dependency surface for a plugin directory. A real pyproject owns packaging; the
    manifest list bridges plugins without one. Raises ``ValueError`` on a malformed declaration."""
    plugin_dir = Path(plugin_dir)
    manifest = _read_manifest(plugin_dir)
    name = str(manifest.get("name") or plugin_dir.name)
    external = str(manifest.get(EXTERNAL_RUNTIME_KEY) or "").strip().lower() == "external"
    pyproject = _pyproject_specs(plugin_dir)
    specs, source = (pyproject, "pyproject") if pyproject is not None else (_manifest_specs(manifest), "manifest")
    return PythonDeclaration(
        plugin_dir=plugin_dir, name=name, specs=tuple(specs), source=source if specs else "",
        external=external, memory_provider=_is_memory_provider(plugin_dir, manifest))


# ── requirement hygiene ───────────────────────────────────────────────────────

def _parse_requirement(spec: str) -> Requirement:
    try:
        return Requirement(spec)
    except InvalidRequirement as exc:
        raise ValueError(f"invalid requirement {spec!r}: {exc}") from exc


def _marker_applies(req: Requirement) -> bool:
    try:
        return req.marker is None or req.marker.evaluate()
    except (InvalidMarker, UndefinedEnvironmentName) as exc:
        raise ValueError(f"invalid marker in {req!s}: {exc}") from exc


def applicable_requirement(spec: str) -> Optional[str]:
    """``spec`` reduced to ``name[extras]specifier`` when Hermes should install it here; ``None`` when
    its environment marker excludes this platform, when it names Hermes itself (always satisfied by the
    running checkout — installing ``hermes-agent`` from an index would clobber it), or when it points
    at a URL/path (see :func:`unsupported_specs`; the resolver must only see index packages)."""
    req = _parse_requirement(spec)
    if req.url or canonicalize_name(req.name) == _CORE_DIST or not _marker_applies(req):
        return None
    extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
    return f"{req.name}{extras}{req.specifier}"


def applicable_specs(specs: Iterable[str]) -> list[str]:
    return [r for r in (applicable_requirement(s) for s in specs) if r]


def unsupported_specs(specs: Iterable[str]) -> list[str]:
    """Direct URL/path requirements: never installed by Hermes (arbitrary git/tarball sources are a
    supply-chain surface the reviewed pin does not cover); surfaced so the user can install them."""
    return [s for s in specs if _parse_requirement(s).url]


def core_constraints(project_root: Path) -> list[str]:
    """Hermes' own declared ranges (``[project].dependencies`` + every extra), as constraint lines.
    Plugins resolve inside these, so they can move transitives but never a core package out of range."""
    document = tomllib.loads((Path(project_root) / "pyproject.toml").read_text(encoding="utf-8"))
    project = document.get("project", {})
    declared: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        declared.extend(group)
    lines = []
    for spec in declared:
        try:
            reduced = applicable_requirement(spec)
        except ValueError:
            continue
        if reduced and not reduced.lower().startswith(_CORE_DIST):
            lines.append(_strip_extras(reduced))
    return sorted(set(lines))


def _strip_extras(spec: str) -> str:
    """Constraint files may not carry extras."""
    return re.sub(r"\[[^\]]*\]", "", spec, count=1)


# ── enumeration across homes ──────────────────────────────────────────────────

def _name_set(config: dict, key: str) -> set[str]:
    values = (config.get("plugins") or {}).get(key) if isinstance(config.get("plugins"), dict) else None
    return set(values) if isinstance(values, list) else set()


def _plugin_enabled(decl: PythonDeclaration, enabled: set[str], disabled: set[str]) -> bool:
    keys = {decl.name, decl.plugin_dir.name}
    return bool(keys & enabled) and not (keys & disabled)


def _read_home_config(home: Path) -> dict:
    """*home*'s effective config through the real loader, scoped to that home for the duration."""
    from hermes_cli.config import load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        return load_config_readonly()
    finally:
        reset_hermes_home_override(token)


def enabled_declarations(home: Path) -> list[PythonDeclaration]:
    """Declarations of every enabled, non-external user plugin under ``<home>/plugins`` that declares
    something. Malformed declarations are skipped with a warning (one bad plugin must not stop the
    union for everyone else)."""
    plugins_dir = Path(home) / "plugins"
    if not plugins_dir.is_dir():
        return []
    config = _read_home_config(Path(home))
    enabled, disabled = _name_set(config, "enabled"), _name_set(config, "disabled")
    found: list[PythonDeclaration] = []
    for plugin_dir in sorted(p for p in plugins_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        try:
            decl = read_declaration(plugin_dir)
        except (ValueError, OSError, tomllib.TOMLDecodeError, yaml.YAMLError) as exc:
            logger.warning("Plugin %s: dependency declaration skipped: %s", plugin_dir.name, exc)
            continue
        if decl.wants_install and _plugin_enabled(decl, enabled, disabled):
            found.append(decl)
    return found


def dependency_homes() -> list[Path]:
    """Every home whose plugins share this venv: the default home plus live named profiles."""
    from hermes_cli.profiles import profiles_to_serve
    return [home for _name, home in profiles_to_serve(multiplex=True, include_standalone=True, include_parked=True)]


def union_specs(declarations: Iterable[PythonDeclaration]) -> list[str]:
    seen: dict[str, None] = {}
    for decl in declarations:
        for spec in applicable_specs(decl.specs):
            seen.setdefault(spec, None)
    return list(seen)


# ── resolve / install ─────────────────────────────────────────────────────────

def _classify(result) -> Optional[Exception]:
    if result.ok:
        return None
    text = (result.stderr + result.stdout).lower()
    if result.blocked:
        return DependencyInstallError(result.reason)
    if any(marker in text for marker in _CONFLICT_MARKERS):
        return DependencyConflict(_tail(result.stderr or result.stdout))
    return DependencyInstallError(_tail(result.stderr or result.stdout) or "installer failed")


def _tail(text: str, limit: int = 600) -> str:
    return text.strip()[-limit:]


def resolve(specs: list[str], constraints: list[str], *, dry_run: bool, timeout: int = 600):
    """Run the shared installer ladder on *specs* under *constraints*. Raises ``DependencyConflict`` or
    ``DependencyInstallError``; returns the installer result on success."""
    from tools.lazy_deps import install_specs
    result = install_specs(specs, timeout=timeout, constraints=constraints, dry_run=dry_run)
    error = _classify(result)
    if error is not None:
        raise error
    return result


def check_candidate(candidate: PythonDeclaration, *, home: Path, project_root: Path) -> list[str]:
    """Prove core + every enabled plugin + *candidate* resolves, without installing. Returns the
    candidate's applicable specs (empty when it declares nothing for this platform)."""
    own = applicable_specs(candidate.specs)
    if not own or candidate.external:
        return []
    resolve(union_specs([candidate, *_peers(candidate, home)]), core_constraints(project_root), dry_run=True)
    return own


def _satisfied(spec: str) -> bool:
    """Installed and inside the specifier (extras are not checked; a missing extra surfaces at import)."""
    from importlib.metadata import PackageNotFoundError, version
    req = Requirement(spec)
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        return False
    return req.specifier.contains(installed, prereleases=True)


def _peers(decl: PythonDeclaration, home: Path) -> list[PythonDeclaration]:
    """Enabled plugins other than *decl* itself (matched by directory, so a re-enable is not its own peer)."""
    return [d for d in enabled_declarations(home) if d.plugin_dir.resolve() != decl.plugin_dir.resolve()]


def install_declaration(decl: PythonDeclaration, *, home: Path, project_root: Path,
                        timeout: int = 900) -> list[str]:
    """Install one plugin's applicable specs into the venv, resolved TOGETHER with every enabled
    peer's specs under core constraints, so installing A can never downgrade what enabled B needs
    (uv refuses instead). No-op when every spec is already satisfied. Returns what applies."""
    specs = applicable_specs(decl.specs)
    if specs and not decl.external and not all(_satisfied(s) for s in specs):
        resolve(union_specs([decl, *_peers(decl, home)]), core_constraints(project_root),
                dry_run=False, timeout=timeout)
    return specs


# ── install-time glue shared by CLI, dashboard and pack installs ──────────────

@dataclass(frozen=True)
class DepsOutcome:
    """What happened to a freshly installed plugin's Python dependencies. ``status`` is one of
    ``none`` (nothing declared / external runtime), ``installed``, ``failed`` (network, build, gate;
    the plugin stays installed and the loader warns at import), ``invalid`` (malformed declaration)."""
    status: str
    specs: tuple[str, ...] = ()
    detail: str = ""
    skipped: tuple[str, ...] = ()  # direct-URL requirements Hermes never installs

    @property
    def message(self) -> str:
        text = _OUTCOME_MESSAGES[self.status](self)
        if self.skipped:
            text += (f"\nNot installed (direct URL requirements are left to you): "
                     f"uv pip install {' '.join(repr(s) for s in self.skipped)}")
        return text.strip()


_OUTCOME_MESSAGES: dict[str, Callable[[DepsOutcome], str]] = {
    "none": lambda o: "",
    "installed": lambda o: f"Installed Python dependencies: {', '.join(o.specs)}",
    "failed": lambda o: (f"Python dependencies not installed ({o.detail}). Run manually: "
                         f"uv pip install {' '.join(o.specs)}"),
    "invalid": lambda o: f"Python dependency declaration ignored: {o.detail}",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def refuse_conflicting_candidate(plugin_dir: Path, *, home: Path) -> None:
    """Install-time gate, run BEFORE a plugin tree is moved into place: a candidate whose deps cannot
    resolve with core + the enabled plugins is refused; a malformed declaration is refused too. A
    network failure during the dry run is not a verdict and lets the install continue."""
    try:
        check_candidate(read_declaration(plugin_dir), home=home, project_root=project_root())
    except DependencyConflict as exc:
        raise DependencyConflict(
            f"its Python dependencies conflict with Hermes or an enabled plugin:\n{exc}") from exc
    except DependencyInstallError as exc:
        logger.warning("Dependency pre-check skipped (%s); the install will retry for real", exc)
    except (ValueError, tomllib.TOMLDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid Python dependency declaration: {exc}") from exc


def install_for_plugin_dir(plugin_dir: Path) -> DepsOutcome:
    """Install an installed plugin's declared deps into the venv. Never raises."""
    try:
        decl = read_declaration(plugin_dir)
    except (ValueError, tomllib.TOMLDecodeError, yaml.YAMLError) as exc:
        return DepsOutcome("invalid", detail=str(exc))
    skipped = () if decl.external else tuple(unsupported_specs(decl.specs))
    try:
        specs = install_declaration(decl, home=get_hermes_home(), project_root=project_root())
    except (DependencyConflict, DependencyInstallError) as exc:
        return DepsOutcome("failed", tuple(applicable_specs(decl.specs)), detail=_tail(str(exc), 300),
                           skipped=skipped)
    return DepsOutcome("installed" if specs else "none", tuple(specs), skipped=skipped)


# ── post-update re-apply ──────────────────────────────────────────────────────

@dataclass
class ReapplyReport:
    installed: list[str]
    dropped: list[tuple[str, str]]   # (plugin name, reason)
    failed: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed


def _drop_order(declarations: list[PythonDeclaration]) -> list[PythonDeclaration]:
    """Plugins in the order they may be sacrificed on conflict: non-memory first, memory last."""
    return sorted(declarations, key=lambda d: (d.memory_provider, d.name))


def _resolves(declarations: list[PythonDeclaration], constraints: list[str]) -> bool:
    try:
        resolve(union_specs(declarations), constraints, dry_run=True)
    except DependencyConflict:
        return False
    return True


def _first_resolving_subset(declarations: list[PythonDeclaration], constraints: list[str]
                            ) -> tuple[list[PythonDeclaration], list[PythonDeclaration]]:
    """Largest subset of *declarations* that resolves. Fast path: the whole union. Otherwise every
    plugin that cannot resolve on its own against core is a culprit and is dropped (never an innocent
    neighbour); what remains is peeled non-memory-first only for plugin-vs-plugin conflicts."""
    if _resolves(declarations, constraints):
        return list(declarations), []
    kept = [d for d in declarations if _resolves([d], constraints)]
    dropped = [d for d in declarations if d not in kept]
    order = _drop_order(kept)
    while kept and not _resolves(kept, constraints):
        victim = order.pop(0)
        kept.remove(victim)
        dropped.append(victim)
    return kept, dropped


def reapply_all(*, project_root: Path, disable: Callable[[Path, str], None]) -> ReapplyReport:
    """Re-install the union of every enabled plugin's deps after a venv rebuild. Conflicts drop
    non-memory plugins first and disable them through *disable(home, name)*; anything else (network,
    gate) is reported without changing plugin state."""
    per_home = [(home, enabled_declarations(home)) for home in dependency_homes()]
    everything = [d for _home, decls in per_home for d in decls]
    if not everything:
        return ReapplyReport(installed=[], dropped=[])
    constraints = core_constraints(project_root)
    try:
        kept, dropped = _first_resolving_subset(everything, constraints)
        specs = union_specs(kept)
        if specs:
            resolve(specs, constraints, dry_run=False)
    except DependencyInstallError as exc:
        return ReapplyReport(installed=[], dropped=[], failed=str(exc))
    report = ReapplyReport(installed=specs, dropped=[])
    for victim in dropped:
        home = next(h for h, decls in per_home if victim in decls)
        reason = "its Python dependencies no longer resolve against this Hermes"
        disable(home, victim.name)
        report.dropped.append((victim.name, reason))
    return report

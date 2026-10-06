"""Read plugin build declarations without importing CLI configuration or plugin code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    @property
    def external(self) -> bool:
        return str(self.manifest.get("python_runtime") or "").strip().lower() == "external"

    @property
    def install_requirements(self) -> tuple[str, ...]:
        return () if self.external else applicable_requirements(self.requirements)

    @property
    def is_member(self) -> bool:
        return not self.external and (self.pyproject is not None or bool(self.install_requirements))

    def python_error(self, version: str) -> str | None:
        """Why this member cannot join a workspace built for *version*; uv intersects every
        member's requires-python, so one excluding member fails the whole lock."""
        from packaging.specifiers import SpecifierSet

        if not self.is_member or not self.requires_python:
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


def locked_versions(lock: Path) -> dict[str, str]:
    """``{canonical name: version}`` for every package a ``uv.lock`` pins."""
    from packaging.utils import canonicalize_name

    return {canonicalize_name(package["name"]): package.get("version", "")
            for package in tomllib.loads(lock.read_text(encoding="utf-8-sig")).get("package", [])}


def quarantine_exemptions(pyproject: Path, core_locked: set[str]) -> tuple[list[str], list[str]]:
    """``(honoured, refused)`` names from a plugin's ``[tool.uv] exclude-newer-package``.

    Same rule core follows for itself: an exemption is honoured only as ``false`` on a
    direct dependency the plugin exact-pins and Hermes core's lock does not hold, so a
    plugin can let its own fresh release through without loosening anything else.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    document = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
    table = document.get("tool", {}).get("uv", {}).get("exclude-newer-package", {})
    exact = set()
    for spec in document.get("project", {}).get("dependencies", []):
        requirement = Requirement(spec)
        operators = [clause.operator for clause in requirement.specifier]
        if operators in (["=="], ["==="]) and "*" not in str(requirement.specifier):
            exact.add(canonicalize_name(requirement.name))
    honoured, refused = [], []
    for name, value in table.items() if isinstance(table, dict) else ():
        key = canonicalize_name(name)
        ok = value is False and key in exact and key not in core_locked
        (honoured if ok else refused).append(key)
    return honoured, refused


def read_python_declaration(plugin_dir: Path) -> PythonDeclaration:
    """One effective surface for consent, membership, builds and currency.

    A real pyproject owns packaging. Legacy manifests bridge plugins without
    one, including old PM-generated pyprojects. Both YAML suffixes and both
    dependency aliases have identical semantics. Reading never initializes a
    home, discovers code, or depends on the caller's CLI import state.
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
        document = tomllib.loads(text) if "GENERATED by pm" not in text else {}
        # A pyproject with no [project] table only configures tools (ruff, pytest);
        # the manifest still declares the dependencies.
        if "project" in document:
            specs = document["project"].get("dependencies", [])
            if not isinstance(specs, list) or any(not isinstance(v, str) for v in specs):
                raise ValueError(f"invalid project.dependencies: {project}")
            requires_python = document["project"].get("requires-python")
            if requires_python is not None and not isinstance(requires_python, str):
                raise ValueError(f"invalid project.requires-python: {project}")
            return PythonDeclaration(tuple(files), project, tuple(specs), manifest, requires_python)
    specs = []
    for key in ("pip_dependencies", "python_dependencies"):
        values = manifest.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"invalid {key}: {native}")
        specs.extend(values)
    return PythonDeclaration(tuple(files), None, tuple(dict.fromkeys(specs)), manifest)
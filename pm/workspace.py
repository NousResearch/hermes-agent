"""Resolve core plus plugin requirements in a writable build snapshot.

Shipped source and locks are inputs, never mutation targets. Candidate
failure propagates without changing plugin configuration or the live venv.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pm.environment import PythonEnvironment

from pm import paths
from pm.package import InstallError

WORKSPACE_DIRNAME = ".pm-workspace"
_MEMBER_EXCLUDE = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})


def _member_ignored(directory, names):
    return [name for name in names if name in _MEMBER_EXCLUDE or name.endswith(".egg-info")]


class ResolutionConflict(InstallError):
    """uv's resolver proved the union has no valid solution."""


# Markers uv prints ONLY when the resolver itself proves no solution
# exists (its conflict report: "Because ...", "no solution found").
# Deliberately narrow: a fetch timeout or index outage must not be
# misread as a conflict — and regardless of classification, nothing
# here ever disables a plugin; the caller decides.
_RESOLVER_MARKERS = (
    "no solution found",
    "conflicting requirements",
    "conflicting urls",
    "because only the following versions",
    "and your pyproject depends on",
)


def classify_uv_failure(stage: str, returncode: int, output: str) -> InstallError:
    """Turn a failed `uv <stage>` into the right classified error.

    Resolver-conflict output → ResolutionConflict; anything else (fetch,
    build, tooling) → plain InstallError with the tail of the output.
    """
    cause = f"uv {stage} exited {returncode}: {output.strip()[-600:]}"
    lowered = output.lower()
    if any(marker in lowered for marker in _RESOLVER_MARKERS):
        return ResolutionConflict("venv", cause)
    return InstallError("venv", cause)


def workspace_root() -> Path:
    """Default preparation root; callers can supply a fresh transaction root."""
    from hermes_cli.runtime_paths import install_state_dir
    return install_state_dir(paths.repo_root()) / WORKSPACE_DIRNAME


def _member_rel(root: Path, plugin_dir: Path) -> str:
    """Use portable separators for a member inside the generated workspace."""
    return os.path.relpath(plugin_dir.resolve(), root.resolve()).replace("\\", "/")


def member_sources(plugin_dirs) -> dict[Path, Path]:
    """Map installed identities to build inputs, including staged plugin updates."""
    rows = plugin_dirs.items() if isinstance(plugin_dirs, Mapping) else ((path, path) for path in plugin_dirs)
    return {Path(identity).resolve(): Path(source).resolve() for identity, source in rows}


def members_stamp(plugin_dirs) -> str:
    """Hash the member inputs copied into a generation, independent of staging paths."""
    h = hashlib.sha256()
    for identity, entry in sorted(member_sources(plugin_dirs).items()):
        h.update(str(identity).encode("utf-8"))
        h.update(b"\0")
        for name in ("pyproject.toml", "plugin.yaml"):
            source = entry / name
            if source.is_file():
                h.update(source.read_bytes())
            h.update(b"\0")
        if (entry / "pyproject.toml").is_file():
            for directory, dirs, files in os.walk(entry):
                dirs[:] = sorted(set(dirs) - set(_member_ignored(directory, dirs)))
                for name in sorted(set(files) - set(_member_ignored(directory, files))):
                    path = Path(directory) / name
                    h.update(path.relative_to(entry).as_posix().encode())
                    h.update(b"\0")
                    h.update(os.readlink(path).encode() if path.is_symlink() else path.read_bytes())
                    h.update(b"\0")
    return h.hexdigest()


def _copy_core_inputs(source: Path, destination: Path) -> None:
    """Build from a writable snapshot, never from signed/read-only source."""
    import shutil

    import fnmatch
    import tomllib

    metadata = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8-sig"))
    project = metadata.get("project", {})
    setuptools = metadata.get("tool", {}).get("setuptools", {})
    patterns = setuptools.get("packages", {}).get("find", {}).get("include", ["*"])
    package_roots = {pattern.split(".", 1)[0] for pattern in patterns}
    files = {"pyproject.toml", "setup.py", "setup.cfg"}
    readme = project.get("readme")
    if isinstance(readme, str):
        files.add(readme)
    elif isinstance(readme, dict) and "file" in readme:
        files.add(readme["file"])
    for pattern in project.get("license-files", []):
        files.update(str(p.relative_to(source)) for p in source.glob(pattern))
    files.update(p.name for p in source.glob("*.py"))

    excluded = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", "release", "uv.lock"}
    def ignore(directory, names):
        return [name for name in names if name in excluded or name.startswith(".")
                or name.endswith(".egg-info") or (Path(directory) / name).is_symlink()]

    for entry in source.iterdir():
        if (entry.is_dir() and not entry.is_symlink() and entry.name not in excluded
                and not entry.name.startswith(".") and entry.resolve() != destination.resolve()
                and any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in package_roots)):
            target = destination / entry.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(entry, target, ignore=ignore)
    for name in files:
        entry = source / name
        if not entry.is_file() or entry.is_symlink():
            continue
        if not entry.resolve().is_relative_to(source.resolve()):
            raise InstallError("venv", f"build input escapes the core project: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, target)


def _generate_pyproject(plugin_dirs: list[Path], root: Optional[Path] = None, *,
                        source: Optional[Path] = None) -> tuple[Path, bool]:
    """(Re)generate the workspace root's pyproject.toml from core's
    pyproject + the enabled plugin members. Idempotent — same inputs,
    same bytes. Returns (root, changed): changed is True when the member
    surface moved (member set or a member's pyproject content), which is
    the signal to re-seed the resolution from the committed lock."""
    if root is None:
        root = workspace_root()
    source = (paths.repo_root() if source is None else source).resolve()
    if root.resolve() == source or source.is_relative_to(root.resolve()):
        raise InstallError("venv", "workspace must not replace the core source")
    root.mkdir(parents=True, exist_ok=True)

    core_pyproject = source / "pyproject.toml"
    core_text = core_pyproject.read_text(encoding="utf-8-sig")

    members = [_member_rel(root, _workspace_member(source, root, identity=identity))
               for identity, source in member_sources(plugin_dirs).items()]

    lines = [core_text.rstrip("\n")]
    if members:
        lines.append("")
        lines.append("[tool.uv.workspace]")
        lines.append("members = [" + ", ".join(f'"{m}"' for m in sorted(members)) + "]")

    text = "\n".join(lines) + "\n"
    target = root / "pyproject.toml"
    try:
        changed = target.read_text(encoding="utf-8") != text
    except OSError:
        changed = True
    _copy_core_inputs(source, root)
    target.write_text(text, encoding="utf-8")
    return root, changed


def _seed_lock(root: Path, seed_lock: Optional[Path] = None, *, source: Optional[Path] = None) -> None:
    """Seed the generated root's uv.lock with the CURRENT resolution.

    Seed precedence: the parent-supplied ``seed_lock`` path first, then
    the root's own existing uv.lock (the current EXTENDED resolution from
    the previous sync), then the committed core lock — so a plugin-driven
    extension keeps every compatible selection it already made, and a
    fresh root extends the committed resolution. uv preserves compatible
    selections from the seed (a plugin's range spec does not move core
    pins); explicit exact requirements stay binding as declared
    constraints. The lock is COPIED — shipped/extended source bytes are
    never rewritten; only the staging root receives the copy.

    Called only when the member surface changed; an unchanged root keeps
    its lock untouched, so repeated syncs are stable. Seed failures
    SURFACE (they would silently degrade the resolution otherwise)."""
    if seed_lock is None:
        existing = root / "uv.lock"
        if existing.is_file():
            seed_lock = existing
        else:
            seed_lock = (paths.repo_root() if source is None else source) / "uv.lock"
    if not seed_lock.is_file():
        return  # nothing committed to seed from; uv resolves from scratch
    (root / "uv.lock").write_bytes(seed_lock.read_bytes())


def _is_member_candidate(plugin_dir: Path) -> bool:
    """A plugin dir is a workspace-member candidate when it declares python
    deps: a pyproject.toml (modern), or legacy pip_dependencies/
    python_dependencies in plugin.yaml (the bridge materializes those)."""
    import stat

    for name in ("pyproject.toml", "plugin.yaml"):
        path = plugin_dir / name
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"could not inspect plugin metadata: {path}") from exc
        if not stat.S_ISREG(mode):
            continue
        if name == "pyproject.toml":
            return True
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"could not read plugin metadata: {path}") from exc
        return "pip_dependencies" in text or "python_dependencies" in text
    return False


def enabled_plugin_dirs(*, proposed_home=None, enabled=None, disabled=None) -> list[Path]:
    """Resolve the effective plugin selection without filtering dependency declarations."""
    from pm.plugins_state import _is_directory, enabled_plugins_ordered

    selection = enabled_plugins_ordered() if proposed_home is None else enabled_plugins_ordered(
        proposed_home=proposed_home, enabled=enabled, disabled=disabled,
    )
    members = []
    for plugins_dir, names in selection.items():
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise InstallError("venv", f"invalid plugin key: {name}")
            plugin_dir = plugins_dir / relative
            if not _is_directory(plugin_dir):
                plugin_dir = paths.repo_root() / "plugins" / relative
            if _is_directory(plugin_dir):
                members.append(plugin_dir)
    return list(dict.fromkeys(members))


def enabled_member_dirs(*, proposed_home=None, enabled=None, disabled=None) -> list[Path]:
    """Keep every selected member or refuse an incompatible selection."""
    from hermes_cli.plugins_cmd import _check_manifest_version, _read_manifest_for_install

    selected = enabled_plugin_dirs(proposed_home=proposed_home, enabled=enabled, disabled=disabled)
    for path in selected:
        _check_manifest_version(_read_manifest_for_install(path), path.name)
    return [path for path in selected if _is_member_candidate(path)]


def _legacy_requirements(plugin_dir: Path) -> list[str]:
    from utils import fast_safe_load

    manifest = plugin_dir / "plugin.yaml"
    if not manifest.is_file():
        return []
    data = fast_safe_load(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise InstallError("venv", f"invalid plugin manifest: {manifest}")
    specs = []
    for key in ("pip_dependencies", "python_dependencies"):
        values = data.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise InstallError("venv", f"invalid {key}: {manifest}")
        specs.extend(values)
    return list(dict.fromkeys(specs))


def _workspace_member(plugin_dir: Path, root: Path, *, identity: Path | None = None) -> Path:
    """Keep workspace members with their generation, not a temporary install clone."""
    import json
    import shutil
    import tomllib

    key = hashlib.sha256(str((identity or plugin_dir).resolve()).encode()).hexdigest()[:16]
    pyproject = plugin_dir / "pyproject.toml"
    if pyproject.is_file() and "GENERATED by pm" not in pyproject.read_text(encoding="utf-8-sig"):
        member = root / "plugin-sources" / key
        if member.exists():
            shutil.rmtree(member)
        shutil.copytree(plugin_dir, member, symlinks=True,
                        ignore=_member_ignored)
        document = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
        changed = False
        for sources in document.get("tool", {}).get("uv", {}).get("sources", {}).values():
            for spec in sources if isinstance(sources, list) else [sources]:
                if not isinstance(spec, dict) or "path" not in spec:
                    continue
                relative = Path(spec["path"])
                if relative.is_absolute():
                    continue
                resolved = (plugin_dir / relative).resolve()
                if resolved.is_relative_to(plugin_dir.resolve()):
                    continue  # The referenced tree was copied with this member.
                spec["path"] = ((identity or plugin_dir) / relative).resolve().as_posix()
                changed = True
        if changed:
            import tomli_w

            (member / "pyproject.toml").write_text(tomli_w.dumps(document), encoding="utf-8")
        return member
    specs = _legacy_requirements(plugin_dir)
    member = root / "plugin-deps" / key
    member.mkdir(parents=True, exist_ok=True)
    (member / "pyproject.toml").write_text(
        f'[project]\nname = "hermes-plugin-{key}"\nversion = "0.0.0"\n'
        'requires-python = ">=3.11"\n'
        f'dependencies = {json.dumps(specs)}\n[tool.uv]\npackage = false\n',
        encoding="utf-8",
    )
    return member


def install_node_sidecar(
    plugin_dir: Path,
    *,
    explicit: bool = False,
) -> Optional[str]:
    """Install plugin-local dependencies using PM's paired npm/Node context.

    Explicit user consent permits acquisition even when on-demand installs
    are disabled. Returns None on success, otherwise a diagnostic.
    """
    package_json = plugin_dir / "package.json"
    if not package_json.is_file():
        return None  # nothing to install

    import pm
    from pm.ensure import lazy_installs_allowed

    # Tool availability does not authorize mutation of the sidecar itself.
    if not explicit and not lazy_installs_allowed():
        return "lazy installs are disabled — run `hermes plugins install` and approve Node dependencies"

    # a lockfile means reproducible `npm ci`; plain `npm install` otherwise
    install_cmd = ["ci"] if (plugin_dir / "package-lock.json").is_file() else ["install"]
    try:
        runner = pm.ensure("npm", explicit=explicit)
        # Resolve inside the composed context, including npm.cmd on Windows;
        # CreateProcess does not search a child's replacement PATH itself.
        npm = shutil.which("npm", path=runner.env.get("PATH", ""))
        if npm is None:
            return "npm is missing from the prepared PM environment"
        proc = runner.run(
            [npm, *install_cmd, "--no-audit", "--no-fund"],
            cwd=str(plugin_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
    except Exception as exc:
        return f"npm {install_cmd[0]} failed to run: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        return f"npm {install_cmd[0]} exited {proc.returncode}: {tail}"
    return None


def lock_and_sync(
    plugin_dirs: list[Path],
    extras: Optional[list[str]] = None,
    *,
    venv_dir: Path,
    root: Optional[Path] = None,
    env: Optional[dict] = None,
    seed_lock: Optional[Path] = None,
    frozen: bool = False,
    replay: Optional[Path] = None,
    source: Optional[Path] = None,
    environment: PythonEnvironment | None = None,
) -> None:
    """Build the root, then `uv lock` + `uv sync --frozen --extra ...`.

    Everything resolves into a parent-supplied STAGING surface: ``root``
    pins the generated workspace dir, ``venv_dir`` pins
    UV_PROJECT_ENVIRONMENT and ``seed_lock`` (optional) pins which
    existing lock seeds the extension (default: the root's current
    extended lock, else the committed core lock). ``env`` replaces the
    ambient base environment when supplied; either way the subprocess
    gets a COPY — the live process environment is never mutated. ``replay``
    copies a recorded sibling workspace and uses its lock without resolution
    or plugin discovery; it is reserved for restoring an existing selection.
    ``source`` and ``environment`` bypass live source/tool discovery when supplied;
    the environment owns the child process policy, cache and interpreter.

    Raises a CLASSIFIED InstallError on failure: ResolutionConflict only
    for a confirmed resolver conflict; network, build and tool failures
    stay generic InstallError — they are not evidence of a dependency
    conflict and must not disable plugins.
    """
    if environment is not None and environment.destination != venv_dir:
        raise ValueError("workspace and environment destinations differ")

    if replay is None:
        generated, changed = _generate_pyproject(plugin_dirs, root, source=source)
        if changed:
            _seed_lock(generated, seed_lock, source=source)
    else:
        import shutil

        if root is None or not (replay / "pyproject.toml").is_file() or not (replay / "uv.lock").is_file():
            raise InstallError("venv", f"recorded workspace is missing: {replay}")
        # Generation workspaces are siblings at the same depth. External
        # member paths still resolve. Generated members move with the copy.
        shutil.copytree(replay, root, ignore=shutil.ignore_patterns("__pycache__", ".venv", "build", "*.egg-info"))
        generated = root
        frozen = True

    if environment is None:
        from pm.environment import managed_environment

        environment = managed_environment(venv_dir, env=env)
    environment.sync(generated, extras=extras or (), frozen=frozen)

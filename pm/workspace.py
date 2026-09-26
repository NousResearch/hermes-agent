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
from pm.plugin_declarations import read_python_declaration, manifest_version_error

import functools
import re

_MEMBER_EXCLUDE = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage", ".tox", ".nox",
})
_DEFAULT_FILE_EXCLUDES = frozenset({".DS_Store", "Thumbs.db"})
_DEFAULT_SUFFIX_EXCLUDES = (
    ".egg-info", ".pyc", ".pyo", ".pyd",
    ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".db-journal",
    ".log",
)


def _gitignore_pattern_to_regex(pattern: str, anchored: bool) -> re.Pattern:
    i, n = 0, len(pattern)
    res = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
                    res.append("(?:.*/)?")
                else:
                    res.append(".*")
            else:
                res.append("[^/]*")
                i += 1
        elif c == "?":
            res.append("[^/]")
            i += 1
        elif c == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j < n:
                res.append(pattern[i : j + 1])
                i = j + 1
            else:
                res.append(r"\[")
                i += 1
        else:
            res.append(re.escape(c))
            i += 1
    body = "".join(res)
    if not anchored:
        pattern_str = f"(?:^|.*/){body}(?:/.*)?$"
    else:
        pattern_str = f"^{body}(?:/.*)?$"
    return re.compile(pattern_str)


def _parse_ignore_lines(lines: list[str]) -> list[tuple[re.Pattern, bool, bool]]:
    rules = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
            if not line or line.startswith("#"):
                continue
        dir_only = line.endswith("/")
        if dir_only:
            line = line.rstrip("/")
        if not line:
            continue
        anchored = line.startswith("/") or ("/" in line)
        line = line.lstrip("/")
        regex = _gitignore_pattern_to_regex(line, anchored)
        rules.append((regex, negated, dir_only))
    return rules


def _find_member_root(start_dir: Path) -> Path:
    current = start_dir
    while True:
        if (
            (current / "pyproject.toml").is_file()
            or (current / "plugin.yaml").is_file()
            or (current / "plugin.yml").is_file()
            or (current / "plugin.json").is_file()
            or (current / ".git").exists()
        ):
            return current
        if current.parent == current:
            return start_dir
        current = current.parent


@functools.lru_cache(maxsize=1024)
def _member_ignore_rules(dir_str: str) -> tuple[tuple[re.Pattern, bool, bool, Path], ...]:
    dir_path = Path(dir_str)
    root = _find_member_root(dir_path)
    chain = []
    curr = dir_path
    while True:
        chain.append(curr)
        if curr == root or curr.parent == curr:
            break
        curr = curr.parent
    chain.reverse()

    all_rules: list[tuple[re.Pattern, bool, bool, Path]] = []
    for folder in chain:
        for fname in (".gitignore", ".hermesignore"):
            ignore_file = folder / fname
            if ignore_file.is_file():
                try:
                    lines = ignore_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                except OSError:
                    continue
                for regex, negated, dir_only in _parse_ignore_lines(lines):
                    all_rules.append((regex, negated, dir_only, folder))
        if folder == root:
            manifest_in = folder / "MANIFEST.in"
            if manifest_in.is_file():
                try:
                    lines = manifest_in.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                    m_lines = []
                    for raw in lines:
                        raw = raw.strip()
                        if raw.startswith("exclude "):
                            m_lines.append(raw.split(None, 1)[1].strip())
                        elif raw.startswith("prune "):
                            m_lines.append(raw.split(None, 1)[1].strip() + "/")
                        elif raw.startswith("global-exclude "):
                            m_lines.append(raw.split(None, 1)[1].strip())
                    for regex, negated, dir_only in _parse_ignore_lines(m_lines):
                        all_rules.append((regex, negated, dir_only, folder))
                except OSError:
                    pass
            pyproject = folder / "pyproject.toml"
            if pyproject.is_file():
                try:
                    import tomllib

                    doc = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
                    excludes = doc.get("tool", {}).get("hermes", {}).get("build", {}).get("exclude", [])
                    if isinstance(excludes, list):
                        for regex, negated, dir_only in _parse_ignore_lines([str(x) for x in excludes]):
                            all_rules.append((regex, negated, dir_only, folder))
                except Exception:
                    pass

    return tuple(all_rules)


def _member_ignored(directory, names):
    dir_path = Path(directory)
    dir_str = str(dir_path.resolve())
    rules = _member_ignore_rules(dir_str)

    ignored = set()
    for name in names:
        if name in _MEMBER_EXCLUDE or name in _DEFAULT_FILE_EXCLUDES or name.endswith(_DEFAULT_SUFFIX_EXCLUDES):
            is_ignored = True
        else:
            is_ignored = False

        if rules:
            item_path = dir_path / name
            is_dir = None
            for regex, negated, dir_only, base_dir in rules:
                if dir_only:
                    if is_dir is None:
                        is_dir = item_path.is_dir()
                    if not is_dir:
                        continue
                try:
                    rel = item_path.relative_to(base_dir).as_posix()
                except ValueError:
                    rel = name
                if regex.search(rel):
                    is_ignored = not negated

        if is_ignored:
            ignored.add(name)

    return sorted(ignored)


# The uv failure classifier lives beside the uv runner (stdlib-only imports): the bootstrap
# runner streams uv output from a pre-3.11 system python where this module's tomllib import
# cannot load. Workspace callers keep reaching it from here.
from pm.environment import ResolutionConflict, classify_uv_failure  # noqa: E402,F401


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
        declaration = read_python_declaration(entry)
        for source in declaration.files:
            h.update(source.name.encode("utf-8"))
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


def _generate_pyproject(plugin_dirs: list[Path] | Mapping[Path, Path], root: Path, *, source: Path) -> None:
    """Snapshot core and plugin build inputs into a fresh generation."""
    source = source.resolve()
    if root.resolve() == source or source.is_relative_to(root.resolve()):
        raise InstallError("venv", "workspace must not replace the core source")
    root.mkdir(parents=True)

    core_pyproject = source / "pyproject.toml"
    core_text = core_pyproject.read_text(encoding="utf-8-sig")

    members = [_workspace_member(source, root, identity=identity).relative_to(root).as_posix()
               for identity, source in member_sources(plugin_dirs).items()
               if _is_member_candidate(source)]

    if members:
        import tomllib

        import tomli_w

        document = tomllib.loads(core_text)
        _core_release_quarantine(document, source / "uv.lock")
        document.setdefault("tool", {}).setdefault("uv", {})["workspace"] = {"members": sorted(members)}
        text = tomli_w.dumps(document)
    else:
        # Byte-identical to core: a member-less generation syncs frozen against core's own lock.
        text = core_text.rstrip("\n") + "\n"
    target = root / "pyproject.toml"
    _copy_core_inputs(source, root)
    target.write_text(text, encoding="utf-8")


def _core_release_quarantine(document: dict, core_lock: Path) -> None:
    """Scope core's ``exclude-newer`` to the packages in core's own lock.

    The quarantine covers Hermes's own dependencies only; a plugin's dependencies
    follow the plugin's own policy, so a catalog pin floored on a fresh release still
    installs. A global cutoff would filter plugin-only packages too, so it moves onto
    every registry package core locks. A plugin still cannot drag one of those past
    the window, and core's own ``= false`` exemptions stay as written.
    """
    import re
    import tomllib

    settings = document.get("tool", {}).get("uv", {})
    cutoff = settings.pop("exclude-newer", None)
    if cutoff is None:
        return

    def normalized(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    per_package = {normalized(name): value
                   for name, value in settings.get("exclude-newer-package", {}).items()}
    for package in tomllib.loads(core_lock.read_text(encoding="utf-8-sig")).get("package", []):
        if "registry" in package.get("source", {}):
            per_package.setdefault(normalized(package["name"]), cutoff)
    settings["exclude-newer-package"] = per_package


def _is_member_candidate(plugin_dir: Path) -> bool:
    return read_python_declaration(plugin_dir).is_member


def enabled_plugin_entries(*, proposed_home=None, enabled=None, disabled=None,
                           installing: Path | None = None,
                           skip_invalid_secondary: bool = False) -> list[tuple[Path, str, Path]]:
    """``(home plugins dir, selection key, plugin dir)`` for every selected plugin, in config order.

    The key is what the home's config names, so a caller can edit that home's selection.
    """
    from pm.plugins_state import _is_directory, enabled_plugins_ordered

    selection = enabled_plugins_ordered(
        proposed_home=proposed_home, enabled=enabled, disabled=disabled, installing=installing,
        skip_invalid_secondary=skip_invalid_secondary,
    )
    entries = []
    for plugins_dir, names in selection.items():
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise InstallError("venv", f"invalid plugin key: {name}")
            plugin_dir = plugins_dir / relative
            proposed = installing is not None and plugin_dir.resolve() == installing.resolve()
            if not proposed and not _is_directory(plugin_dir):
                plugin_dir = paths.repo_root() / "plugins" / relative
            if proposed or _is_directory(plugin_dir):
                entries.append((plugins_dir, name, plugin_dir))
    return entries


def enabled_plugin_dirs(*, proposed_home=None, enabled=None, disabled=None,
                        installing: Path | None = None, skip_invalid_secondary: bool = False) -> list[Path]:
    """Resolve the effective plugin selection without filtering dependency declarations."""
    entries = enabled_plugin_entries(proposed_home=proposed_home, enabled=enabled, disabled=disabled,
                                     installing=installing, skip_invalid_secondary=skip_invalid_secondary)
    return list(dict.fromkeys(plugin_dir for _plugins_dir, _name, plugin_dir in entries))


def enabled_member_dirs(*, proposed_home=None, enabled=None, disabled=None) -> list[Path]:
    """Keep every selected member or refuse an incompatible selection.

    A member whose requires_hermes rejects the running version sits out instead: the
    verdict is only as good as our version identity (an untagged source checkout reads
    as an older release), the loader skips that plugin anyway, and the member rejoins
    as soon as the verdict flips. Enabling one is still refused at admission.
    """
    selected = enabled_plugin_dirs(proposed_home=proposed_home, enabled=enabled, disabled=disabled,
                                   skip_invalid_secondary=proposed_home is None)
    members = []
    for path in selected:
        # Per plugin: with none selected, PM must not import the application's manifest module.
        from hermes_cli.plugins_manifest import requires_hermes_error

        declaration = read_python_declaration(path)
        if requires_hermes_error(declaration.manifest):
            continue
        reason = manifest_version_error(declaration.manifest, path.name)
        if reason:
            raise InstallError("venv", reason)
        if declaration.is_member:
            members.append(path)
    return members


def _member_key(identity: Path) -> str:
    """``<plugin dir name>-<sha256(path)[:16]>``: the hash keeps two same-named plugins from
    different homes apart; the name is what a user sees in uv's conflict text
    (``hermes-plugin-<key> depends on …``) — a bare hash told them nothing to disable."""
    import re

    digest = hashlib.sha256(str(identity.resolve()).encode()).hexdigest()[:16]
    name = re.sub(r"[^a-z0-9._-]+", "-", identity.name.lower()).strip("-.") or "plugin"
    return f"{name}-{digest}"


def _workspace_member(plugin_dir: Path, root: Path, *, identity: Path) -> Path:
    """Keep workspace members with their generation, not a temporary install clone."""
    import json
    import tomllib

    key = _member_key(identity)
    declaration = read_python_declaration(plugin_dir)
    pyproject = declaration.pyproject
    if pyproject is not None:
        member = root / "plugin-sources" / key
        shutil.copytree(plugin_dir, member, symlinks=True,
                        ignore=_member_ignored)
        document = tomllib.loads(pyproject.read_text(encoding="utf-8-sig"))
        # uv identifies a workspace member by [project].name, so the same virtual
        # plugin enabled in two profiles would declare one name twice and fail
        # `uv lock`. A member with no build backend is metadata-only: it can carry
        # the unique key in its name, as manifest-only members already do. A
        # buildable member keeps its declared name — uv verifies it against the
        # package metadata its backend produces.
        # tool.uv.package = true opts into uv package mode: real build metadata, so buildable.
        virtual = "build-system" not in document and document.get("tool", {}).get("uv", {}).get("package") is not True
        changed = declaration.install_requirements != declaration.requirements
        if changed:
            document["project"]["dependencies"] = list(declaration.install_requirements)
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
                spec["path"] = (identity / relative).resolve().as_posix()
                changed = True
        if virtual:
            document.setdefault("project", {})["name"] = f"hermes-plugin-{key}"
        if virtual or changed:
            import tomli_w

            (member / "pyproject.toml").write_text(tomli_w.dumps(document), encoding="utf-8")
        return member
    specs = declaration.install_requirements
    member = root / "plugin-deps" / key
    member.mkdir(parents=True)
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
    from pm.install import lazy_installs_allowed

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
    plugin_dirs: list[Path] | Mapping[Path, Path],
    extras: list[str],
    *,
    root: Path,
    source: Path,
    seed_lock: Path | None,
    environment: PythonEnvironment,
    frozen: bool = False,
    replay: Path | None = None,
) -> None:
    """Prepare a fresh generation using explicit inputs and a prepared engine.

    The caller selects the seed; uv retains its compatible versions. Repair
    copies the recorded build inputs and never reads current manifests.
    Resolver conflicts remain distinct from download/build failures.
    """
    if root.exists() or root.is_symlink():
        raise InstallError("venv", f"workspace must be fresh: {root}")
    if replay is None:
        _generate_pyproject(plugin_dirs, root, source=source)
        if seed_lock is not None:
            (root / "uv.lock").write_bytes(seed_lock.read_bytes())
    else:
        if not (replay / "pyproject.toml").is_file() or not (replay / "uv.lock").is_file():
            raise InstallError("venv", f"recorded workspace is missing: {replay}")
        # Sibling generations keep external relative paths at the same depth;
        # snapshotted members and their exact lock travel with the workspace.
        # Use the snapshot's exclusions: build/ may hold an in-tree backend.
        shutil.copytree(replay, root, symlinks=True, ignore=_member_ignored)
        frozen = True

    environment.sync(root, extras=extras, frozen=frozen)

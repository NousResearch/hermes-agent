"""Resolve core plus plugin requirements in a writable build snapshot.

Shipped source and locks are inputs, never mutation targets. Candidate
failure propagates without changing plugin configuration or the live venv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pm.environment import PythonEnvironment  # noqa: TID251

from pm import paths
from pm.package import InstallError
from pm.plugin_declarations import read_python_declaration, manifest_version_error

_MEMBER_EXCLUDE = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})


def _member_ignored(directory, names):
    return [name for name in names if name in _MEMBER_EXCLUDE or name.endswith(".egg-info")]


# The uv failure classifier lives beside the uv runner (stdlib-only imports): the bootstrap
# runner streams uv output from a pre-3.11 system python where this module's tomllib import
# cannot load. Workspace callers keep reaching it from here.
from pm.environment import ResolutionConflict, classify_uv_failure  # noqa: TID251


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


def _published_store_entries(directory: Path) -> frozenset[str]:
    """Names of the PM tool-store entries published in *directory*, if it is a store root.

    The container image keeps its store in the ``tools`` package directory, so the snapshot would
    otherwise copy every pinned interpreter, browser and media binary (about 1.5 GB) into each
    dependency generation. Nothing builds from them: the build runs the store's own tools in
    place. A directory counts as a store only when its ``facts.json`` records published entries,
    and only those entries (plus hash-keyed fetch caches) are left out.
    """
    try:
        facts = json.loads((directory / "facts.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return frozenset()
    packages = facts.get("packages") if isinstance(facts, dict) else None
    if not isinstance(packages, dict):
        return frozenset()
    names = {fact["entry"] for fact in packages.values()
             if isinstance(fact, dict) and isinstance(fact.get("entry"), str)}
    names = {name for name in names if name and Path(name).name == name}
    names.update(child.name for child in directory.glob("fetch-*") if child.is_dir())
    return frozenset(name for name in names if (directory / name).is_dir())


def _copy_core_inputs(source: Path, destination: Path) -> None:
    """Build from a writable snapshot, never from signed/read-only source."""
    import fnmatch
    import tomllib

    metadata = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8-sig"))
    project = metadata.get("project", {})
    setuptools = metadata.get("tool", {}).get("setuptools", {})
    find = setuptools.get("packages", {}).get("find", {})
    patterns = find.get("include", ["*"])
    package_roots = {pattern.split(".", 1)[0] for pattern in patterns}
    # A ``where`` root (``src``) holds packages the include patterns name below it, not by its own name.
    where_roots = {Path(where).parts[0] for where in find.get("where", ["."]) if Path(where).parts}
    files = {"pyproject.toml", "setup.py", "setup.cfg"}
    readme = project.get("readme")
    if isinstance(readme, str):
        files.add(readme)
    elif isinstance(readme, dict) and "file" in readme:
        files.add(readme["file"])
    for pattern in project.get("license-files", []):
        files.update(str(p.relative_to(source)) for p in source.glob(pattern))
    files.update(p.name for p in source.glob("*.py"))

    # uv.lock is not excluded: the root lock is never copied (only ``files`` are; lock_and_sync
    # seeds or resolves it), and pm/uv.lock is the PM runtime's input (pm/runtime.py::_inputs).
    excluded = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", "release"}
    # A root dist/ is build output, but below a package root it is shipped: the managed
    # environment runs from this snapshot and serves bundled plugins' dashboard/dist/.
    nested_excluded = excluded - {"dist"}
    def ignore(directory, names):
        published = _published_store_entries(Path(directory))
        return [name for name in names if name in nested_excluded or name.startswith(".")
                or name in published or name.endswith(".egg-info") or (Path(directory) / name).is_symlink()]

    for entry in source.iterdir():
        if (entry.is_dir() and not entry.is_symlink() and entry.name not in excluded
                and not entry.name.startswith(".") and entry.resolve() != destination.resolve()
                and (entry.name in where_roots
                     or any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in package_roots))):
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


def _generate_pyproject(plugin_dirs: list[Path] | Mapping[Path, Path], root: Path, *, source: Path,
                        python: Path, extras: Iterable[str] = ()) -> None:
    """Snapshot core and plugin build inputs into a fresh generation for the *python* that installs it."""
    source = source.resolve()
    if root.resolve() == source or source.is_relative_to(root.resolve()):
        raise InstallError("venv", "workspace must not replace the core source")
    root.mkdir(parents=True)

    core_pyproject = source / "pyproject.toml"
    core_text = core_pyproject.read_text(encoding="utf-8-sig")

    candidates = {identity: source for identity, source in member_sources(plugin_dirs).items()
                  if _is_member_candidate(source)}
    target = _target_environment(python) if candidates else {}
    members = [_workspace_member(source, root, identity=identity, extras=extras, target=target)
               .relative_to(root).as_posix() for identity, source in candidates.items()]

    if members:
        import tomllib

        import tomli_w

        document = tomllib.loads(core_text)
        uv_policy = document.setdefault("tool", {}).setdefault("uv", {})
        # Release age is checked when Hermes locks; users trust that lock (#135589).
        uv_policy.pop("exclude-newer", None)
        uv_policy.pop("exclude-newer-package", None)
        uv_policy["workspace"] = {"members": sorted(members)}
        # A member requirement naming core (a selected extra's ``hermes-agent[x] @ git+…``)
        # names a workspace member, which uv refuses without a source; root sources apply
        # to every member, renamed virtual ones included.
        uv_policy.setdefault("sources", {})[document["project"]["name"]] = {"workspace": True}
        _solve_only_for(uv_policy, target)
        text = tomli_w.dumps(document)
    else:
        # A member-less generation syncs frozen against core's own lock.
        text = core_text.rstrip("\n") + "\n"
    target = root / "pyproject.toml"
    text = _without_release_age(text)
    _copy_core_inputs(source, root)
    target.write_text(text, encoding="utf-8")


def _without_release_age(text: str) -> str:
    """Drop ``[tool.uv] exclude-newer`` from a generated root's pyproject text.

    The lock already holds checked versions; on the user's machine the cutoff only gates the
    build tools uv fetches for source builds (setuptools for Hermes itself), and a mirror that
    serves no upload times rejects all of them (#135589). Text-level so the pre-3.11 bootstrap
    python, which has no tomllib, can run it.
    """
    import re

    return re.sub(r"(?m)^exclude-newer\s*=.*\n", "", text)


# packaging.markers.default_environment(), computed by the target interpreter itself; stdlib
# only, because that interpreter need not have packaging installed.
_MARKER_ENVIRONMENT_PROBE = """
import json, os, platform, sys
v = sys.implementation.version
level = "" if v.releaselevel == "final" else f"{v.releaselevel[0]}{v.serial}"
print(json.dumps({
    "implementation_name": sys.implementation.name,
    "implementation_version": f"{v.major}.{v.minor}.{v.micro}{level}",
    "os_name": os.name,
    "platform_machine": platform.machine(),
    "platform_release": platform.release(),
    "platform_system": platform.system(),
    "platform_version": platform.version(),
    "python_full_version": platform.python_version(),
    "platform_python_implementation": platform.python_implementation(),
    "python_version": ".".join(platform.python_version_tuple()[:2]),
    "sys_platform": sys.platform,
}))
"""


def _target_environment(python: Path) -> dict[str, str]:
    """The PEP 508 marker environment of the interpreter that installs the generation.

    Asked of *python* rather than taken from this process, because PM can run under a
    bootstrap Python (an older Termux interpreter reports ``linux``) while uv evaluates
    markers against *python*.
    """
    import subprocess

    from pm.filesystem import native

    probe = subprocess.run([native(python), "-I", "-c", _MARKER_ENVIRONMENT_PROBE],
                           capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
    return json.loads(probe.stdout)


def _solve_only_for(uv_policy: dict, target: Mapping[str, str]) -> None:
    """Narrow ``tool.uv.environments`` to the platform of the interpreter that installs the generation.

    uv solves every platform the environments admit, and since uv 0.12.14 it fails the whole
    lock when one fork has no wheels for a dependency: onnxruntime and torch ship none for
    Android, so a Linux or macOS update evicted the plugin over a platform it never runs on.
    A generation installs into one interpreter on one machine (*target*, from
    ``_target_environment``); only core's committed lock has to cover every platform, and it
    never passes through here.
    """
    host = f"sys_platform == {target['sys_platform']!r} and platform_machine == {target['platform_machine']!r}"
    uv_policy["environments"] = [f"({entry}) and {host}" for entry in uv_policy.get("environments") or []] or [host]


def _is_member_candidate(plugin_dir: Path) -> bool:
    return read_python_declaration(plugin_dir).installs_packages


def enabled_plugin_entries(*, proposed_home=None, enabled=None, disabled=None,
                           installing: Path | None = None,
                           skip_invalid_secondary: bool = False) -> list[tuple[Path, str, Path]]:
    """``(home plugins dir, selection key, plugin dir)`` for every selected plugin, in config order.

    The key is what the home's config names, so a caller can edit that home's selection.
    """
    from pm.plugins_state import enabled_plugins_ordered

    selection = enabled_plugins_ordered(
        proposed_home=proposed_home, enabled=enabled, disabled=disabled, installing=installing,
        skip_invalid_secondary=skip_invalid_secondary,
    )
    entries = []
    for plugins_dir, names in selection.items():
        for name in names:
            plugin_dir = selected_plugin_dir(plugins_dir, name, installing=installing)
            if plugin_dir is not None:
                entries.append((plugins_dir, name, plugin_dir))
    return entries


def selected_plugin_dir(plugins_dir: Path, name: str, *, installing: Path | None = None) -> Path | None:
    """The tree a home's selection key names: its own plugins dir first, then core's, else None."""
    from pm.plugins_state import _is_directory

    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise InstallError("venv", f"invalid plugin key: {name}")
    plugin_dir = plugins_dir / relative
    if installing is not None and plugin_dir.resolve() == installing.resolve():
        return plugin_dir
    if not _is_directory(plugin_dir):
        plugin_dir = paths.repo_root() / "plugins" / relative
    return plugin_dir if _is_directory(plugin_dir) else None


def enabled_plugin_dirs(*, proposed_home=None, enabled=None, disabled=None,
                        installing: Path | None = None, skip_invalid_secondary: bool = False) -> list[Path]:
    """Resolve the effective plugin selection without filtering dependency declarations."""
    entries = enabled_plugin_entries(proposed_home=proposed_home, enabled=enabled, disabled=disabled,
                                     installing=installing, skip_invalid_secondary=skip_invalid_secondary)
    return list(dict.fromkeys(plugin_dir for _plugins_dir, _name, plugin_dir in entries))


def enabled_member_dirs(*, proposed_home=None, enabled=None, disabled=None) -> list[Path]:
    """Keep every selected member or refuse an incompatible selection.

    A member that does not accept the running Hermes (``hermes_fit_error``) sits out instead:
    the verdict is only as good as our version identity (an untagged source checkout reads
    as an older release), the loader skips that plugin anyway, and the member rejoins as
    soon as the verdict flips. Enabling one is still refused at admission. Judged with no
    core extra selected; ``build_member_dirs`` adds the extras a build selects.
    """
    selected = enabled_plugin_dirs(proposed_home=proposed_home, enabled=enabled, disabled=disabled,
                                   skip_invalid_secondary=proposed_home is None)
    members = []
    for path in selected:
        declaration = read_python_declaration(path)
        if hermes_fit_error(declaration, ()):
            continue
        reason = manifest_version_error(declaration.manifest, path.name)
        if reason:
            raise InstallError("venv", reason)
        if declaration.installs_packages:
            members.append(path)
    return members


def build_member_dirs(extras: Iterable[str], python: Path | None = None) -> list[Path]:
    """``enabled_member_dirs`` for a build selecting core *extras*: a plugin extra they select
    can add a ``hermes-agent`` requirement the running Hermes misses, which sits out too.
    A build passes its interpreter *python*, whose markers the generation (and so the loader)
    sees; PM itself may run under a bootstrap Python."""
    extras, members = list(extras), enabled_member_dirs()
    target = _target_environment(python) if members and python is not None else None
    return [path for path in members
            if not core_requirement_error(read_python_declaration(path), extras, target)]


def _member_key(identity: Path) -> str:
    """``<plugin dir name>-<sha256(path)[:16]>``: the hash keeps two same-named plugins from
    different homes apart; the name is what a user sees in uv's conflict text
    (``hermes-plugin-<key> depends on …``) — a bare hash told them nothing to disable."""
    import re

    digest = hashlib.sha256(str(identity.resolve()).encode()).hexdigest()[:16]
    name = re.sub(r"[^a-z0-9._-]+", "-", identity.name.lower()).strip("-.") or "plugin"
    return f"{name}-{digest}"


def _selected_member_extras(document: dict, extras: Iterable[str],
                            target: Mapping[str, str] | None = None) -> tuple[set[str], set[str]]:
    """The member's extras this generation installs, and those it only names.

    Installed: those ``uv sync --extra`` names (it applies a name to every member declaring
    it), closed over the member's own ``name[extra]`` references, which install whatever
    references them. A reference is evaluated on *target* (the installing interpreter's
    marker environment) under every extra this sync installs, so a base dependency
    ``plugin[feature]; extra == 'runtime'`` installs ``feature`` once ``runtime`` is in, and
    ``plugin[windows]; sys_platform == 'win32'`` installs no ``windows`` on Linux; that extra
    is returned as named, because the build backend still validates the reference against
    the extras table. With no *target*, or a marker it cannot evaluate, the reference installs.
    """
    from packaging.utils import canonicalize_name

    from pm.install import _extra_key

    project = document.get("project", {})
    declared = {_extra_key(key): specs for key, specs in project.get("optional-dependencies", {}).items()}
    own = canonicalize_name(str(project.get("name", "")))
    selected = {_extra_key(extra) for extra in extras} & declared.keys()
    references = {"": _self_references(project.get("dependencies", []), own)}
    while True:
        for key in selected - references.keys():
            references[key] = _self_references(declared[key], own)
        contexts = ("", *sorted(selected))
        reached = {key for requirements in references.values() for requirement in requirements
                   if _reference_active(requirement, contexts, target)
                   for key in {_extra_key(name) for name in requirement.extras} & declared.keys()}
        if reached <= selected:
            break
        selected |= reached
    named = {key for requirements in references.values() for requirement in requirements
             for key in {_extra_key(name) for name in requirement.extras} & declared.keys()}
    return selected, named - selected


def _self_references(specs: Iterable[str], own: str) -> list:
    """The requirements among *specs* that name the project *own* itself."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    references = []
    for spec in specs:
        try:
            requirement = Requirement(spec)
        except InvalidRequirement:
            continue  # uv reports it; this pass only decides what to keep.
        if canonicalize_name(requirement.name) == own:
            references.append(requirement)
    return references


def _reference_active(requirement, contexts: Iterable[str], target: Mapping[str, str] | None) -> bool:
    """Whether *requirement* applies on *target* under any of the installed extra *contexts*.

    uv lowers an extra's requirements to ``marker and extra == '<that extra>'``, and that extra
    is among *contexts* once it is reached, so one evaluation per installed extra covers base
    and extra requirements alike; a conjunction over two extras is kept, never dropped.
    """
    from packaging.markers import UndefinedComparison, UndefinedEnvironmentName

    if requirement.marker is None or target is None:
        return True
    try:
        return any(requirement.marker.evaluate({**target, "extra": extra}) for extra in contexts)
    except (UndefinedComparison, UndefinedEnvironmentName):
        return True


def _prune_member_sources(settings: dict, installed: set[str]) -> bool:
    """Drop ``tool.uv.sources`` alternatives for an extra or group this generation removed.

    uv refuses a source qualified by an extra or group the project no longer declares
    ("Source entry ... only applies to extra `dev`, but the `dev` extra does not exist"), so a
    pruned development extra must take its sources with it. An extra declared but left empty
    installs nothing, so its sources go too. A package keeps its key while any alternative
    remains; unqualified (runtime) alternatives always stay.
    """
    from pm.install import _extra_key

    sources = settings.get("sources")
    if not isinstance(sources, dict):
        return False

    def applies(spec) -> bool:
        if not isinstance(spec, dict):
            return True
        if "group" in spec:
            return False
        return "extra" not in spec or _extra_key(str(spec["extra"])) in installed

    pruned = {}
    for package, entry in sources.items():
        if isinstance(entry, list):
            alternatives = [spec for spec in entry if applies(spec)]
            if alternatives:
                pruned[package] = alternatives
        elif applies(entry):
            pruned[package] = entry
    if pruned == sources:
        return False
    if pruned:
        settings["sources"] = pruned
    else:
        settings.pop("sources")
    return True


def _rename_virtual_member(document: dict, name: str) -> None:
    """Give a virtual member its unique workspace *name*, its self-references included.

    ``dependencies = ["myplugin[runtime]"]`` in a project named ``myplugin`` installs the
    member's own ``runtime`` extra. Left on the declared name after the rename, uv reads it as
    a different package: a lock failure when no index has that name, or someone else's
    package installed in place of the extra when one does. Extras, specifiers and markers
    stay as written; only the leading name changes.
    """
    import re

    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    project = document.setdefault("project", {})
    own = canonicalize_name(str(project.get("name", "")))
    project["name"] = name
    if not own:
        return

    def renamed(spec):
        try:
            if not isinstance(spec, str) or canonicalize_name(Requirement(spec).name) != own:
                return spec
        except InvalidRequirement:
            return spec  # uv reports it; renaming cannot make it valid.
        return re.sub(r"^\s*[A-Za-z0-9][A-Za-z0-9._-]*", name, spec, count=1)

    if "dependencies" in project:
        project["dependencies"] = [renamed(spec) for spec in project["dependencies"]]
    for extra, specs in project.get("optional-dependencies", {}).items():
        project["optional-dependencies"][extra] = [renamed(spec) for spec in specs]
    for conflict in document.get("tool", {}).get("uv", {}).get("conflicts", []):
        for item in conflict:
            if isinstance(item, dict) and canonicalize_name(str(item.get("package", ""))) == own:
                item["package"] = name


def _runtime_member_metadata(document: dict, extras: Iterable[str],
                             target: Mapping[str, str] | None = None) -> bool:
    """Keep what this generation installs, not a plugin checkout's development environment.

    uv locks every declared group and extra of every member even when sync excludes it, so a
    test pin (``dev = ["httpx<0.28"]``) would fail the lock against core's runtime pins. Sync
    never installs a member's groups (core's ``default-groups = []`` covers the workspace) and
    installs only the extras named by ``--extra``. The original plugin source stays untouched.
    """
    from pm.install import _extra_key

    changed = "dependency-groups" in document
    document.pop("dependency-groups", None)
    project = document.get("project", {})
    optional = project.get("optional-dependencies")
    keep: set[str] = set()
    if optional:
        keep, named = _selected_member_extras(document, extras, target)
        # A named-only extra stays declared but empty: its pins cannot reach the lock.
        kept = {key: specs if _extra_key(key) in keep else [] for key, specs in optional.items()
                if _extra_key(key) in keep | named}
        if kept != optional:
            changed = True
            if kept:
                project["optional-dependencies"] = kept
            else:
                project.pop("optional-dependencies")
    kept_extras = {_extra_key(key) for key in project.get("optional-dependencies", {})}
    settings = document.get("tool", {}).get("uv", {})
    changed = _prune_member_sources(settings, keep) or changed
    for key in ("default-groups", "dependency-groups", "dev-dependencies"):
        if key in settings:
            settings.pop(key)
            changed = True
    if "conflicts" in settings:
        conflicts = [[spec for spec in conflict
                      if "group" not in spec and ("extra" not in spec or _extra_key(spec["extra"]) in kept_extras)]
                     for conflict in settings["conflicts"]]
        conflicts = [conflict for conflict in conflicts if len(conflict) > 1]
        if conflicts != settings["conflicts"]:
            if conflicts:
                settings["conflicts"] = conflicts
            else:
                settings.pop("conflicts")
            changed = True
    return changed


def _recorded_extras() -> list[str]:
    """The extras of the generation this install selected: what a gate asked outside a build
    (admission, the loader) evaluates a plugin's extras under."""
    from pm.environments import runtime_facts_path
    from pm.lock import Facts

    extras = (Facts(runtime_facts_path(paths.repo_root())).get("venv") or {}).get("extras")
    return [extra for extra in extras if isinstance(extra, str)] if isinstance(extras, list) else []


def core_requirement_error(declaration, extras: Iterable[str] | None = None,
                           target: Mapping[str, str] | None = None) -> Optional[str]:
    """Why the plugin's applicable requirement on ``hermes-agent`` rejects the running Hermes.

    PM satisfies ``hermes-agent`` from the checkout itself: a runtime requirement naming it is
    dropped, and a selected extra naming it binds to the workspace root, where uv ignores the
    specifier. So the specifier is judged here, PEP 440 like any requirement, under *extras*
    (the core extras the build selects; ``None`` reads the recorded selection) and the markers
    of *target* (the installing interpreter; ``None`` is this process, which the loader runs
    in). The identity is the one ``requires_hermes`` reads (the committed pyproject version is
    a placeholder); with none (``unknown``) nothing is judged. It is read only for a plugin
    that names core: the read walks git, and every plugin passes here at startup.
    """
    import tomllib

    from packaging.markers import default_environment
    from packaging.version import InvalidVersion, Version

    from hermes_cli.plugins_manifest import running_hermes_version
    from pm.install import _extra_key

    if declaration.external:
        return None
    environment = default_environment() if target is None else target
    specs, installed = list(declaration.requirements), set()
    if declaration.pyproject is not None:
        document = tomllib.loads(declaration.pyproject.read_text(encoding="utf-8-sig"))
        installed = _selected_member_extras(document, _recorded_extras() if extras is None else extras, environment)[0]
        specs += [spec for name, values in document["project"].get("optional-dependencies", {}).items()
                  if _extra_key(name) in installed for spec in values]
    pins = [requirement for requirement in _self_references(specs, "hermes-agent")
            if requirement.specifier and _reference_active(requirement, ("", *sorted(installed)), environment)]
    if not pins:
        return None
    current = running_hermes_version()
    try:
        Version(current)
    except InvalidVersion:
        return None
    rejected = [str(requirement) for requirement in pins
                if not requirement.specifier.contains(current, prereleases=True)]
    return f"requires {rejected[0]}, running {current}" if rejected else None


def hermes_fit_error(declaration, extras: Iterable[str] | None = None) -> Optional[str]:
    """Why this Hermes is not one the plugin accepts: its ``requires_hermes`` or an applicable
    ``hermes-agent`` requirement rejects the running version. Either verdict is only as good as
    our version identity, so a selected misfit sits out of the build (config untouched) and
    rejoins once it flips; admission refuses it."""
    from hermes_cli.plugins_manifest import requires_hermes_error

    return requires_hermes_error(declaration.manifest) or core_requirement_error(declaration, extras)


def _workspace_member(plugin_dir: Path, root: Path, *, identity: Path, extras: Iterable[str] = (),
                      target: Mapping[str, str] | None = None) -> Path:
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
        changed = False
        if declaration.install_requirements != declaration.requirements:
            document["project"]["dependencies"] = list(declaration.install_requirements)
            changed = True
        if virtual:
            # Before the extras pass, so its self-reference closure reads the names uv will see.
            _rename_virtual_member(document, f"hermes-plugin-{key}")
        changed = _runtime_member_metadata(document, extras, target) or changed
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
        project = document.get("project")
        # uv rejects any [project] table whose version is neither set nor dynamic, at
        # parse time and for buildable members too, so a version-less member would abort
        # every `uv lock` of the workspace. Give it the same inert version plugin-deps use.
        # PEP 621 forbids a static version while "version" is listed under
        # project.dynamic, so an explicitly dynamic version is left alone.
        if project is not None and "version" not in project and "version" not in project.get("dynamic", ()):
            project["version"] = "0.0.0"
            changed = True
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


def core_lock_upgrades(core_lock: Path, seed_lock: Path) -> list[str]:
    """``uv lock -P`` requirements moving a seeded generation onto core's newer pins.

    uv keeps every seeded pin that still resolves, so a core uv.lock bump (a security fix in a
    transitive) never reached an install with plugin members (#91424). Only an index package
    core's lock holds at a version newer than every pin the seed has for it moves: a package
    only a plugin brings is not core's to pin, and one a plugin moved above core's pin stays
    where the plugin needs it. Path, git and URL sources follow core's manifest already.
    """
    import tomllib

    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    def entries(lock: Path) -> list[dict]:
        return tomllib.loads(lock.read_text(encoding="utf-8-sig")).get("package", [])

    seeded: dict[str, list[Version]] = {}
    forks: dict[tuple[str, tuple[str, ...]], list[Version]] = {}
    for package in entries(seed_lock):
        if "registry" not in package.get("source", {}):
            continue
        try:
            name, pinned = canonicalize_name(package["name"]), Version(package["version"])
        except (KeyError, InvalidVersion):
            continue
        seeded.setdefault(name, []).append(pinned)
        forks.setdefault((name, tuple(package.get("resolution-markers", []))), []).append(pinned)
    upgrades = []
    for package in entries(core_lock):
        name = canonicalize_name(package.get("name", ""))
        if "registry" not in package.get("source", {}) or name not in seeded:
            continue
        try:
            version = Version(package["version"])
        except (KeyError, InvalidVersion):
            continue
        # A platform fork compares against the seed's same fork, so a bump to one fork still moves.
        prior = forks.get((name, tuple(package.get("resolution-markers", []))), seeded[name])
        if all(version > pinned for pinned in prior):
            # A name core forks by platform carries each fork's markers, or the exact pins clash.
            markers = " or ".join(f"({marker})" for marker in package.get("resolution-markers", []))
            upgrades.append(f"{name}=={version}" + (f"; {markers}" if markers else ""))
    return upgrades


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
    upgrade: Sequence[str] = (),
) -> None:
    """Prepare a fresh generation using explicit inputs and a prepared engine.

    The caller selects the seed; uv retains its compatible versions except the
    *upgrade* requirements (``uv lock -P``), which bind like core pins. Repair
    copies the recorded build inputs and never reads current manifests.
    Resolver conflicts remain distinct from download/build failures.
    """
    if root.exists() or root.is_symlink():
        raise InstallError("venv", f"workspace must be fresh: {root}")
    if replay is None:
        _generate_pyproject(plugin_dirs, root, source=source, python=environment.python, extras=extras)
        if seed_lock is not None:
            (root / "uv.lock").write_bytes(seed_lock.read_bytes())
    else:
        if not (replay / "pyproject.toml").is_file() or not (replay / "uv.lock").is_file():
            raise InstallError("venv", f"recorded workspace is missing: {replay}")
        # Sibling generations keep external relative paths at the same depth;
        # snapshotted members and their exact lock travel with the workspace.
        # Use the snapshot's exclusions: build/ may hold an in-tree backend.
        shutil.copytree(replay, root, symlinks=True, ignore=_member_ignored)
        # A generation recorded before the cutoff left the user's machine still carries it.
        manifest = root / "pyproject.toml"
        manifest.write_text(_without_release_age(manifest.read_text(encoding="utf-8-sig")), encoding="utf-8")
        frozen = True

    environment.sync(root, extras=extras, frozen=frozen, upgrade=() if frozen else upgrade)

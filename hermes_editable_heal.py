"""Heal a dangling setuptools editable finder after an interrupted update.

Stdlib-only. Safe to import on a half-broken venv.

An interrupted ``hermes update`` can uninstall physical copies of first-party
packages from site-packages while leaving ``__editable___*_finder.py`` mapping
those names at the now-deleted paths. The ``hermes`` console script then dies
with ``ModuleNotFoundError: No module named 'hermes_cli'`` before
``hermes_cli._early_recovery`` can run (#97819).

This module:

- Retargets ``MAPPING`` / ``NAMESPACES`` at the git checkout recorded in
  ``hermes_agent-*.dist-info/direct_url.json`` (or an explicit project root).
- Never accepts a first-party mapping that lives *inside* site-packages.
- Drops a physical ``__hermes_editable_heal.pth`` sidecar next to the
  generated finder so the next interpreter start heals *before* the
  console script imports ``hermes_cli.main``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FINDER_GLOB = "__editable___hermes_agent_*_finder.py"
DIST_INFO_GLOB = "hermes_agent-*.dist-info"
HOOK_MODULE = "__hermes_editable_heal.py"
HOOK_PTH = "__hermes_editable_heal.pth"
CANONICAL_MODULE = "hermes_editable_heal.py"
# Thin sidecar: never a copy of this file. Loaded by the .pth before the
# console script imports hermes_cli. Loads the canonical module by path so a
# dangling finder cannot hide it, then calls heal() once.
_THIN_HOOK = """\
from pathlib import Path
import importlib.util

def _run() -> None:
    src = Path(__file__).resolve().parent / "hermes_editable_heal.py"
    if not src.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        "_hermes_editable_heal_impl", src
    )
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.heal()

try:
    _run()
except Exception:
    pass
"""
_PTH_LINE = "import __hermes_editable_heal\n"

_ASSIGN_RE = re.compile(
    r"^(?P<name>MAPPING|NAMESPACES)(?P<ann>:[^=]+)? = (?P<value>.+)$",
    re.MULTILINE,
)


def _unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        out.append(path)
    return out


def site_packages_dirs(
    project_root: Path | None = None,
    *,
    extra: Path | None = None,
) -> list[Path]:
    """Likely site-packages dirs for this install.

    When *project_root* is set, only that project's ``venv`` / ``.venv``
    (plus *extra*) are scanned — never the running interpreter's prefix.
    A test or recovery call must not rewrite some other checkout's finder.
    When neither is set (startup ``.pth`` hook), scan ``sys.prefix`` only.
    """
    found: list[Path] = []
    if extra is not None:
        found.append(Path(extra))
    if project_root is not None:
        root = Path(project_root)
        for name in ("venv", ".venv"):
            venv = root / name
            found.extend(venv.glob("lib/python*/site-packages"))
            found.append(venv / "Lib" / "site-packages")
        return _unique_existing(found)
    if extra is None:
        prefix = Path(sys.prefix)
        found.extend(prefix.glob("lib/python*/site-packages"))
        found.append(prefix / "Lib" / "site-packages")
    return _unique_existing(found)


def _file_url_to_path(url: str) -> Path | None:
    if not url.startswith("file:"):
        return None
    rest = url[5:]
    if rest.startswith("///"):
        rest = rest[2:]  # file:///X -> /X  (POSIX) or /C:/... (Windows)
    elif rest.startswith("//"):
        rest = rest[1:]
    if rest.startswith("/") and len(rest) >= 3 and rest[2] == ":":
        rest = rest[1:]  # /C:/Users/... -> C:/Users/...
    try:
        from urllib.parse import unquote

        rest = unquote(rest)
    except Exception:
        pass
    path = Path(rest)
    return path if path.exists() else path


def read_editable_project_root(site_packages: Path) -> Path | None:
    """Repo root from the editable install's ``direct_url.json``, if present."""
    for info in site_packages.glob(f"{DIST_INFO_GLOB}/direct_url.json"):
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        dir_info = data.get("dir_info") or {}
        if isinstance(dir_info, dict) and dir_info.get("editable") is False:
            continue
        path = _file_url_to_path(str(data.get("url") or ""))
        if path is None:
            continue
        if (path / "pyproject.toml").is_file():
            return path.resolve()
    return None


def resolve_project_root(
    hint: str | Path | None = None,
    *,
    site_packages: Path | None = None,
) -> str | None:
    """Best checkout root when ``hermes_cli`` was imported from site-packages."""
    if hint is not None:
        hinted = Path(hint)
        if (hinted / "pyproject.toml").is_file():
            return str(hinted.resolve())
        from_hint = read_editable_project_root(hinted)
        if from_hint is not None:
            return str(from_hint)
    dirs = [site_packages] if site_packages is not None else site_packages_dirs()
    for sp in dirs:
        if sp is None:
            continue
        root = read_editable_project_root(Path(sp))
        if root is not None:
            return str(root)
    return None


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _repo_target(repo_root: Path, name: str) -> Path | None:
    pkg = repo_root / name
    if pkg.is_dir():
        return pkg
    module = repo_root / f"{name}.py"
    if module.is_file():
        return repo_root / name
    return None


def _target_is_stale(target: Path, site_packages: Path) -> bool:
    if not target.exists():
        return True
    return _is_under(target, site_packages)


def retarget_mapping(
    mapping: dict,
    repo_root: Path,
    site_packages: Path,
) -> tuple[dict, bool]:
    """Rewrite finder MAPPING values that are missing or inside site-packages."""
    changed = False
    out: dict = {}
    for name, raw in mapping.items():
        dest = Path(str(raw))
        replacement = _repo_target(repo_root, str(name))
        if replacement is not None and _target_is_stale(dest, site_packages):
            out[name] = str(replacement)
            changed = True
        else:
            out[name] = raw
    return out, changed


def retarget_namespaces(
    namespaces: dict,
    repo_root: Path,
    site_packages: Path,
) -> tuple[dict, bool]:
    changed = False
    out: dict = {}
    for name, raw_paths in namespaces.items():
        paths = list(raw_paths) if isinstance(raw_paths, (list, tuple)) else [raw_paths]
        new_paths: list[str] = []
        for raw in paths:
            dest = Path(str(raw))
            if not _target_is_stale(dest, site_packages):
                new_paths.append(str(raw))
                continue
            # tools.wakewords -> repo_root / tools / wakewords
            rel = Path(*str(name).split("."))
            candidate = repo_root / rel
            if candidate.exists():
                new_paths.append(str(candidate))
                changed = True
            else:
                new_paths.append(str(raw))
        out[name] = new_paths
    return out, changed


def _replace_assignment(source: str, name: str, value: object) -> str:
    def _sub(match: re.Match[str]) -> str:
        if match.group("name") != name:
            return match.group(0)
        ann = match.group("ann") or ""
        return f"{name}{ann} = {repr(value)}"

    new, count = _ASSIGN_RE.subn(_sub, source)
    if count == 0:
        raise ValueError(f"could not find {name} assignment in finder")
    return new


def rewrite_finder_file(
    finder_path: Path,
    *,
    mapping: dict | None = None,
    namespaces: dict | None = None,
) -> None:
    text = finder_path.read_text(encoding="utf-8")
    if mapping is not None:
        text = _replace_assignment(text, "MAPPING", mapping)
    if namespaces is not None:
        text = _replace_assignment(text, "NAMESPACES", namespaces)
    finder_path.write_text(text, encoding="utf-8")


def _apply_to_loaded_finder(mapping: dict, namespaces: dict | None) -> None:
    for mod in list(sys.modules.values()):
        mapping_obj = getattr(mod, "MAPPING", None)
        if not isinstance(mapping_obj, dict):
            continue
        name = getattr(mod, "__name__", "")
        file = getattr(mod, "__file__", "") or ""
        if "editable" not in name and "editable" not in file.replace("\\", "/"):
            continue
        mapping_obj.clear()
        mapping_obj.update(mapping)
        if namespaces is not None:
            ns = getattr(mod, "NAMESPACES", None)
            if isinstance(ns, dict):
                ns.clear()
                ns.update(namespaces)


def heal_finder(
    finder_path: Path,
    repo_root: Path,
    site_packages: Path,
) -> bool:
    """Rewrite one finder file. Returns True when the on-disk mapping changed."""
    ns: dict = {}
    mapping: dict = {}
    try:
        source = finder_path.read_text(encoding="utf-8")
        # Exec just the two assignments in an isolated dict — the file is
        # generated by setuptools and is data, not a trust boundary we invent.
        isolated: dict = {}
        code = "\n".join(
            f"{m.group('name')} = {m.group('value')}" for m in _ASSIGN_RE.finditer(source)
        )
        exec(code, {}, isolated)  # noqa: S102 — parser for generated finder
        mapping = isolated.get("MAPPING") or {}
        ns = isolated.get("NAMESPACES") or {}
    except Exception:
        return False
    if not isinstance(mapping, dict):
        return False

    new_mapping, map_changed = retarget_mapping(mapping, repo_root, site_packages)
    new_ns, ns_changed = retarget_namespaces(
        ns if isinstance(ns, dict) else {}, repo_root, site_packages
    )
    if not map_changed and not ns_changed:
        _apply_to_loaded_finder(new_mapping, new_ns if isinstance(ns, dict) else None)
        return False
    try:
        rewrite_finder_file(
            finder_path,
            mapping=new_mapping,
            namespaces=new_ns if ns_changed or map_changed else None,
        )
    except (OSError, ValueError):
        return False
    _apply_to_loaded_finder(new_mapping, new_ns)
    return True


def _write_if_changed(path: Path, text: str) -> None:
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.write_text(text, encoding="utf-8")


def _canonical_source() -> str:
    """Source of hermes_editable_heal.py — never the thin sidecar."""
    here = Path(__file__).resolve()
    if here.name == HOOK_MODULE:
        sibling = here.parent / CANONICAL_MODULE
        return sibling.read_text(encoding="utf-8")
    return here.read_text(encoding="utf-8")


def install_startup_hook(site_packages: Path) -> None:
    """Write an idempotent .pth hook that heals before console-script import.

    The sidecar is a fixed thin loader. It must not be a growing copy of this
    module: ``heal()`` runs on every interpreter start and would otherwise
    append another ``try: heal()`` footer each launch.
    """
    _write_if_changed(site_packages / CANONICAL_MODULE, _canonical_source())
    _write_if_changed(site_packages / HOOK_MODULE, _THIN_HOOK)
    _write_if_changed(site_packages / HOOK_PTH, _PTH_LINE)


def heal(
    project_root: Path | str | None = None,
    site_packages: Path | str | None = None,
) -> bool:
    """Heal every hermes-agent editable finder we can see. Never raises."""
    try:
        extra = Path(site_packages) if site_packages is not None else None
        if extra is None:
            here = Path(__file__).resolve().parent
            if any(here.glob(FINDER_GLOB)):
                extra = here
        root = Path(project_root) if project_root is not None else None
        if root is not None and not (root / "pyproject.toml").is_file():
            root = None
        dirs = site_packages_dirs(root, extra=extra)
        healed = False
        for sp in dirs:
            repo = root or read_editable_project_root(sp)
            if repo is None:
                continue
            for finder in sp.glob(FINDER_GLOB):
                if heal_finder(finder, repo, sp):
                    healed = True
            try:
                install_startup_hook(sp)
            except OSError:
                pass
        return healed
    except Exception:
        return False

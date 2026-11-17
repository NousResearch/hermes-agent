"""Read-only inspection of a plugin source before install: which halves (agent, desktop) it ships.

The gateway's answer to the install dialog's repository check, for a client with no Git of its own
(the browser-hosted app). Native Desktop runs the same check in Electron
(``apps/desktop/electron/desktop-plugin-install.ts`` ``probePluginRepo``); the detection rules and the
result shape match it. The source goes through the installer's own resolution, kill list and
clone (non-interactive, credentials scrubbed, ``plugins.clone_timeout_seconds``) into a temporary
directory outside every profile's ``plugins/``. Nothing is installed, imported or run: only the
manifest files are read, as text.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

import hermes_yaml as yaml

#: A manifest is a few hundred bytes; never read a hostile multi-megabyte one into memory.
_MANIFEST_MAX_BYTES = 256 * 1024
#: How long a successful inspection answers a repeat of the same request (see ``inspect_plugin_source``).
_RESULT_TTL_SECONDS = 600
#: ``(home, catalog_name, identifier)`` -> ``[future, finished_at]``; ``finished_at`` is None while it runs.
_inspections: dict[tuple[str, str, str], list] = {}
_inspections_lock = threading.Lock()


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _read_manifest(path: Optional[Path], loads) -> dict[str, Any]:
    """The manifest as a dict when the file is a small regular file that parses, else ``{}``."""
    if path is None:
        return {}
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MANIFEST_MAX_BYTES:
            return {}
        data = loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _manifest_name(data: dict[str, Any]) -> Optional[str]:
    name = data.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


#: Files beside a ``plugin.yaml`` that the agent side loads: a ``register(ctx)`` package, or a dashboard
#: tab/backend (``dashboard/manifest.json``, often with a ``plugin_api.py``) the gateway mounts.
_AGENT_PAYLOAD_FILES = ("__init__.py", "dashboard/manifest.json")


def _yaml_ships_agent_payload(root: Path, manifest: dict[str, Any]) -> bool:
    """Whether a ``plugin.yaml`` folder has something for the agent side to load. A manifest alone is not
    enough: desktop-only plugins ship one beside ``desktop/plugin.js`` for metadata, and installing those
    into the backend's ``plugins/`` puts down a package that loads nothing."""
    if any((root / rel).is_file() for rel in _AGENT_PAYLOAD_FILES):
        return True
    # The loader's manifest-only language pack (``_is_manifest_only_language_pack``), judged the way
    # ``hermes plugins validate`` does: declared locales plus the folder that holds them.
    return bool(manifest.get("provides_locales")) and (root / "locales").is_dir()


def detect_components(root: Path, fallback_name: str) -> dict[str, Any]:
    """Electron's ``detectPluginComponents``: an agent half is a portable ``plugin.json``, or a
    ``plugin.yaml``/``plugin.yml`` with an agent payload beside it (:func:`_yaml_ships_agent_payload`);
    a desktop half is ``plugin.js`` at the root or in ``desktop/``. ``__init__.py`` is one payload, not
    a requirement: a manifest-only package with a ``dashboard/plugin_api.py`` backend is installable."""
    yaml_path = next((root / n for n in ("plugin.yaml", "plugin.yml") if (root / n).is_file()), None)
    yaml_manifest = _read_manifest(yaml_path, yaml.safe_load)
    portable = root / "plugin.json"
    agent = (yaml_path is not None and _yaml_ships_agent_payload(root, yaml_manifest)) or portable.is_file()
    desktop = (root / "plugin.js").is_file() or (root / "desktop" / "plugin.js").is_file()
    agent_name = None
    if agent:
        agent_name = (_manifest_name(yaml_manifest) or _manifest_name(_read_manifest(portable, json.loads))
                      or fallback_name)
    return {"agent": agent, "desktop": desktop, "agent_name": agent_name}


def _desktop_folder_name(git_url: str, subdir: Optional[str]) -> str:
    """Electron's ``desktopPluginFolderName``: the subdirectory's last real segment, else the repo name."""
    parts = [p for p in (subdir or "").replace("\\", "/").split("/") if p and p not in (".", "desktop")]
    return parts[-1] if parts else _pc()._repo_name_from_url(git_url)


def _failure(error: str, warnings: list[str], insecure: bool) -> dict[str, Any]:
    return {"ok": False, "agent": False, "desktop": False, "warnings": warnings, "insecure": insecure,
            "error": error}


def inspect_plugin_source(identifier: str, *, catalog_name: Optional[str] = None) -> dict[str, Any]:
    """``{ok, agent, desktop, agent_name, desktop_name, sha, warnings, insecure}``, or ``ok: False`` with
    ``error``. *catalog_name* inspects the curated entry at its reviewed pin (its repo replaces
    *identifier*), so the dialog shows exactly what the catalog install will put down. ``sha`` is the
    commit that was read: a source that moves after review must not install a tree nobody inspected,
    so the install is held to it, as native Desktop's probe is.

    A repeat of a request (same profile home, catalog name and source) joins the clone still running
    for it, or reads a successful answer it produced within :data:`_RESULT_TTL_SECONDS`. The browser
    stops waiting before a slow clone can finish (``plugins.clone_timeout_seconds`` reaches an hour per
    Git step); its Retry must collect that clone's answer, not start another under the same deadline.
    A failed read is never kept, so a Retry after one inspects afresh."""
    from hermes_constants import get_hermes_home
    key = (str(get_hermes_home()), catalog_name or "", identifier)
    now = time.monotonic()
    with _inspections_lock:
        for stale in [k for k, (_, done) in _inspections.items()
                      if done is not None and now - done > _RESULT_TTL_SECONDS]:
            del _inspections[stale]
        slot = _inspections.get(key)
        owner = slot is None
        if owner:
            slot = _inspections[key] = [Future(), None]
    future: Future = slot[0]
    if not owner:
        return _copy(future.result())
    try:
        result = _inspect(identifier, catalog_name)
    except BaseException as exc:
        with _inspections_lock:
            _inspections.pop(key, None)
        future.set_exception(exc)
        raise
    with _inspections_lock:
        if result["ok"]:
            slot[1] = time.monotonic()
        else:
            _inspections.pop(key, None)
    future.set_result(result)
    return _copy(result)


def _copy(result: dict[str, Any]) -> dict[str, Any]:
    """Each caller gets its own answer: one shared by joined requests must not be edited through another."""
    return {**result, "warnings": list(result["warnings"])}


def forget_inspection(identifier: str, *, catalog_name: Optional[str] = None) -> None:
    """Drop the kept answer for this source in the current profile home once an install consumed it: a
    source that moved since must be read afresh when the dialog re-opens, not served for the rest of
    :data:`_RESULT_TTL_SECONDS`. A read still running is left for the requests joined to it."""
    from hermes_constants import get_hermes_home
    key = (str(get_hermes_home()), catalog_name or "", identifier)
    with _inspections_lock:
        if _inspections.get(key, [None, None])[1] is not None:
            del _inspections[key]


def _inspect(identifier: str, catalog_name: Optional[str]) -> dict[str, Any]:
    pc = _pc()
    revision = None
    names: tuple[str, ...] = ()
    if catalog_name:
        from hermes_cli import plugins_cmd_catalog as catalog
        entry = catalog.get_live_catalog_entry(catalog_name)
        if entry is None:
            return _failure(f"'{catalog_name}' is not in the Hermes plugin catalog.", [], False)
        identifier, revision, names = entry.install_identifier, entry.sha or None, (entry.name,)
    try:
        git_url, subdir = pc._resolve_git_url(identifier)
    except ValueError as exc:
        return _failure(str(exc), [], False)
    insecure = git_url.startswith(("http://", "file://"))
    warnings = ["This URL uses an insecure or local scheme. Prefer https:// or git@ for production installs."] \
        if insecure else []
    try:
        from hermes_cli.plugins_cmd_catalog import raise_if_removed
        raise_if_removed(identifier, git_url, *names)
        with tempfile.TemporaryDirectory(prefix="hermes-plugin-inspect-") as tmp:
            clone = Path(tmp) / "plugin"
            sha = pc._clone_plugin_repo(clone, git_url, revision, subdir)
            root = pc._resolve_subdir_within(clone, subdir) if subdir else clone
            fallback = subdir.rstrip("/").rsplit("/", 1)[-1] if subdir else pc._repo_name_from_url(git_url)
            found = detect_components(root, fallback)
    except pc.PluginOperationError as exc:
        return _failure(str(exc), warnings, insecure)
    if not found["agent"] and not found["desktop"]:
        return _failure("No agent or desktop plugin artifacts found in this repository.", warnings, insecure)
    return {"ok": True, **found, "desktop_name": _desktop_folder_name(git_url, subdir) if found["desktop"] else None,
            "sha": sha, "warnings": warnings, "insecure": insecure}

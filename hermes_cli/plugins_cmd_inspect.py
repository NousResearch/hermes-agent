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
from pathlib import Path
from typing import Any, Optional

import hermes_yaml as yaml

#: A manifest is a few hundred bytes; never read a hostile multi-megabyte one into memory.
_MANIFEST_MAX_BYTES = 256 * 1024


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _manifest_name(path: Path, loads) -> Optional[str]:
    """The manifest's ``name`` when the file is a small regular file that parses, else None."""
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MANIFEST_MAX_BYTES:
            return None
        data = loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def detect_components(root: Path, fallback_name: str) -> dict[str, Any]:
    """Electron's ``detectPluginComponents``: an agent half is ``plugin.yaml``/``plugin.yml`` with
    ``__init__.py``, or a portable ``plugin.json``; a desktop half is ``plugin.js`` at the root or in
    ``desktop/``."""
    yaml_path = next((root / n for n in ("plugin.yaml", "plugin.yml") if (root / n).is_file()), None)
    portable = root / "plugin.json"
    agent = (yaml_path is not None and (root / "__init__.py").is_file()) or portable.is_file()
    desktop = (root / "plugin.js").is_file() or (root / "desktop" / "plugin.js").is_file()
    agent_name = None
    if agent:
        agent_name = (_manifest_name(yaml_path, yaml.safe_load) if yaml_path is not None else None) or (
            _manifest_name(portable, json.loads) if portable.is_file() else None) or fallback_name
    return {"agent": agent, "desktop": desktop, "agent_name": agent_name}


def _desktop_folder_name(git_url: str, subdir: Optional[str]) -> str:
    """Electron's ``desktopPluginFolderName``: the subdirectory's last real segment, else the repo name."""
    parts = [p for p in (subdir or "").replace("\\", "/").split("/") if p and p not in (".", "desktop")]
    return parts[-1] if parts else _pc()._repo_name_from_url(git_url)


def _failure(error: str, warnings: list[str], insecure: bool) -> dict[str, Any]:
    return {"ok": False, "agent": False, "desktop": False, "warnings": warnings, "insecure": insecure,
            "error": error}


def inspect_plugin_source(identifier: str, *, catalog_name: Optional[str] = None) -> dict[str, Any]:
    """``{ok, agent, desktop, agent_name, desktop_name, warnings, insecure}``, or ``ok: False`` with
    ``error``. *catalog_name* inspects the curated entry at its reviewed pin (its repo replaces
    *identifier*), so the dialog shows exactly what the catalog install will put down."""
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
            pc._clone_plugin_repo(clone, git_url, revision, subdir)
            root = pc._resolve_subdir_within(clone, subdir) if subdir else clone
            fallback = subdir.rstrip("/").rsplit("/", 1)[-1] if subdir else pc._repo_name_from_url(git_url)
            found = detect_components(root, fallback)
    except pc.PluginOperationError as exc:
        return _failure(str(exc), warnings, insecure)
    if not found["agent"] and not found["desktop"]:
        return _failure("No agent or desktop plugin artifacts found in this repository.", warnings, insecure)
    return {"ok": True, **found, "desktop_name": _desktop_folder_name(git_url, subdir) if found["desktop"] else None,
            "warnings": warnings, "insecure": insecure}

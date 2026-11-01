"""Plugins Hermes installed whose files are gone from disk.

``hermes plugins install`` writes an installer-owned record to ``plugins/.install-metadata.json``. When the
``plugins/<name>/`` tree later disappears (a manual ``rm -rf`` to clear a dependency conflict, a disk
cleanup), discovery no longer sees the plugin, so ``plugins list`` omitted it and ``enable``/``disable``/
``remove`` all answered "No plugin named" while ``plugins.enabled`` still named it. The record is the proof
it was installed and carries the source a reinstall needs.
"""

from __future__ import annotations

from typing import Any


def _pc():
    from hermes_cli import plugins_cmd
    return plugins_cmd


def missing_installed_plugins() -> dict[str, str]:
    """``{name: install source}`` for every installer record whose ``plugins/<name>/`` tree is gone."""
    pc = _pc()
    try:
        records = pc._read_install_metadata()
    except pc.PluginOperationError:
        return {}
    plugins_dir = pc._plugins_dir()
    return {
        name: str(record["source"])
        for name, record in records.items()
        if isinstance(record, dict) and isinstance(record.get("source"), str) and record["source"]
        and not (plugins_dir / name).exists()
    }


def enabled_missing_plugins() -> dict[str, str]:
    """The :func:`missing_installed_plugins` that ``plugins.enabled`` still turns on."""
    enabled = _pc()._get_enabled_set()
    return {name: source for name, source in missing_installed_plugins().items() if name in enabled}


def repair_hint(name: str, source: str) -> str:
    return f"reinstall: hermes plugins install {source}  ·  forget it: hermes plugins remove {name}"


def forget_missing_plugin(name: str) -> dict[str, Any] | None:
    """Drop a missing plugin's install record and config entries; ``None`` when *name* is not missing."""
    if name not in missing_installed_plugins():
        return None
    pc = _pc()
    pc._update_install_record(name, lambda _current: None)
    return {"ok": True, "name": name, **pc._forget_plugin_config({name})}

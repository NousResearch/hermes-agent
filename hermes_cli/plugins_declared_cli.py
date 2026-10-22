"""Manifest-declared plugin CLI commands: ``plugin.yaml`` ``cli_commands:`` rows read WITHOUT
importing any plugin, each materialising only its own plugin when argparse sets it up.

Re-exported by :mod:`hermes_cli.plugins`; the plugin machinery is read through that module at call
time, so a test that patches ``hermes_cli.plugins.<name>`` patches what these functions see."""

from __future__ import annotations

from typing import Any


def cli_command_key(name: str, parent: str | None = None) -> str:
    """``PluginManager._cli_commands`` key: the argv spelling, ``"<parent> <name>"`` or ``"<name>"``."""
    return f"{parent} {name}" if parent else name


def _materialize_declared_cli_command(manifest: Any, key: str, parser: Any) -> None:
    """``setup_fn`` of a manifest-declared command: load ONLY ``manifest``'s plugin, then run the
    parser setup its ``register(ctx)`` registered under ``key``. Never runs ``discover_plugins``."""
    from hermes_cli import plugins

    manager = plugins.get_plugin_manager()
    plugin_key = plugins.manifest_key(manifest)
    entry = manager._cli_commands.get(key)
    if entry is None or entry.get("plugin_key") != plugin_key:
        manager._load_plugin(manifest)
        entry = manager._cli_commands.get(key)
    if entry is None or entry.get("plugin_key") != plugin_key:
        raise RuntimeError(f"plugin {plugin_key!r} declares CLI command {key!r} in its manifest "
                           "but register(ctx) did not register it")
    entry["setup_fn"](parser)
    if entry.get("handler_fn") is not None:
        parser.set_defaults(func=entry["handler_fn"])


def discover_declared_cli_commands() -> list[dict[str, Any]]:
    """CLI commands declared in ``plugin.yaml`` ``cli_commands:``, found WITHOUT importing any plugin.

    One ``register_cli_command``-shaped descriptor per row, for every directory manifest discovery
    would load (same precedence and enable gate; ``HERMES_SAFE_MODE`` yields nothing). Each
    ``setup_fn`` materialises only its own plugin, so running a declared command costs one plugin
    import instead of ``discover_plugins``. Entry-point plugins have no manifest to read and
    keep the discovery path.
    """
    from hermes_cli import plugins

    if plugins._env_enabled("HERMES_SAFE_MODE"):
        return []
    winners = {plugins.manifest_key(m): m for m in plugins.collect_directory_manifests()}
    disabled, enabled = plugins._get_disabled_plugins(), plugins._get_enabled_plugins()
    commands: list[dict[str, Any]] = []
    for plugin_key, manifest in winners.items():
        if plugins.gate_manifest(manifest, disabled, enabled).action not in ("load", "load_now"):
            continue
        for row in manifest.cli_commands:
            key = cli_command_key(row["name"], row["parent"])
            commands.append({
                **row, "parent": row["parent"] or None, "help": row["help"] or manifest.description or "",
                "handler_fn": None, "plugin": manifest.name, "plugin_key": plugin_key,
                "setup_fn": lambda parser, _m=manifest, _k=key: _materialize_declared_cli_command(_m, _k, parser),
            })
    return commands

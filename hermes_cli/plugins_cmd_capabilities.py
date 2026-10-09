"""Capability consent (#64228): declared-vs-granted reads, the consent screen, ``hermes plugins
capabilities`` and the legacy ``allow_tool_override`` grant.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hermes_cli.plugin_capabilities import _child_dict


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _set_plugin_entry_flag(plugin_id: str, key: str, value: bool) -> None:
    """Write ``plugins.entries.<plugin_id>.<key> = value`` into config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    entry = _child_dict(_child_dict(_child_dict(config, "plugins"), "entries"), plugin_id)
    entry[key] = bool(value)
    save_config(config)


# ── Capability consent flow (#64228) ─────────────────────────────────────────
def _declared_capabilities_from_manifest(manifest: dict, plugin_name: str = "?") -> list:
    """Extract + normalize the ``capabilities:`` declaration from a manifest."""
    from hermes_cli.plugin_capabilities import parse_declared_capabilities
    return parse_declared_capabilities((manifest or {}).get("capabilities"), plugin_name)


def _declared_capabilities_for_key(key: str) -> list:
    """Read the declared capabilities for an installed/bundled plugin by key."""
    entry = _pc()._find_plugin_entry(key)
    if entry is None:
        return []
    if entry[3] == "entrypoint":
        from hermes_cli.plugins import discover_entrypoint_manifests
        for manifest in discover_entrypoint_manifests():
            if key in (manifest.key, manifest.name):
                return list(manifest.capabilities)
        return []
    if not entry[4]:
        return []
    return _declared_capabilities_from_manifest(_pc()._read_manifest(Path(entry[4])), entry[0])


def _run_capability_consent(console, plugin_id: str, declared: list, *, context: str = "install") -> bool:
    """Show the capability consent screen and record the decision; True when granted.

    On consent the pending capabilities are granted under
    ``plugins.entries.<id>.granted_capabilities`` with a hash of the declared set. On decline —
    or in ANY non-interactive context — they stay ungranted (fail closed) and the plugin must
    degrade via ``ctx.has_capability()``. Consent + audit, NOT a sandbox.
    """
    from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY, pending_capabilities, record_consent
    pending = pending_capabilities(plugin_id, declared)
    if not pending:
        # Refresh the consent hash so a later declaration change is detected.
        if declared:
            record_consent(plugin_id, [], declared)
        return True

    verb = "requests" if context == "install" else "now requests"
    console.print(f"\n  [yellow]Plugin [bold]{plugin_id}[/bold] {verb} the following capabilities:[/yellow]")
    for cap in pending:
        spec = CAPABILITY_REGISTRY.get(cap)
        console.print(f"    [bold]{cap}[/bold] — {spec.description if spec else ''}")
    console.print(
        "  [dim]Granting trusts the plugin author with these host surfaces. "
        "This is consent, not a sandbox — plugins run as regular Python "
        "in-process.[/dim]")

    if not _pc()._is_tty():
        console.print(
            "  [yellow]Non-interactive session: capabilities NOT granted "
            "(fail closed).[/yellow] Run "
            f"`hermes plugins capabilities {plugin_id}` to review and "
            f"`hermes plugins enable {plugin_id}` to grant interactively.")
        return False

    if _pc()._ask_yes("  Grant these capabilities? [y/N] ", console.input):
        record_consent(plugin_id, pending, declared)
        console.print(
            f"  [green]✓[/green] Granted: {', '.join(pending)} "
            f"([dim]plugins.entries.{plugin_id}.granted_capabilities[/dim])")
        return True

    console.print(
        f"  [dim]Declined. {plugin_id} stays enabled with these capabilities "
        "off; it should degrade gracefully (ctx.has_capability()). Re-run "
        f"`hermes plugins enable {plugin_id}` to grant later.[/dim]")
    return False


def cmd_capabilities(name: Optional[str] = None) -> None:
    """``hermes plugins capabilities [<id>]`` — declared vs granted."""
    from hermes_cli.plugin_capabilities import (
        CAPABILITY_REGISTRY,
        granted_capabilities,
        plugin_capability_granted,
    )
    console = _pc()._console()
    rows = []
    for entry in _pc()._discover_all_plugins():
        key = entry[5] or entry[0]
        if name is not None and name not in (key, entry[0]):
            continue
        declared = _declared_capabilities_for_key(key)
        granted = granted_capabilities(key)
        # Effective state includes grants live via deprecated allow_* keys.
        effective = {cap for cap in CAPABILITY_REGISTRY if plugin_capability_granted(key, cap)}
        if not declared and not effective and name is None:
            continue
        rows.append((key, entry[3], declared, granted, effective))

    if name is not None and not rows:
        _pc()._fail(console, _pc()._unknown_plugin_message(name))
    if not rows:
        console.print("[dim]No plugins declare or hold capabilities.[/dim]")
        return

    for key, source, declared, granted, effective in sorted(rows):
        console.print(f"[bold]{key}[/bold] [dim]({source})[/dim]")
        if not declared:
            console.print("  declared: [dim](none)[/dim]")
        for cap in declared:
            if cap not in effective:
                mark = "[yellow]not granted[/yellow]"
            elif cap in granted:
                mark = "[green]granted[/green]"
            else:
                mark = "[green]granted[/green] [dim](via legacy allow_* key — deprecated)[/dim]"
            console.print(f"  {cap}: {mark}")
        for cap in sorted(effective - set(declared)):
            console.print(f"  {cap}: [green]granted[/green] [dim](not declared in manifest)[/dim]")


def _resolve_tool_override_grant(console, key: str, allow_tool_override: Optional[bool]) -> None:
    """Resolve and persist the ``allow_tool_override`` grant for a plugin."""
    if allow_tool_override is None:
        # Default NO: a blind Enter or a non-interactive stdin denies safely.
        allow_tool_override = _pc()._ask_yes(
            "[yellow]Allow this plugin to replace built-in tools "
            "(e.g. shell_exec, write_file)?[/yellow]\n"
            "  This is a privileged capability: an override can intercept "
            "everything the agent routes through that tool.\n"
            "  Grant it? [y/N] ",
            console.input,
        )
    _set_plugin_entry_flag(key, "allow_tool_override", allow_tool_override)
    if allow_tool_override:
        console.print(
            f"[green]✓[/green] Granted [bold]{key}[/bold] permission to "
            "override built-in tools "
            f"([dim]plugins.entries.{key}.allow_tool_override: true[/dim]).")
    else:
        console.print(
            f"[dim]{key} may not override built-in tools. Re-run "
            f"`hermes plugins enable {key} --allow-tool-override` to grant "
            "this later.[/dim]")

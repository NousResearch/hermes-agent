"""External-editor command resolution shared by interactive surfaces."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def resolve_editor_command(
    *, config: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Resolve an editor without invoking a shell.

    Process environment remains the conventional override. The per-profile
    setting is the durable choice for a Desktop-launched backend with neither
    ``VISUAL`` nor ``EDITOR`` set.
    """
    env = os.environ if environ is None else environ
    if command := (env.get("VISUAL") or env.get("EDITOR")):
        return command
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    editor = config.get("editor") if isinstance(config, Mapping) else None
    if isinstance(editor, Mapping) and isinstance(command := editor.get("command"), str) and command.strip():
        return command.strip()
    return "notepad" if os.name == "nt" else "nano"

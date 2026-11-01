"""Public, read-only TUI feature discovery for terminal launchers and plugins.

``hermes --tui-capabilities`` is introspection only: no gateway, model,
provider authentication, user plugin registration or session creation.
"""
from __future__ import annotations

import json


def tui_capabilities() -> dict:
    """Version the contract independently of Hermes's release number."""
    return {
        "schema": 1,
        "ui": "hermes-tui",
        "features": {
            "session_tabs": 1,
            "tab_commands": 1,
            "tab_persistence": 1,
            "background_activity": 1,
            "slash_completion": 1,
        },
    }


def capabilities_json() -> str:
    return json.dumps(tui_capabilities(), sort_keys=True)


def print_requested_capabilities(argv: list[str]) -> bool:
    """The standalone probe bypasses normal startup and recovery side effects."""
    if argv != ["--tui-capabilities"]:
        return False
    print(capabilities_json())
    return True

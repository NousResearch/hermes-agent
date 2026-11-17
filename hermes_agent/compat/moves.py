"""Old module name -> current module name.

A key is added when a module moves and is never removed; when the target moves again, only its
value changes. Plain dicts and no imports, because a process from an older release reads this file.
"""

MOVES: dict[str, str] = {
    "tools.mcp_elicitation_form": "tools.mcp.elicitation_form",
    "tools.mcp_oauth": "tools.mcp.oauth",
    "tools.mcp_tool_discovery": "tools.mcp.discovery",
    "tools.plugin_guard": "hermes_cli.plugins_install_scan",
}

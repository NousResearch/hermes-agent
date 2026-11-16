"""Old module name -> current module name.

A key is added when a module moves and is never removed; when the target moves again, only its
value changes. Plain dicts and no imports, because a process from an older release reads this file.
"""

MOVES: dict[str, str] = {
    "tools.browser_supervisor": "browser.supervisor",
    "tools.browser_supervisor_capture": "browser.supervisor_capture",
    "tools.plugin_guard": "hermes_cli.plugins_install_scan",
}

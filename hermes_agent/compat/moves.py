"""Old module name -> current module name.

A key is added when a module moves and is never removed; when the target moves again, only its
value changes. Plain dicts and no imports, because a process from an older release reads this file.

A module folded into one that already existed is a key too, only when a catalog-pinned plugin imports
it and the target defines every name the old module published, unchanged. Path guards judge the
target by the first key that names it, so the oldest one is listed first.
"""

MOVES: dict[str, str] = {
    "tools.plugin_guard": "hermes_cli.plugins_install_scan",
    # Folded into session_rules.turns by the storage interface (#149567); the OpenViking and hermes-council pins import it.
    "agent.message_content": "session_rules.turns",
    # Folded into sqlite_support by the storage interface (#151855); the holographic 0.29.x pin imports it.
    "hermes_state_wal": "sqlite_support",
}

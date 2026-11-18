"""Old module name -> current module name.

A key is added when a module moves and is never removed; when the target moves again, only its
value changes. Plain dicts and no imports, because a process from an older release reads this file.

A module folded into one that already existed is a key too, when the target defines every name the
old module published, unchanged: outside plugins keep importing the old name. Path guards judge the
target by the first key that names it, so the oldest one is listed first.
"""

MOVES: dict[str, str] = {
    "tools.plugin_guard": "hermes_cli.plugins_install_scan",
    # Folded into session_rules.turns by the storage interface (#149567); catalog plugins import the first.
    "agent.message_content": "session_rules.turns",
    "agent.context_compressor_retry": "session_rules.turns",
}

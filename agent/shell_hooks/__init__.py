"""Shell-script hooks: ``hooks:`` config -> first-use consent per ``(event, command)`` -> callbacks on
the plugin hook manager, so every ``invoke_hook()`` site dispatches to the scripts. Stdin carries a
JSON payload ``{hook_event_name, tool_name, tool_input, session_id, cwd, profile, extra}``; stdout may
return ``{"decision"|"action": "block"|"modify", ...}`` / ``{"context": ...}``. Exit code 2 blocks a
``pre_tool_call`` even without JSON (Claude Code / Cursor). Fail open unless ``fail_closed``.

The names below are the cross-package surface. Modules:

- ``spec``: ``ShellHookSpec`` and the ``hooks:`` parser.
- ``wire``: the stdin payload and the per-event stdout directive parsers.
- ``spawn``: the single subprocess site.
- ``runner``: the per-firing callback and its failure semantics.
- ``consent``: the allowlist, the first-use prompt, the opt-in channels, script facts for the CLI.
- ``registry``: registration on the hook manager, the gateway resync, launch/process opt-ins.
- ``rewake``: ``async_rewake`` hooks, run in the background, that wake their session on exit 2.
"""

from agent.shell_hooks.consent import (
    ALLOWLIST_FILENAME,
    accept_for_process,
    allowlist_entry_for,
    allowlist_path,
    allowlisted_pairs,
    load_allowlist,
    record_approval,
    revoke,
    save_allowlist,
    script_is_executable,
    script_mtime_iso,
)
from agent.shell_hooks.registry import (
    accept_for_launch,
    re_register_config_hooks,
    record_consent_from_config,
    register_from_config,
    reset_for_tests,
    sync_from_config,
)
from agent.shell_hooks.rewake import cancel_session, stop_all
from agent.shell_hooks.runner import run_once
from agent.shell_hooks.spec import DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS, ShellHookSpec, iter_configured_hooks

__all__ = [
    "ALLOWLIST_FILENAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "ShellHookSpec",
    "accept_for_launch",
    "accept_for_process",
    "allowlist_entry_for",
    "allowlist_path",
    "allowlisted_pairs",
    "cancel_session",
    "iter_configured_hooks",
    "load_allowlist",
    "re_register_config_hooks",
    "record_approval",
    "record_consent_from_config",
    "register_from_config",
    "reset_for_tests",
    "revoke",
    "run_once",
    "save_allowlist",
    "script_is_executable",
    "script_mtime_iso",
    "stop_all",
    "sync_from_config",
]

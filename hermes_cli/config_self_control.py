"""Config keys that decide whether a security gate runs are not settable by the process they guard.

``hermes config set`` is deliberately available to an agent (the protected-instruction write guard points
at it instead of letting a tool edit ``config.yaml``), which is right for ordinary knobs and wrong for a
gate's own switch: a dispatched worker whose ``HERMES_HOME`` is its own profile could run
``hermes -p <lane> config set security.protected_instruction_files false`` and the next process it spawned
would read the gate as off — the exact approval the gate exists to require, granted by the requesting
party (measured on a kanban run of t_ab071361, 2026-11-13, after its first write was correctly BLOCKED).

A person at a terminal keeps the knob: ``security.protected_instruction_files`` is a documented per-profile
setting and turning it off is a deliberate act. A session that declares itself unattended (``-q``, cron, a
gateway turn, a programmatic platform), or a plain agent-spawned subprocess — which never has a TTY on
stdin — is refused and told to use a terminal, or to edit ``config.yaml`` directly.

Reading the switch from the install-level config instead would not hold: ``HERMES_HOME=<install> hermes
config set ...`` reaches that file from an agent's own shell, so the check belongs on the writing path.
"""

from __future__ import annotations

import sys
from typing import Any

# The protected-instruction gate's control surface: ``..._files`` decides whether the gate runs at all,
# ``..._extra_patterns`` decides what it covers.
_PROTECTED_SWITCH_LEAVES = ("protected_instruction_files", "protected_instruction_extra_patterns")


def _touches_protected_switch(key: str, value: Any) -> bool:
    """True when *key* names a protected switch, or writes a section that carries one.

    ``config set security '{"protected_instruction_files": false}'`` replaces the whole section, so the
    leaf alone does not recognise every weakening write."""
    parts = [part for part in str(key).split(".") if part]
    if not parts:
        return False
    if parts[-1] in _PROTECTED_SWITCH_LEAVES:
        return True
    return "security" in parts and any(leaf in str(value) for leaf in _PROTECTED_SWITCH_LEAVES)


def _human_at_a_terminal() -> bool:
    """True when a person is at this process's terminal.

    Unattended sessions say so in the environment (``-q``, cron, gateway, programmatic platforms) and
    anything an agent spawns inherits that — plus an agent's subprocess never has a TTY on stdin."""
    try:
        from tools.approval_context import _is_gateway_approval_context, _no_user_can_answer

        if _no_user_can_answer() or _is_gateway_approval_context():
            return False
    except ImportError:  # no approval subsystem to ask: judge by the terminal alone
        pass
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, OSError, ValueError):  # a closed or replaced stdin is not a human
        return False


def refuse_unattended_security_self_control(key: str, value: Any) -> None:
    """Exit 1 when an unattended process tries to move the protected-instruction gate's own switch.

    Called by :func:`hermes_cli.config.set_config_value` before anything is written."""
    if not _touches_protected_switch(key, value) or _human_at_a_terminal():
        return
    print(
        f"✗ Refusing to set '{key}' from a session with no terminal a person is at.\n"
        "  It decides whether the protected-agent-instruction approval gate runs at all, and a process\n"
        "  that gate guards must not be the one granting the change. Nothing was written.\n"
        "  Run 'hermes config set' in your own terminal, or edit config.yaml directly.",
        file=sys.stderr)
    sys.exit(1)

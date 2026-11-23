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

Two things this module gets right, both measured on the PR that introduced it:

* **The decision is made on the parsed structure, never on the raw argument.** ``set_config_value``
  coerces the argument (``_coerce_config_set_value`` → ``yaml.safe_load``) as well, so a YAML/JSON escape
  decoded to the real leaf while the guard was substring-matching text: ``config set security
  '{"protected_instruction_\\u0066iles": false}'`` (also ``\\x66``, ``\\u0070…``) returned 0 and left the
  gate off. The same parse the writer uses runs here first, so escapes, anchors and merge keys are
  already resolved when the leaves are looked for.
* **Only writes that leave the gate weaker are refused.** Refusing the enabling direction too broke the
  repair path: the host watchdog (``~/.hermes/scripts/protected-instruction-guard.py``) restores a flipped
  home with ``hermes config set security.protected_instruction_files true`` from a cron tick, and that
  call was refused, leaving the only automated mitigation failing open. A write that leaves the gate ON —
  the repair itself — is allowed; nobody unattended can still turn it off.

Reading the switch from the install-level config instead would not hold: ``HERMES_HOME=<install> hermes
config set ...`` reaches that file from an agent's own shell, so the check belongs on the writing path.
"""

from __future__ import annotations

import copy
import sys
from typing import Any

_SECURITY_SECTION = "security"
# The protected-instruction gate's control surface: ``..._files`` decides whether the gate runs at all,
# ``..._extra_patterns`` decides what it covers.
_PROTECTED_SWITCH_LEAVES = ("protected_instruction_files", "protected_instruction_extra_patterns")
_ENABLE_LEAF = "protected_instruction_files"
_COVERAGE_LEAF = "protected_instruction_extra_patterns"


def _parsed_value(key: str, value: Any) -> Any:
    """The structure ``set_config_value`` is going to store for *key* = *value*.

    Same coercion the writer applies (``_coerce_config_set_value``: scalar words, then ``yaml.safe_load``
    for a list/mapping literal), so a YAML/JSON escape is decoded before anything is matched. A value
    that is not a string was already coerced by the caller and is returned as-is.
    """
    if not isinstance(value, str):
        return value
    try:
        from hermes_cli.config import _coerce_config_set_value
    except Exception:  # not importable (module used on its own): decode the literal ourselves
        _coerce_config_set_value = None
    if _coerce_config_set_value is not None:
        return _coerce_config_set_value(key, value)
    try:  # pragma: no cover - fallback path
        import yaml

        parsed = yaml.safe_load(value)
    except Exception:  # pragma: no cover - a value the writer would store as a plain string
        return value
    return parsed if isinstance(parsed, (dict, list)) else value  # pragma: no cover


def _names_switch_leaf(structure: Any) -> bool:
    """True when a parsed structure carries a protected leaf at any depth.

    Walks mappings and sequences instead of matching text: by this point escapes, anchors and merge keys
    are resolved, so the question asked is what the gate will actually read.
    """
    if isinstance(structure, dict):
        for name, item in structure.items():
            if str(name) in _PROTECTED_SWITCH_LEAVES or _names_switch_leaf(item):
                return True
        return False
    if isinstance(structure, (list, tuple)):
        return any(_names_switch_leaf(item) for item in structure)
    return False


def _key_may_name_switch(key: str) -> bool:
    """Cheap pre-check on the dotted key alone, so an unrelated write is never parsed here."""
    parts = [part for part in str(key).split(".") if part]
    return bool(parts) and (parts[-1] in _PROTECTED_SWITCH_LEAVES or _SECURITY_SECTION in parts)


def _touches_protected_switch(key: str, value: Any) -> bool:
    """True when *key* names a protected switch, or writes a section that carries one.

    ``config set security '{"protected_instruction_files": false}'`` replaces the whole section, so the
    leaf alone does not recognise every weakening write. *value* is the parsed structure (see
    :func:`_parsed_value`).
    """
    parts = [part for part in str(key).split(".") if part]
    if not parts:
        return False
    if parts[-1] in _PROTECTED_SWITCH_LEAVES:
        return True
    if _SECURITY_SECTION not in parts:
        return False
    return _names_switch_leaf(value)


def _read_switch(config: Any) -> tuple[Any, list[Any]]:
    """The ``(files, extra_patterns)`` the gate reads from *config* (a raw mapping, or None)."""
    section = config.get(_SECURITY_SECTION) if isinstance(config, dict) else None
    if not isinstance(section, dict):
        return True, []  # absent (or unreadable): the gate's documented default is ON
    extra = section.get(_COVERAGE_LEAF, [])
    return section.get(_ENABLE_LEAF, True), list(extra) if isinstance(extra, (list, tuple)) else []


def _config_after_write(key: str, value: Any, current: Any) -> dict[str, Any] | None:
    """*current* with the write applied exactly as ``set_config_value`` applies it, or None."""
    base = copy.deepcopy(current) if isinstance(current, dict) else {}
    try:
        from hermes_cli.config import _set_nested

        _set_nested(base, str(key), value)
    except Exception:  # a shape the writer itself will refuse: fail closed
        return None
    return base


def _leaves_gate_enabled(key: str, value: Any, current: Any) -> bool:
    """True when the write still leaves the gate ON, covering what it covered before.

    That is the repair path (``..._files true``) plus coverage writes that change nothing. A write that
    drops the switch to a falsy value, or replaces the coverage patterns, is not allowed unattended.
    """
    after = _config_after_write(key, value, current)
    if after is None:
        return False
    files, extra = _read_switch(after)
    if not files:
        return False
    _before_files, before_extra = _read_switch(current)
    return extra == before_extra


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


def refuse_unattended_security_self_control(key: str, value: Any, current_config: Any = None) -> None:
    """Exit 1 when an unattended process would leave the protected-instruction gate weaker.

    Refused, and only where nobody is at a terminal:

    * a write that leaves ``security.protected_instruction_files`` OFF — the self-grant the gate exists
      to stop, whatever spelling of the leaf the payload used;
    * a write that changes ``security.protected_instruction_extra_patterns`` — narrowing what the gate
      covers is the same self-grant by another name, and no repair path needs to widen it.

    Everything else passes, including ``..._files true``: the host watchdog repairs a flipped home with
    exactly that write from a cron tick. *current_config* is the raw user config ``set_config_value`` is
    about to write into, so the guard can tell a repair from a flip.

    Called by :func:`hermes_cli.config.set_config_value` before anything is written.
    """
    if _human_at_a_terminal() or not _key_may_name_switch(key):
        return
    parsed = _parsed_value(key, value)
    if not _touches_protected_switch(key, parsed):
        return
    if _leaves_gate_enabled(key, parsed, current_config):
        return
    print(
        f"✗ Refusing to set '{key}' from a session with no terminal a person is at.\n"
        "  It decides whether the protected-agent-instruction approval gate runs at all, or what it\n"
        "  covers, and a process that gate guards must not be the one granting the change. Nothing was\n"
        "  written. A write that leaves the gate ON (the repair direction) is still allowed.\n"
        "  Run 'hermes config set' in your own terminal, or edit config.yaml directly.",
        file=sys.stderr)
    sys.exit(1)

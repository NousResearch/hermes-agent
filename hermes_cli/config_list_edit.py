"""``hermes config add`` / ``hermes config remove``: add or remove one item of a list setting.

``config set`` replaces a whole value, so growing ``toolsets`` or ``plugins.enabled`` by one entry
meant retyping the list (and losing the defaults nobody remembers) or editing ``config.yaml`` by
hand. These commands edit one item in place:

- the value is a string unless ``--json`` parses it (``'{"command": "check.py"}'``, ``5``);
- adding an item the list already holds, or removing one it lacks, changes nothing;
- a key absent from ``config.yaml`` starts from its default list, so ``add toolsets web`` keeps
  ``hermes-cli``;
- a key whose default or saved value is not a list, and an administrator-managed key, are refused;
- the read-modify-write holds the config lock (threads and processes) and goes through the
  comment-preserving round-trip writer ``config set`` uses.
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any


def _same(item: Any, value: Any) -> bool:
    # ``True == 1`` in Python; a bool and a number are different YAML items, at any depth
    # (readers check flags like ``models_discovered`` with ``is True``).
    if isinstance(item, bool) is not isinstance(value, bool):
        return False
    if isinstance(item, dict) and isinstance(value, dict):
        return len(item) == len(value) and all(k in value and _same(v, value[k]) for k, v in item.items())
    if isinstance(item, list) and isinstance(value, list):
        return len(item) == len(value) and all(map(_same, item, value))
    return item == value


def _parse_value(key: str, value: str, as_json: bool) -> Any:
    if not as_json:
        return value
    from hermes_cli.config import _exit_invalid

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        _exit_invalid(f"✗ --json value for '{key}' is not valid JSON ({exc.msg} at column {exc.colno}).")


def _refuse_unlistable_key(key: str, verb: str) -> None:
    """Keys that are not list settings in ``config.yaml``: malformed paths, .env settings and
    administrator-managed keys."""
    from hermes_cli.config import _exit_if_key_managed, _exit_invalid, _is_env_config_key, _split_key_path
    from hermes_cli.config_env_routing import is_env_setting_key

    if key != key.strip() or not key.strip() or "" in _split_key_path(key):
        _exit_invalid(f"✗ Invalid config key: {key!r} (empty path segment or surrounding whitespace).")
    _exit_if_key_managed(key, verb)
    if _is_env_config_key(key) or is_env_setting_key(key):
        _exit_invalid(f"✗ '{key}' is an environment setting in .env, not a list in config.yaml.")


def _current_list(user_config: dict, key: str, default: Any) -> list:
    """The saved list at *key*, else a copy of its default list (``[]`` when it has none)."""
    from hermes_cli.config import _MISSING, _SCALAR_AS_ONE_ITEM_LIST_KEYS, _exit_invalid, _get_nested

    current = _get_nested(user_config, key)
    if current is _MISSING or current is None:
        return copy.deepcopy(default) if isinstance(default, list) else []
    if isinstance(current, str) and key in _SCALAR_AS_ONE_ITEM_LIST_KEYS:
        return [current]  # its reader already treats a bare name as a one-item list
    if not isinstance(current, list):
        _exit_invalid(
            f"✗ '{key}' holds {type(current).__name__} {current!r} in config.yaml, not a list. "
            f"Use `hermes config set {key} ...` to replace it.")
    return list(current)


def _refuse_path_through_missing_list(user_config: dict, key: str) -> None:
    """``custom_providers.0.models`` must address an item that exists. An absent or ``null`` list
    would become the mapping ``custom_providers: {'0': ...}``, which every reader of that list
    ignores, and an index past the end of a saved list cannot be written at all."""
    from hermes_cli.config import _MISSING, _exit_invalid, _expected_container_type, _get_nested, _split_key_path

    parts = _split_key_path(key)
    for depth in range(1, len(parts)):
        prefix = ".".join(parts[:depth])
        if _expected_container_type(prefix, user_config) != "list":
            continue
        has_item = (isinstance(_get_nested(user_config, prefix), list)
                    and _get_nested(user_config, ".".join(parts[:depth + 1])) is not _MISSING)
        if not has_item:
            _exit_invalid(f"✗ '{prefix}' is a list with no item {parts[depth]!r} yet; nothing was written.")


def _refuse_non_list_schema(key: str) -> Any:
    """*key*'s default list (or ``None``); exits when the schema fixes it to anything but a list."""
    from hermes_cli.config import _KNOWN_CONTAINER_TYPES, _MISSING, DEFAULT_CONFIG, _exit_invalid, _get_nested

    default = _get_nested(DEFAULT_CONFIG, key)
    if default is _MISSING:
        default = None
    if _KNOWN_CONTAINER_TYPES.get(key) == "mapping" or isinstance(default, dict):
        shape = "mapping"
    else:
        shape = None if default is None or isinstance(default, list) else type(default).__name__
    if shape:
        _exit_invalid(
            f"✗ '{key}' is not a list setting (it holds a {shape}). "
            f"Use `hermes config set {key} ...` instead.")
    return default


def edit_config_list(key: str, raw_value: str, *, remove: bool = False, as_json: bool = False) -> None:
    """Add *raw_value* to (or remove it from) the list setting at dotted *key*."""
    from hermes_cli import config as config_mod
    from hermes_cli.plugins_state import _locked_plugin_state

    verb = "remove from" if remove else "add to"
    if config_mod.is_managed():
        config_mod.managed_error(f"{verb} configuration lists")
        return
    _refuse_unlistable_key(key, verb)
    value = _parse_value(key, raw_value, as_json)
    default = _refuse_non_list_schema(key)
    is_known, suggestion = config_mod._validate_config_key(key)
    # List slots the schema table knows without a seeded default are real settings.
    is_known = is_known or key in config_mod._KNOWN_CONTAINER_TYPES or key in config_mod._SCALAR_AS_ONE_ITEM_LIST_KEYS
    if not is_known and config_mod._is_wrong_prefix_suggestion(key, suggestion):
        from hermes_cli.config_key_notices import unknown_subkey_refusal
        config_mod._exit_invalid(unknown_subkey_refusal(key, suggestion, force_hint=False))

    config_path = config_mod.get_config_path()
    # Saved list items can carry credentials (``custom_providers[].api_key``); print masked copies.
    shown = json.dumps(config_mod.redact_config_value(value), ensure_ascii=False) if as_json else repr(value)
    with _locked_plugin_state(config_path), config_mod._CONFIG_LOCK:
        user_config = config_mod.require_readable_config_before_write(config_path)
        _refuse_path_through_missing_list(user_config, key)
        items = _current_list(user_config, key, default)
        kept = [item for item in items if not _same(item, value)]
        if remove and len(kept) == len(items):
            print(f"· {shown} is not in {key}; nothing changed.")
            return
        if not remove and len(kept) < len(items):
            print(f"· {key} already contains {shown}; nothing changed.")
            return
        items = kept if remove else [*items, value]
        try:
            config_mod._set_nested(user_config, key, items)
        except ValueError as exc:
            config_mod._exit_invalid(f"✗ {exc}")
        config_mod._write_user_config(config_path, user_config)

    # terminal.docker_volumes & co. are mirrored into .env for the terminal tool, like `config set`.
    env_var = config_mod.terminal_config_env_var_for_key(key)
    if env_var:
        config_mod.save_env_value(env_var, config_mod._terminal_env_value(items))
    print(f"✓ {'Removed' if remove else 'Added'} {shown} {'from' if remove else 'to'} {key} in {config_path}")
    print(f"  {key} = {json.dumps(config_mod.redact_config_value(items), ensure_ascii=False)}")
    if not is_known:
        from hermes_cli.config_key_notices import print_unknown_key_notice
        print_unknown_key_notice(key, suggestion, force_hint=False)


def cmd_config_list_edit(args, *, remove: bool) -> None:
    key, value = getattr(args, "key", None), getattr(args, "value", None)
    if not key or value is None:
        name = "remove" if remove else "add"
        print(f"Usage: hermes config {name} [--json] <list-key> <value>", file=sys.stderr)
        print(f"  e.g. hermes config {name} toolsets web", file=sys.stderr)
        print(f"       hermes config {name} --json hooks.pre_llm_call '{{\"command\": \"~/check.py\"}}'",
              file=sys.stderr)
        sys.exit(1)
    from hermes_cli.config import _run_write_command

    _run_write_command(lambda: edit_config_list(key, value, remove=remove, as_json=bool(getattr(args, "json", False))))

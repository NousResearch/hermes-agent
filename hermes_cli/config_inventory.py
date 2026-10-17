"""``hermes config keys``: the registered config-path inventory, and resolved values on request.

Shell completion (``hermes completion``) reads the same list, so a path that completes is a
path ``config get/set/unset`` addresses.
"""

import json


def _escape(segment) -> str:
    # ``_split_key_path`` splits on unescaped dots, so a literal dotted name (``service.name``,
    # model ids like ``grok-4.6``) must be escaped or the printed path addresses a phantom nest
    # that ``config set`` would create beside the real key.
    return str(segment).replace(".", "\\.")


def _flatten(mapping, prefix="", *, leaves_only=False):
    for name, value in mapping.items():
        if str(name).startswith("_"):
            continue
        key = f"{prefix}.{_escape(name)}" if prefix else _escape(name)
        nested = isinstance(value, dict) and bool(value)
        if not (leaves_only and nested):
            yield key, value
        if nested:
            yield from _flatten(value, key, leaves_only=leaves_only)


def registered_config_keys() -> list[str]:
    """Registered roots plus every path declared in ``DEFAULT_CONFIG``; reads no user config."""
    from hermes_cli.config import DEFAULT_CONFIG, _known_top_level_keys

    keys = {key for key in _known_top_level_keys() if not key.startswith("_")}
    keys.update(key for key, _ in _flatten(DEFAULT_CONFIG))
    return sorted(keys)


def boolean_config_keys() -> list[str]:
    """Paths whose default is a bool (shell completion offers ``true``/``false`` for them)."""
    from hermes_cli.config import DEFAULT_CONFIG

    return sorted(key for key, value in _flatten(DEFAULT_CONFIG) if isinstance(value, bool))


def resolved_config_values() -> dict:
    """Every leaf of the resolved config (defaults + config.yaml + managed overlay), with
    credential-shaped values masked exactly like ``config get`` unless redaction is off."""
    from agent.redact import _redact_enabled
    from hermes_cli.config import load_config, redact_config_value

    config = load_config()
    if _redact_enabled():
        config = redact_config_value(config)
    return dict(sorted(_flatten(config, leaves_only=True)))


def _scalar(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def config_keys_command(args) -> None:
    as_json = getattr(args, "json", False)
    if getattr(args, "values", False):
        values = resolved_config_values()
        if as_json:
            print(json.dumps(values, ensure_ascii=False, default=str))
        else:
            for key, value in values.items():
                print(f"{key}={_scalar(value)}")
        return
    ordered = registered_config_keys()
    if as_json:
        print(json.dumps(ordered))
    else:
        for key in ordered:
            print(key)

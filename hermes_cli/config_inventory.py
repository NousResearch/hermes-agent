"""Read-only discovery of registered configuration paths."""

import json


def config_keys_command(args) -> None:
    from hermes_cli.config import DEFAULT_CONFIG, _known_top_level_keys

    keys = {key for key in _known_top_level_keys() if not key.startswith("_")}

    def visit(mapping, prefix=""):
        for name, value in mapping.items():
            if name.startswith("_"):
                continue
            key = f"{prefix}.{name}" if prefix else name
            keys.add(key)
            if isinstance(value, dict):
                visit(value, key)

    visit(DEFAULT_CONFIG)
    ordered = sorted(keys)
    if getattr(args, "json", False):
        print(json.dumps(ordered))
    else:
        for key in ordered:
            print(key)

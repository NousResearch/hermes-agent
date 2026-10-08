"""Container-shape diagnostics for config.yaml."""

from typing import Any, Dict


def container_slots(default_config: Dict[str, Any], known_container_types: Dict[str, str]) -> Dict[str, str]:
    """Dotted key -> ``"list"``/``"mapping"`` for every slot the schema fixes to a container:
    ``DEFAULT_CONFIG`` (sections included) plus the known-container table for roots it omits."""
    slots: Dict[str, str] = {}

    def walk(node: Dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                slots[path] = "mapping"
                walk(value, path)
            elif isinstance(value, list):
                slots[path] = "list"

    walk(default_config, "")
    slots.update(known_container_types)
    return slots


def validate_quoted_containers(config, issues, slots, scalar_as_one_item_list_keys,
                               cfg_get, looks_structured_value, issue, yaml, shlex) -> None:
    """A container slot holding ONE quoted string (``enabled: '["a","b"]'``) is skipped by every
    isinstance-gated reader while ``config get`` echoes it back, so plugins silently unmount and
    exclusions silently lapse (#83308, #105706). Finding only — the file is never rewritten."""
    for key, kind in slots.items():
        # ``parse_config_string_list`` readers accept the quoted form; nothing is ignored there.
        if key in scalar_as_one_item_list_keys:
            continue
        value = cfg_get(config, *key.split("."))
        if not isinstance(value, str) or not looks_structured_value(value):
            continue
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            continue
        if isinstance(parsed, (list, dict)):
            issue(issues, "warning",
                  f"{key} is the quoted string {value!r} — Hermes expects a YAML {kind} here "
                  "and every reader ignores the string",
                  f"Run: hermes config set {key} {shlex.quote(value)}  (stores a real {kind}), "
                  "or remove the quotes in config.yaml")

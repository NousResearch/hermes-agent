"""Shape checks for container slots and dotted root keys, run by
``hermes_cli.config.validate_config_structure`` (startup warnings, doctor, ``hermes config check``).

A value of the wrong shape in a list slot, or a nested path written as one literal root key, is
skipped by every isinstance-gated reader while ``config get`` echoes it back. Findings only: the
file is never rewritten.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

import hermes_yaml as yaml
from hermes_cli.config_issues import ConfigIssue, _issue

# List slots whose own reader also accepts one plain value (``shlex``-split, wrapped, or parsed),
# on top of the ``parse_config_string_list`` ones: a scalar there is a supported spelling.
_SCALAR_TOLERANT_LIST_KEYS = frozenset({
    "browser.chromium_args", "command_allowlist", "dashboard.trusted_proxies", "desktop.electron_flags",
    "plugins.host.launcher", "security.fake_ip_ranges", "voice.stop_phrases"})


def _container_slots() -> dict[str, str]:
    """Dotted key -> ``"list"``/``"mapping"`` for every slot the schema fixes to a container:
    ``DEFAULT_CONFIG`` (sections included) plus the known-container table for roots it omits."""
    from hermes_cli.config import _KNOWN_CONTAINER_TYPES, DEFAULT_CONFIG

    slots: dict[str, str] = {}

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                slots[path] = "mapping"
                walk(value, path)
            elif isinstance(value, list):
                slots[path] = "list"

    walk(DEFAULT_CONFIG, "")
    slots.update(_KNOWN_CONTAINER_TYPES)
    return slots


def _quoted_container_issue(key: str, kind: str, value: str, issues: list[ConfigIssue]) -> bool:
    """``enabled: '["a","b"]'``: one quoted string holding a list/mapping literal (#83308, #105706)."""
    from hermes_cli.config import _looks_structured_value

    if not _looks_structured_value(value):
        return False
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return False
    if not isinstance(parsed, (list, dict)):
        return False
    _issue(issues, "warning",
           f"{key} is the quoted string {value!r} — Hermes expects a YAML {kind} here "
           "and every reader ignores the string",
           f"Run: hermes config set {key} {shlex.quote(value)}  (stores a real {kind}), "
           "or remove the quotes in config.yaml")
    return True


def _scalar_in_list_issue(key: str, value: Any, issues: list[ConfigIssue]) -> None:
    """``enabled: omh`` / ``enabled: {omh: true}``: readers skip it, so ``plugins.enabled`` loads no
    user plugin at all and an exclusion list excludes nothing."""
    shape = "a mapping" if isinstance(value, dict) else f"the plain value {value!r}"
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        # JSON, not ``[value]``: brackets alone split ``a,b`` in two and turn ``on`` into a bool.
        item = json.dumps(value, ensure_ascii=False)
        hint = (f"Run: hermes config set {key} {shlex.quote(f'[{item}]')}  (stores a one-item list), "
                f"or write it in config.yaml as a list item ('- {item}')")
    else:
        hint = f"Write {key} in config.yaml as a YAML list (one '- item' per line)"
    _issue(issues, "warning",
           f"{key} is {shape} — Hermes expects a YAML list here and ignores this value", hint)


def validate_container_shapes(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """Flag a container slot holding the wrong shape. ``None`` is \"unset\" everywhere and stays silent."""
    from hermes_cli.config import _SCALAR_AS_ONE_ITEM_LIST_KEYS, cfg_get

    for key, kind in _container_slots().items():
        value = cfg_get(config, *key.split("."))
        if value is None or key in _SCALAR_AS_ONE_ITEM_LIST_KEYS:
            continue
        if isinstance(value, str) and _quoted_container_issue(key, kind, value, issues):
            continue
        if kind == "list" and not isinstance(value, list) and key not in _SCALAR_TOLERANT_LIST_KEYS:
            _scalar_in_list_issue(key, value, issues)


def validate_dotted_root_keys(config: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """``plugins.enabled:`` at column 0 is one literal root key nothing reads. Only keys whose first
    segment is a known section are flagged: other root keys are env-bridged on purpose."""
    from hermes_cli.config import _known_top_level_keys

    known = _known_top_level_keys()
    for key in config:
        if not isinstance(key, str) or "." not in key or key.startswith("_"):
            continue
        section, _, rest = key.partition(".")
        if section in known and rest:
            _issue(issues, "warning",
                   f"Root-level key '{key}' is a dotted path written as one key — Hermes reads nested "
                   f"YAML only, so this value is ignored",
                   f"Write it nested under '{section}:' in config.yaml, or run: hermes config unset {key} "
                   f"(removes this literal key), then hermes config set {key} <value>")

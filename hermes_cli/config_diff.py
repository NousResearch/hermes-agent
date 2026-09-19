"""Tree-form diff of ``DEFAULT_CONFIG`` vs the user's ``config.yaml`` (``hermes config diff``).

``load_config()`` resolves ``DEFAULT_CONFIG`` into every read, so "what did I actually change?"
is invisible in the merged view. This module diffs the two layers directly: the user layer comes
from ``load_user_config_effective`` (config.yaml + managed overlay + ``${VAR}`` expansion, no
defaults merged), the schema layer is ``DEFAULT_CONFIG``. Output mirrors the tree style of
``config get``:

    ~ key: <user value>   (default: <schema value>)   # set differently than the schema
    + key: <user value>                               # not defined by the schema

Unchanged keys and branches with no diffs are pruned. Credential-shaped leaves are masked with
the same key-shape rules as ``config get`` unless ``raw=True``. User-only leaves include both
deliberate unseeded-but-live keys (``tts.speed``, ``stt.provider``) and machinery-written
state (``known_builtin_toolsets``, ``platform_toolsets``); the schema walk cannot tell them
apart, same limitation as the phantom-key notice on ``config get``/``set``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from hermes_cli.colors import Colors, color
from hermes_cli.config_defaults import DEFAULT_CONFIG

# (marker, path-tuple, default_value, user_value); marker: "changed" | "user_only"
Entry = Tuple[str, Tuple[str, ...], Any, Any]


def _descend(mapping: Any, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    """Walk a dotted path into nested mappings. Returns (found, value)."""
    node = mapping
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _short(value: Any, limit: int = 70) -> str:
    """One-line rendering: flow-style YAML for containers, repr for scalars, ellipsis past limit."""
    if isinstance(value, (dict, list)):
        s = yaml.dump(value, default_flow_style=True, sort_keys=False, allow_unicode=True).strip()
    else:
        s = repr(value)
    return s if len(s) <= limit else s[:limit] + "…"


def _collect(default: Any, user: Any, path: Tuple[str, ...] = ()) -> Iterator[Entry]:
    """Walk both mappings; yield leaf diffs. Dict-vs-dict recurses, everything else is a leaf."""
    if not isinstance(default, dict) or not isinstance(user, dict):
        return
    for key in list(default) + [k for k in user if k not in default]:
        child = path + (str(key),)
        in_default, in_user = key in default, key in user
        d_val, u_val = default.get(key), user.get(key)
        if in_default and in_user and isinstance(d_val, dict) and isinstance(u_val, dict):
            yield from _collect(d_val, u_val, child)
        elif in_default and in_user and d_val != u_val:
            yield ("changed", child, d_val, u_val)
        elif in_user and not in_default:
            yield ("user_only", child, None, u_val)


def _mask_leaf(path: Tuple[str, ...], value: Any, raw: bool) -> Any:
    """Mask credential-shaped leaves for display (``--raw`` bypasses, same contract as get)."""
    if raw:
        return value
    from hermes_cli.config import _is_secret_config_key, redact_config_value
    if isinstance(value, str) and value and _is_secret_config_key(".".join(path)):
        from agent.redact import mask_secret
        return mask_secret(value)
    if isinstance(value, (dict, list)):
        return redact_config_value(value)
    return value


def diff_entries(*, raw: bool = False, defaults: Optional[Dict[str, Any]] = None,
                key: Optional[str] = None) -> List[Entry]:
    """Diff the effective user layer against the schema. Reads DEFAULT_CONFIG at call time
    so tests can monkeypatch the schema. ``key`` (dotted, e.g. ``agent`` or ``display.sections``)
    scopes the diff to that subtree on both sides — like ``git diff <path>``. Raises KeyError
    when the path exists in neither layer."""
    from hermes_cli.config_effective import load_user_config_effective
    user = load_user_config_effective()
    schema = DEFAULT_CONFIG if defaults is None else defaults
    if key:
        path = tuple(part for part in key.split(".") if part)
        d_found, schema = _descend(schema, path)
        u_found, user = _descend(user, path)
        if not d_found and not u_found:
            raise KeyError(key)
        # A scalar on either side: diff the nodes directly as one leaf.
        if not (isinstance(schema, dict) and isinstance(user, dict)):
            marker = "changed" if (d_found and u_found and schema != user) else (
                "user_only" if u_found and not d_found else None)
            return [(marker, (), schema, user)] if marker else []
    return [
        (marker, path, d_val, _mask_leaf(path, u_val, raw))
        for marker, path, d_val, u_val in _collect(schema, user)
    ]


def format_tree(entries: List[Entry], root_name: str = "") -> List[str]:
    """Render entries as an indented tree (one line per leaf, section headers pruned-empty).
    Git's color scheme: ``+`` green (added), ``~`` yellow (modified), the replaced default
    red (the removed side). ``root_name`` labels a whole-subtree scalar diff (empty path)."""
    tree: Dict[str, Any] = {}
    for marker, path, d_val, u_val in entries:
        node = tree
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1] if path else root_name or "(root)"] = (marker, d_val, u_val)

    lines: List[str] = []

    def walk(node: Dict[str, Any], indent: int) -> None:
        pad = "  " * indent
        for key, val in node.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                walk(val, indent + 1)
            else:
                marker, d_val, u_val = val
                if marker == "changed":
                    lines.append(pad + color("~", Colors.YELLOW) + f" {key}: "
                               + color(_short(u_val), Colors.GREEN)
                               + color(f"   (default: {_short(d_val, 40)})", Colors.RED))
                else:
                    lines.append(pad + color("+", Colors.GREEN) + f" {key}: "
                               + color(_short(u_val), Colors.GREEN))

    walk(tree, 0)
    return lines


def format_json(entries: List[Entry]) -> str:
    """Scriptable form: dotted keys, both values, already masked per the raw flag."""
    payload: Dict[str, Any] = {"changed": {}, "user_only": {}}
    for marker, path, d_val, u_val in entries:
        dotted = ".".join(path)
        if marker == "changed":
            payload["changed"][dotted] = {"default": d_val, "user": u_val}
        else:
            payload["user_only"][dotted] = u_val
    return json.dumps(payload, indent=2, default=str)


def run_config_diff(*, as_json: bool = False, raw: bool = False,
                   key: Optional[str] = None) -> None:
    """CLI entry: print the tree (or JSON) diff with a trailing summary. ``key`` scopes to a
    dotted subtree (``hermes config diff agent``); an unknown path exits 1 like git."""
    try:
        entries = diff_entries(raw=raw, key=key)
    except KeyError:
        print(color(f"✗ no config subtree '{key}' in defaults or your config.yaml", Colors.RED))
        raise SystemExit(1)
    if as_json:
        print(format_json(entries), flush=True)
        return
    lines = format_tree(entries, root_name=key or "")
    scope = f" under '{key}'" if key else ""
    print("\n".join(lines) if lines else f"(config.yaml matches defaults{scope})")
    changed = sum(1 for m, *_ in entries if m == "changed")
    user_only = sum(1 for m, *_ in entries if m == "user_only")
    print(f"\n-- {changed} changed, {user_only} user-only leaves", flush=True)

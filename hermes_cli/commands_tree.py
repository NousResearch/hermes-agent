"""``hermes commands`` — the whole CLI command tree in one invocation.

``hermes --help`` lists ~70 top-level groups with nothing under them, so
discovering that ``hermes cron resnap`` exists costs a second ``--help`` call
per group. Agents (Hermes itself via the ``hermes-agent`` skill, other
coding agents operating a Hermes install) pay that N+1 cost every time, and
hand-maintained command tables in skills and docs drift because nothing ties
them to the parser. Walking the live argparse tree makes the inventory
self-documenting; ``--json`` feeds tooling without scraping text.

Ported from block/buzz#7584 (``buzz --help`` renders the full tree); kept as
a separate subcommand here because 400+ rows would bury the human-facing
``--help``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

_INDENT = 2
_NAME_COL = 30


def _subcommands(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """Visible children of ``parser`` in registration order (= ``--help`` order).

    A subcommand registered without help is hidden on purpose (deprecated aliases like
    ``login``); it is omitted exactly as ``--help`` omits it. ``_choices_actions`` holds one
    entry per canonical name, so aliases never duplicate a node.
    """
    nodes: list[dict[str, Any]] = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for pseudo in action._choices_actions:
            child = action.choices.get(pseudo.dest)
            help_text = pseudo.help
            if child is None or not help_text or help_text is argparse.SUPPRESS:
                continue
            nodes.append({
                "name": pseudo.dest,
                "help": " ".join(str(help_text).split()),
                "subcommands": _subcommands(child),
            })
    return nodes


def _rows(nodes: list[dict[str, Any]], depth: int = 0) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for node in nodes:
        rows.append((depth, node["name"], node["help"]))
        rows.extend(_rows(node["subcommands"], depth + 1))
    return rows


def render_command_tree(parser: argparse.ArgumentParser, *, as_json: bool = False) -> str:
    """Every visible group and subcommand under ``parser`` as an indented text tree or JSON."""
    nodes = _subcommands(parser)
    if as_json:
        return json.dumps({"prog": parser.prog, "subcommands": nodes}, indent=2)

    rows = _rows(nodes)
    lines = [
        f"{parser.prog} command tree — {len(nodes)} groups, {len(rows)} commands.",
        f"Flags for any node: `{parser.prog} <command> [<subcommand>] --help`.",
        "",
    ]
    for depth, name, help_text in rows:
        label = " " * (_INDENT * depth) + name
        pad = " " * max(2, _NAME_COL - len(label))
        lines.append(f"{label}{pad}{help_text}")
    return "\n".join(lines)


def cmd_commands(args, parser: argparse.ArgumentParser) -> None:
    print(render_command_tree(parser, as_json=getattr(args, "json", False)))

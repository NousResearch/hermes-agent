"""A small systemd unit-file parser for the deploy tests.

It follows systemd.syntax(7) closely enough to assert on directives: ``[Section]``
headers, ``Key=value`` lines, ``#``/``;`` comments, trailing-backslash continuation
lines, repeated keys that accumulate, and an empty assignment (``Key=``) that
resets the list, as systemd does for list-valued settings.
"""

from __future__ import annotations

from typing import Dict, List

Unit = Dict[str, Dict[str, List[str]]]


class UnitSyntaxError(ValueError):
    pass


def parse_unit(text: str) -> Unit:
    sections: Unit = {}
    current = None
    pending = ""
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not pending and (not line or line.startswith(("#", ";"))):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        line, pending = (pending + line).strip(), ""
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if name in sections:
                raise UnitSyntaxError(f"line {number}: section [{name}] repeated")
            current = sections[name] = {}
            continue
        if current is None:
            raise UnitSyntaxError(f"line {number}: directive outside a section: {line}")
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or not key.replace("-", "").isalnum():
            raise UnitSyntaxError(f"line {number}: not a directive: {line}")
        value = value.strip()
        if value:
            current.setdefault(key, []).append(value)
        else:
            current[key] = []
    if pending:
        raise UnitSyntaxError("file ends inside a continuation line")
    return sections


def space_list(unit: Unit, section: str, key: str) -> List[str]:
    """Every whitespace-separated item of a list-valued directive, across its lines."""
    return [item for value in unit.get(section, {}).get(key, []) for item in value.split()]

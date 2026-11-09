"""Read pnpm-lock.yaml without a YAML parser.

CI runs these scripts on the runner's bare Python, before any dependency is
installed. pnpm writes this file itself in a fixed shape (two-space indent, one
key per line); anything that does not fit that shape is an error, never a guess.
scripts/build/pnpm-lock.mjs is the Node twin of this reader.
"""
from __future__ import annotations

import json
import re

_KEY = re.compile(r"^  (\S.*?):(?: .*)?$")
# A key (quoted, or plain up to the first ``: ``) and its inline value, if any.
_ENTRY = re.compile(r"""^('(?:[^']|'')*'|"(?:[^"\\]|\\.)*"|[^'"\s].*?):(?: (.*))?$""")
# Any top-level key (plain, or quoted as pnpm's emitter quotes ambiguous keys) ends
# a section: pnpm keeps root keys it does not define (Bit's ``bit:`` block) and adds
# new ones over time. Any other column-0 line (a comment, a conflict marker, a
# tab-indented field) fails instead of silently truncating the section.
_TOP_LEVEL = re.compile(r"""^(?:\w.*?|'(?:[^']|'')*'|"(?:[^"\\]|\\.)*"):(?: .*)?$""")


def _unquote(key: str) -> str:
    if key.startswith("'") and key.endswith("'"):
        return key[1:-1].replace("''", "'")
    if key.startswith('"') and key.endswith('"'):
        return json.loads(key)
    return key


def _document_lines(text: str) -> list[str]:
    """The lines of the lock's only YAML document, after checking its format version.

    The repository lock is one YAML document (``pmOnFail: ignore`` keeps pnpm from
    prepending a ``packageManagerDependencies`` document); several documents are
    refused, because lockfile scanners read only the first one.
    """
    documents = [part for part in re.split(r"^---\r?\n", text, flags=re.MULTILINE) if part.strip()]
    if len(documents) != 1:
        raise ValueError(f"pnpm-lock.yaml: expected one YAML document, found {len(documents)}")
    lines = documents[0].splitlines()
    version = next((line for line in lines if line.startswith("lockfileVersion:")), None)
    if version != "lockfileVersion: '9.0'":
        raise ValueError(f"pnpm-lock.yaml: unsupported {version or 'lockfileVersion <missing>'}")
    return lines


def _section_lines(text: str, name: str) -> list[str]:
    """The lines of top-level section ``name`` (empty when the lock has none).

    The section runs to the next top-level key or the end of the file; any other
    line that is not indented by spaces is refused.
    """
    lines = _document_lines(text)
    if f"{name}:" not in lines:
        return []
    body = []
    for line in lines[lines.index(f"{name}:") + 1:]:
        if line.startswith(" ") or not line.strip():
            body.append(line)
        elif _TOP_LEVEL.match(line):
            break
        else:
            raise ValueError(f"pnpm-lock.yaml: unreadable {name} line: {line}")
    return body


def _scalar(value: str) -> dict | str:
    """An inline value: ``{}`` (an entry without fields) or a (maybe quoted) string.

    Other flow values (``resolution: {integrity: ...}``, ``cpu: [x64]``) stay raw text.
    """
    return {} if value == "{}" else _unquote(value)


def section(text: str, name: str) -> dict:
    """A top-level map (``importers``, ``packages``, ``snapshots``) as nested dicts.

    Block maps become dicts, ``- item`` blocks lists, everything else strings. A
    lockfile without dependencies has no ``packages``/``snapshots`` section at all.
    """
    root: dict = {}
    # (indent of the lines this frame holds, container, its parent, its key in the parent)
    stack: list[tuple[int, dict | list, dict | None, str | None]] = [(2, root, None, None)]
    for line in _section_lines(text, name):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        level, container, parent, key = stack[-1]
        if indent != level:
            raise ValueError(f"pnpm-lock.yaml: unreadable {name} line: {line}")
        body = line[indent:]
        if body.startswith("- "):
            if container == {} and parent is not None:
                container = parent[key] = []  # the first child shows the block is a list
                stack[-1] = (level, container, parent, key)
            if not isinstance(container, list):
                raise ValueError(f"pnpm-lock.yaml: unreadable {name} line: {line}")
            container.append(_unquote(body[2:]))
            continue
        match = _ENTRY.match(body)
        if not match or not isinstance(container, dict):
            raise ValueError(f"pnpm-lock.yaml: unreadable {name} line: {line}")
        entry = _unquote(match[1])
        if match[2] is None:
            container[entry] = {}
            stack.append((indent + 2, container[entry], container, entry))
        else:
            container[entry] = _scalar(match[2])
    return root


def section_keys(text: str, name: str) -> list[str]:
    """The keys of a top-level map (``importers``, ``packages``) of the lock.

    Reads only the keys, so it does not depend on the shape of each entry's fields.
    """
    keys = []
    for line in _section_lines(text, name):
        if not line.strip() or line.startswith("   "):
            continue  # blank separators and each entry's own fields
        match = _KEY.match(line)
        if not match:
            raise ValueError(f"pnpm-lock.yaml: unreadable {name} line: {line}")
        keys.append(_unquote(match[1]))
    return keys

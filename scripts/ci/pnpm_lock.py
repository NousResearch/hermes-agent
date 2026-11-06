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


def _unquote(key: str) -> str:
    if key.startswith("'") and key.endswith("'"):
        return key[1:-1].replace("''", "'")
    if key.startswith('"') and key.endswith('"'):
        return json.loads(key)
    return key


def section_keys(text: str, section: str) -> list[str]:
    """The keys of a top-level map (``importers``, ``packages``) of the lock.

    The repository lock is one YAML document (``pmOnFail: ignore`` keeps pnpm from
    prepending a ``packageManagerDependencies`` document); several documents are
    refused, because lockfile scanners read only the first one. A lockfile without
    dependencies has no ``packages`` section at all.
    """
    documents = [part for part in re.split(r"^---\r?\n", text, flags=re.MULTILINE) if part.strip()]
    if len(documents) != 1:
        raise ValueError(f"pnpm-lock.yaml: expected one YAML document, found {len(documents)}")
    lines = documents[0].splitlines()
    version = next((line for line in lines if line.startswith("lockfileVersion:")), None)
    if version != "lockfileVersion: '9.0'":
        raise ValueError(f"pnpm-lock.yaml: unsupported {version or 'lockfileVersion <missing>'}")
    if f"{section}:" not in lines:
        return []
    keys = []
    for line in lines[lines.index(f"{section}:") + 1:]:
        if not line.strip() or line.startswith("   "):
            continue  # blank separators and each entry's own fields
        if not line.startswith(" "):
            break
        match = _KEY.match(line)
        if not match:
            raise ValueError(f"pnpm-lock.yaml: unreadable {section} line: {line}")
        keys.append(_unquote(match[1]))
    return keys

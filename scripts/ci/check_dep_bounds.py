#!/usr/bin/env python3
"""Print PyPI requirement strings that carry no upper bound, one per line.

Reads the added ``pyproject.toml`` lines of a diff on stdin (the supply-chain-audit
``dep-bounds`` job pipes them in). CONTRIBUTING's pinning policy wants every PyPI
dependency capped: ``>=floor,<next_major`` or an exact ``==`` pin. A requirement is
reported when it has version clauses and none of them bounds it from above
(``<``, ``<=``, ``==``, ``===``, ``~=``).

Every quoted string that parses as ``name[extras] <clauses> [; marker]`` is checked,
so environment markers, spaces around operators, dotted names (``ruamel.yaml``)
and clauses in any order are all covered. Direct references (``name @ git+...``)
are pinned by commit and skipped, as are bare names: a quoted word on an added
line cannot be told apart from any other TOML string.

Run: git diff BASE...HEAD -- pyproject.toml | grep '^+' | python3 scripts/ci/check_dep_bounds.py
"""

from __future__ import annotations

import re
import sys
from typing import Iterable, List

_QUOTED = re.compile(r'"([^"\n]*)"|\'([^\'\n]*)\'')
_REQUIREMENT = re.compile(
    r"^\s*[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\s*(?:\[[^\]]*\])?\s*"
    r"(?P<clauses>(?:===|==|~=|!=|<=|>=|<|>)[^;]*?)\s*(?:;.*)?$"
)
_CLAUSE_OP = re.compile(r"^\s*(===|==|~=|!=|<=|>=|<|>)\s*[0-9*]")
_UPPER_BOUNDING = {"<", "<=", "==", "===", "~="}


def unbounded_requirements(lines: Iterable[str]) -> List[str]:
    """Requirement strings on *lines* whose version clauses leave the top open."""
    found = []
    for line in lines:
        for match in _QUOTED.finditer(line):
            text = match.group(1) if match.group(1) is not None else match.group(2)
            req = _REQUIREMENT.match(text)
            if req is None:
                continue
            ops = [m.group(1) for m in map(_CLAUSE_OP.match, req.group("clauses").split(",")) if m]
            if ops and not _UPPER_BOUNDING.intersection(ops):
                found.append(f'"{text}"')
    return found


def main() -> int:
    for requirement in unbounded_requirements(sys.stdin):
        print(requirement)
    return 0


if __name__ == "__main__":
    sys.exit(main())

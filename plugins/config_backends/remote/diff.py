"""Remote write computation (contract.md §6.4, §15.2).

Pure functions over JSON-shaped documents; no I/O. The plane runs the authoritative write check;
the agent runs the same check before sending so a locked write fails with a clear message and
nothing goes on the wire.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .paths import Path, covers, encode

Locks = Sequence[Tuple[Path, str]]


def json_equal(a: Any, b: Any) -> bool:
    """Structural JSON equality: numbers by value, but ``True`` is not ``1``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def diff(base: Dict[str, Any], new: Dict[str, Any], prefix: Path = ()) -> Tuple[Dict[Path, Any], List[Path]]:
    """``(set, unset)`` turning ``base`` into ``new`` (§15.2 step 5). A changed list is one
    whole-list ``set`` (D8); a key equal to its base value is not sent."""
    sets: Dict[Path, Any] = {}
    unsets: List[Path] = []
    for key, value in new.items():
        p = prefix + (key,)
        if key not in base:
            sets[p] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            s, u = diff(base[key], value, p)
            sets.update(s)
            unsets.extend(u)
        elif not json_equal(base[key], value):
            sets[p] = value
    unsets.extend(prefix + (key,) for key in base if key not in new)
    return sets, unsets


def _pop_path(doc: Dict[str, Any], path: Path) -> bool:
    node: Any = doc
    for seg in path[:-1]:
        if not isinstance(node, dict) or not isinstance(node.get(seg), dict):
            return False
        node = node[seg]
    if isinstance(node, dict) and path[-1] in node:
        del node[path[-1]]
        return True
    return False


def _get_path(doc: Dict[str, Any], path: Path) -> Tuple[bool, Any]:
    node: Any = doc
    for seg in path:
        if not isinstance(node, dict) or seg not in node:
            return False, None
        node = node[seg]
    return True, node


def strip_locked(base: Dict[str, Any], new: Dict[str, Any], locks: Locks) -> Tuple[Dict[str, Any], Dict[str, Any], List[Path]]:
    """Drop every locked path from BOTH documents (§15.2 step 1), so a bulk write neither sets
    nor unsets a key it may not touch. Returns the copies and the locked paths whose new value
    differed from the effective one (the ones the caller tried to change)."""
    base, new = copy.deepcopy(base), copy.deepcopy(new)
    changed: List[Path] = []
    for lock, _level in locks:
        in_base, base_val = _get_path(base, lock)
        in_new, new_val = _get_path(new, lock)
        if in_base != in_new or (in_new and not json_equal(base_val, new_val)):
            changed.append(lock)
        _pop_path(base, lock)
        _pop_path(new, lock)
    return base, new, changed


def _touched(path: Path, value: Any) -> Iterator[Tuple[Path, Any]]:
    """``(touched path, value written there)`` for ``set path = value`` (§6.4 rule 2): the path
    itself and every path inside ``value``, descending through objects only."""
    yield path, value
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _touched(path + (k,), v)


def write_check(sets: Dict[Path, Any], unsets: Sequence[Path], locks: Locks) -> Optional[Tuple[Path, str]]:
    """The ``(lock path, level)`` that refuses this write (§6.4 rules 1-3), or None."""
    for p in unsets:
        for lock, level in locks:
            if covers(lock, p):
                return lock, level
    for p, v in sets.items():
        for t, tv in _touched(p, v):
            for lock, level in locks:
                if covers(lock, t):
                    return lock, level
                # R7: a non-object written at a strict ancestor of a lock would replace the
                # locked subtree through deep merge.
                if len(t) < len(lock) and covers(t, lock) and not isinstance(tv, dict):
                    return lock, level
    return None


def encode_changes(sets: Dict[Path, Any], unsets: Sequence[Path]) -> Tuple[Dict[str, Any], List[str]]:
    return {encode(p): v for p, v in sets.items()}, [encode(p) for p in unsets]

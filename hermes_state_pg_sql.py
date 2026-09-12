"""Translate the SQLite SQL emitted by SessionDB to PostgreSQL.

Keep adaptation here, beside the PostgreSQL connection wrapper. Quoted SQL
and comments must survive unchanged: they can contain question marks, percent
signs, or text that happens to look like a function call.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

_STRICT_ENV = "HERMES_PG_ADAPTER_STRICT"
_SQLITE_JSON_FNS = ("json_extract", "json_type", "json_remove", "json_set", "json_valid", "json_object")
_STRICT_FORBIDDEN = (
    *_SQLITE_JSON_FNS, "json_quote", "json_patch", "json_each", "json_array_length",
    "autoincrement", "pragma", " glob ", "strftime", "fts5", "match ", "x'", "instr(", "char(",
)

# State SQL uses ordinary quoted strings/identifiers and comments, not
# dollar-quoted procedural bodies (PostgreSQL-only DDL runs on the raw driver).
_OPAQUE_SQL = r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*(?:\n|$)|/\*[\s\S]*?\*/"
_OPAQUE_RE = re.compile(_OPAQUE_SQL)
_CALL_OR_OPAQUE_RE = re.compile(_OPAQUE_SQL + r"|\b(?P<function>[A-Za-z_]\w*)\s*\(")


def _map_code(sql: str, transform: Callable[[str], str]) -> str:
    parts = []
    start = 0
    for match in _OPAQUE_RE.finditer(sql):
        parts.extend((transform(sql[start:match.start()]), match.group()))
        start = match.end()
    parts.append(transform(sql[start:]))
    return "".join(parts)


def _code_only(sql: str) -> str:
    # Keep empty quotes so strict mode can still recognize untranslated X'...'.
    return _OPAQUE_RE.sub(lambda m: "''" if m.group().startswith("'") else " ", sql)


def _split_balanced_args(sql: str, open_idx: int) -> Tuple[List[str], int]:
    """Split a SQL call without splitting nested calls, strings, or comments."""
    args: List[str] = []
    depth = 1
    start = i = open_idx + 1
    while i < len(sql):
        opaque = _OPAQUE_RE.match(sql, i)
        if opaque:
            i = opaque.end()
            continue
        ch = sql[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append(sql[start:i].strip())
                return args, i + 1
        elif ch == "," and depth == 1:
            args.append(sql[start:i].strip())
            start = i + 1
        i += 1
    raise ValueError(f"unbalanced parentheses in SQL near offset {open_idx}")


def _json_path_key(arg: str) -> Optional[str]:
    match = re.fullmatch(r"'\$\.([A-Za-z_]\w*)'", arg.strip())
    return match.group(1) if match else None


def _jsonb(expr: str) -> str:
    return f"COALESCE({expr}, '{{}}')::jsonb"


def _translate_json_call(name: str, args: List[str]) -> Optional[str]:
    low = name.lower()
    if low == "json_object" and args in ([], [""]):
        # SQLite returns TEXT. A jsonb expression here cannot share a CASE
        # branch with the TEXT model_config column used by _sql_json_extract.
        return "'{}'"
    if low == "json_valid" and len(args) == 1:
        return f"({args[0]} IS JSON)"
    if len(args) < 2:
        return None
    key = _json_path_key(args[1])
    if key is None:
        return None
    col = args[0]
    if low == "json_extract" and len(args) == 2:
        return f"({_jsonb(col)} ->> '{key}')"
    if low == "json_type" and len(args) == 2:
        return f"jsonb_typeof({_jsonb(col)} -> '{key}')"
    if low == "json_remove" and len(args) == 2:
        return f"(({_jsonb(col)} - '{key}')::text)"
    if low == "json_set" and len(args) == 3:
        return f"(jsonb_set({_jsonb(col)}, '{{{key}}}', to_jsonb({args[2]}))::text)"
    return None


def _translate_call(name: str, args: List[str], *, json_only: bool) -> Optional[str]:
    low = name.lower()
    if low in _SQLITE_JSON_FNS:
        return _translate_json_call(low, args)
    if json_only:
        return None
    if low == "instr" and len(args) == 2:
        return f"strpos({args[0]}, {args[1]})"
    if low == "char" and args and args != [""]:
        return " || ".join(f"chr({arg})" for arg in args)
    # These are the non-null scalar clamps used by SessionDB, not aggregate
    # MIN/MAX (which keep their PostgreSQL spelling).
    if low in ("min", "max") and len(args) > 1:
        function = "least" if low == "min" else "greatest"
        return f"{function}({', '.join(args)})"
    if low == "substr":
        if len(args) == 2 and re.fullmatch(r"-[1-9]\d*", args[1]):
            # SQLite counts a negative start from the end; PostgreSQL counts
            # from before the beginning. Skill previews need the actual tail.
            return f"right({args[0]}, {-int(args[1])})"
        if len(args) == 3 and args[1] == "1" and not re.fullmatch(r"\d+", args[2]):
            # The preview's instr(marker)-1 is -1 when the marker is absent.
            # SQLite returns '' at start=1; PostgreSQL otherwise raises even
            # when another part of the WHERE predicate rejects that row.
            return f"substr({args[0]}, 1, greatest(0, {args[2]}))"
    return None


def _rewrite_calls(sql: str, *, json_only: bool = False) -> str:
    """Rewrite nested calls inside out, with no fixed cap on sibling calls."""
    parts = []
    start = 0
    while match := _CALL_OR_OPAQUE_RE.search(sql, start):
        name = match.group("function")
        if name is None:
            parts.append(sql[start:match.end()])
            start = match.end()
            continue
        open_idx = match.end() - 1
        try:
            args, end = _split_balanced_args(sql, open_idx)
        except ValueError:
            # Leave incomplete/unknown SQL for strict mode or the server.
            break
        rewritten = [_rewrite_calls(arg, json_only=json_only) for arg in args]
        replacement = _translate_call(name, rewritten, json_only=json_only)
        if replacement is None:
            replacement = sql[match.start():open_idx + 1] + ", ".join(rewritten) + ")"
        parts.extend((sql[start:match.start()], replacement))
        start = end
    parts.append(sql[start:])
    return "".join(parts)


def _rewrite_sqlite_json_fns(sql: str) -> str:
    return _rewrite_calls(sql, json_only=True)


def _strict_enabled() -> bool:
    return os.environ.get(_STRICT_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _rewrite_code(sql: str) -> str:
    # Lease claims and transcript guards read before they mutate. SQLite's
    # writer lock serializes the whole callback; READ COMMITTED can act on a
    # stale read after a concurrent refresh. PostgreSQL must instead abort
    # that transaction so _execute_write retries the entire callback.
    sql = re.sub(r"\bBEGIN\s+IMMEDIATE\b", "BEGIN ISOLATION LEVEL SERIALIZABLE", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", "INSERT INTO", sql, flags=re.IGNORECASE)
    sql = re.sub(
        r"\bIS\s+(NOT\s+)?(\?|%s)(?=\s|[),;]|$)",
        lambda m: ("IS DISTINCT FROM " if m.group(1) else "IS NOT DISTINCT FROM ") + m.group(2),
        sql, flags=re.IGNORECASE,
    )
    # SQLite uses -1 for unlimited; PostgreSQL uses NULL. Keep every bind once
    # so call sites can keep their original parameter lists.
    return re.sub(
        r"\bLIMIT\s+(\?|%s|-1)(?=\s|[);]|$)",
        lambda m: f"LIMIT NULLIF({m.group(1)}, -1)", sql, flags=re.IGNORECASE,
    )


def _bind_parameters(sql: str) -> str:
    """Translate positional/named binds without changing casts or quoted SQL."""
    def code(part: str) -> str:
        # Match native placeholders before literal percent signs, and skip
        # both colons of PostgreSQL ::type casts. A single pass keeps newly
        # generated %(name)s / %s placeholders out of percent escaping.
        return re.sub(
            r"%\([^)]+\)[sbt]|%[sbt]|%%|%|\?|(?<!:):(?P<name>[A-Za-z_]\w*)",
            lambda m: f"%({m.group('name')})s" if m.group("name") else {"?": "%s", "%": "%%"}.get(m.group(), m.group()),
            part,
        )

    parts = []
    start = 0
    for match in _OPAQUE_RE.finditer(sql):
        parts.extend((code(sql[start:match.start()]), match.group().replace("%", "%%")))
        start = match.end()
    parts.append(code(sql[start:]))
    return "".join(parts)


def _translate_sql(sql: str) -> str:
    """Translate the closed SQLite dialect used by SessionDB, preserving binds."""
    insert_or_ignore = re.search(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", _code_only(sql), re.I) is not None
    # The only hexadecimal literals emitted by state SQL represent newline and
    # carriage return. Do not replace those character sequences inside strings.
    sql = re.sub(
        _OPAQUE_SQL + r"|\bX'(?P<hex>0A|0D)'",
        lambda m: f"chr({int(m.group('hex'), 16)})" if m.group('hex') else m.group(),
        sql, flags=re.IGNORECASE,
    )
    translated = _map_code(_rewrite_calls(sql), _rewrite_code)
    if insert_or_ignore and "ON CONFLICT" not in _code_only(translated).upper():
        translated = translated.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    if _strict_enabled():
        lowered = _code_only(translated).lower()
        for token in _STRICT_FORBIDDEN:
            if token in lowered:
                raise RuntimeError(
                    "PostgreSQL adapter strict mode: untranslated SQLite-only "
                    f"idiom {token!r} in statement: {sql.strip()[:160]}"
                )
    return _bind_parameters(translated)


def _needs_returning_id(sql: str) -> bool:
    return re.match(r"\s*INSERT\s+INTO\s+messages\b", sql, re.I) is not None

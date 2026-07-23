#!/usr/bin/env python3
"""
Ephemeral DuckDB runner for skill workflows.

Loads optional file bindings into an in-memory connection, runs one SQL statement,
and prints rows as JSON (default) or TSV. Paths are bound as parameters — never
concatenate untrusted paths into raw SQL strings from the shell.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_alias(name: str) -> str:
    if not _ALIAS_RE.match(name):
        raise ValueError(f"invalid view alias: {name!r} (use letters, digits, underscore)")
    return name


def _sql_path_literal(raw: str) -> str:
    """
    Resolve a user path to an absolute file path and return a DuckDB string literal body.
    Rejects characters that would break out of a single-quoted SQL string.
    """
    p = Path(raw).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"not a readable file: {p}")
    posix = p.as_posix()
    if "'" in posix or "\x00" in posix:
        raise ValueError("path contains characters that cannot be used in SQL string literals")
    return posix


def _register_file(con: object, kind: str, alias: str, path: str) -> None:
    """Create a view `alias` from a single file (literal path; validated above)."""
    alias = _validate_alias(alias)
    lit = _sql_path_literal(path)
    if kind == "parquet":
        sql = f'CREATE OR REPLACE VIEW "{alias}" AS SELECT * FROM read_parquet(\'{lit}\')'
    elif kind == "csv":
        sql = f'CREATE OR REPLACE VIEW "{alias}" AS SELECT * FROM read_csv_auto(\'{lit}\')'
    else:
        raise ValueError(f"unsupported kind: {kind}")
    con.execute(sql)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DuckDB SQL against in-memory DB.")
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Single DuckDB SQL statement to execute (SELECT recommended).",
    )
    parser.add_argument(
        "--parquet",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="Register Parquet file as a view (repeatable). Example: --parquet sales=data/sales.parquet",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="Register CSV file as a view via read_csv_auto (repeatable).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "tsv"),
        default="json",
        help="Row output format (default: json).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        import duckdb
    except ImportError:
        print(
            "duckdb package not found. Install with: uv pip install duckdb\n"
            "Or run: uv run --with duckdb python ...",
            file=sys.stderr,
        )
        return 2

    con = duckdb.connect(":memory:")
    try:
        for spec in args.parquet:
            if "=" not in spec:
                print("--parquet requires ALIAS=PATH", file=sys.stderr)
                return 1
            alias, path = spec.split("=", 1)
            _register_file(con, "parquet", alias.strip(), path.strip())
        for spec in args.csv:
            if "=" not in spec:
                print("--csv requires ALIAS=PATH", file=sys.stderr)
                return 1
            alias, path = spec.split("=", 1)
            _register_file(con, "csv", alias.strip(), path.strip())

        result = con.execute(args.query)
        rows = result.fetchall()
        colnames = [d[0] for d in result.description] if result.description else []

        if args.format == "json":
            out = [dict(zip(colnames, row)) for row in rows]
            json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\t".join(colnames) + "\n")
            for row in rows:
                sys.stdout.write("\t".join(str(c) for c in row) + "\n")
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

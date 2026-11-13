#!/usr/bin/env python3
"""Universal SQL executor for Hermes Agent."""
import argparse
import os
import sys
import csv
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

try:
    from tabulate import tabulate
except ImportError:
    print("Error: tabulate not installed. Run: pip install tabulate", file=sys.stderr)
    sys.exit(1)

# Driver imports (lazy loaded)
_DRIVERS = {}

def get_driver(dsn: str):
    """Get appropriate driver for DSN."""
    global _DRIVERS
    
    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        if "postgres" not in _DRIVERS:
            try:
                import psycopg2
                import psycopg2.extras
                _DRIVERS["postgres"] = PostgresDriver()
            except ImportError:
                print("Error: psycopg2-binary not installed. Run: pip install psycopg2-binary", file=sys.stderr)
                sys.exit(1)
        return _DRIVERS["postgres"]
    
    elif dsn.startswith("mysql://"):
        if "mysql" not in _DRIVERS:
            try:
                import pymysql
                _DRIVERS["mysql"] = MySQLDriver()
            except ImportError:
                print("Error: pymysql not installed. Run: pip install pymysql", file=sys.stderr)
                sys.exit(1)
        return _DRIVERS["mysql"]
    
    elif dsn.startswith("sqlite://"):
        if "sqlite" not in _DRIVERS:
            _DRIVERS["sqlite"] = SQLiteDriver()
        return _DRIVERS["sqlite"]
    
    elif dsn.startswith("clickhouse://"):
        if "clickhouse" not in _DRIVERS:
            try:
                import clickhouse_connect
                _DRIVERS["clickhouse"] = ClickHouseDriver()
            except ImportError:
                print("Error: clickhouse-connect not installed. Run: pip install clickhouse-connect", file=sys.stderr)
                sys.exit(1)
        return _DRIVERS["clickhouse"]
    
    else:
        print(f"Error: Unsupported DSN scheme. Supported: postgresql://, mysql://, sqlite://, clickhouse://", file=sys.stderr)
        sys.exit(1)


class BaseDriver:
    def execute(self, sql: str, params: dict, limit: int) -> Tuple[List[str], List[Tuple]]:
        raise NotImplementedError


class PostgresDriver(BaseDriver):
    def execute(self, sql: str, params: dict, limit: int) -> Tuple[List[str], List[Tuple]]:
        import psycopg2
        import psycopg2.extras
        
        # Parse DSN from environment or use default
        dsn = os.environ.get("DATABASE_DSN", "")
        if not dsn:
            raise ValueError("DATABASE_DSN environment variable not set")
        
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Apply limit if not already present
                if limit and "limit" not in sql.lower():
                    sql = f"{sql.rstrip(';')} LIMIT {limit}"
                cur.execute(sql, params)
                rows = cur.fetchall()
                if rows:
                    cols = list(rows[0].keys())
                    data = [tuple(row.values()) for row in rows]
                else:
                    cols = [desc[0] for desc in cur.description] if cur.description else []
                    data = []
                return cols, data
        finally:
            conn.close()


class MySQLDriver(BaseDriver):
    def execute(self, sql: str, params: dict, limit: int) -> Tuple[List[str], List[Tuple]]:
        import pymysql
        
        dsn = os.environ.get("DATABASE_DSN", "")
        if not dsn:
            raise ValueError("DATABASE_DSN environment variable not set")
        
        # Parse mysql://user:pass@host:port/db
        import urllib.parse
        parsed = urllib.parse.urlparse(dsn)
        conn = pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") if parsed.path else None,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            with conn.cursor() as cur:
                if limit and "limit" not in sql.lower():
                    sql = f"{sql.rstrip(';')} LIMIT {limit}"
                cur.execute(sql, params)
                rows = cur.fetchall()
                if rows:
                    cols = list(rows[0].keys())
                    data = [tuple(row.values()) for row in rows]
                else:
                    cols = [desc[0] for desc in cur.description] if cur.description else []
                    data = []
                return cols, data
        finally:
            conn.close()


class SQLiteDriver(BaseDriver):
    def execute(self, sql: str, params: dict, limit: int) -> Tuple[List[str], List[Tuple]]:
        import sqlite3
        
        dsn = os.environ.get("DATABASE_DSN", "")
        if not dsn:
            raise ValueError("DATABASE_DSN environment variable not set")
        
        # sqlite:///path/to/db or sqlite:////absolute/path
        db_path = dsn.replace("sqlite://", "")
        if db_path.startswith("///"):
            db_path = db_path[2:]  # sqlite:///relative -> relative
        elif db_path.startswith("//"):
            db_path = db_path[1:]  # sqlite:////absolute -> /absolute
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            if limit and "limit" not in sql.lower():
                sql = f"{sql.rstrip(';')} LIMIT {limit}"
            cur.execute(sql, params)
            rows = cur.fetchall()
            if rows:
                cols = list(rows[0].keys())
                data = [tuple(row[key] for key in cols) for row in rows]
            else:
                cols = [desc[0] for desc in cur.description] if cur.description else []
                data = []
            return cols, data
        finally:
            conn.close()


class ClickHouseDriver(BaseDriver):
    def execute(self, sql: str, params: dict, limit: int) -> Tuple[List[str], List[Tuple]]:
        import clickhouse_connect
        
        dsn = os.environ.get("DATABASE_DSN", "")
        if not dsn:
            raise ValueError("DATABASE_DSN environment variable not set")
        
        # clickhouse://user:pass@host:port/db
        import urllib.parse
        parsed = urllib.parse.urlparse(dsn)
        client = clickhouse_connect.get_client(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8123,
            username=parsed.username or "default",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") if parsed.path else "default"
        )
        try:
            if limit and "limit" not in sql.lower():
                sql = f"{sql.rstrip(';')} LIMIT {limit}"
            result = client.query(sql, parameters=params)
            cols = result.column_names
            data = result.result_rows
            return cols, data
        finally:
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Execute SQL across databases")
    parser.add_argument("--dsn", help="Database DSN (postgresql://, mysql://, sqlite://, clickhouse://). Can also use DATABASE_DSN env var.")
    parser.add_argument("--query", help="SQL query string")
    parser.add_argument("--file", help="SQL file path")
    parser.add_argument("--format", choices=["table", "csv", "json", "markdown"], default="table")
    parser.add_argument("--limit", type=int, default=1000, help="Row limit (default: 1000)")
    parser.add_argument("--params", help="JSON params for parameterized query")
    args = parser.parse_args()

    # Use env var if --dsn not provided
    dsn = args.dsn or os.environ.get("DATABASE_DSN")
    if not dsn:
        parser.error("DSN required via --dsn or DATABASE_DSN environment variable")

    if not args.query and not args.file:
        parser.error("Either --query or --file required")

    sql = args.query or Path(args.file).read_text()
    params = json.loads(args.params) if args.params else {}

    driver = get_driver(dsn)
    cols, rows = driver.execute(sql, params, limit=args.limit)

    if args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(cols)
        writer.writerows(rows)
    elif args.format == "json":
        json.dump([dict(zip(cols, r)) for r in rows], sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    elif args.format == "markdown":
        print(tabulate(rows, headers=cols, tablefmt="pipe"))
    else:
        print(tabulate(rows, headers=cols, tablefmt="grid"))


if __name__ == "__main__":
    main()
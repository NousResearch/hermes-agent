"""CLI commands for managing ClickHouse memory provider.

Registered via the plugin system — ``hermes clickhouse <command>``.

Commands
--------
- ``hermes clickhouse setup`` — interactive setup wizard (config + schema)
- ``hermes clickhouse status`` — check connection and table stats
- ``hermes clickhouse query <sql>`` — run raw SQL (admin only)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_cli(subparsers) -> None:
    """Register ``hermes clickhouse`` subcommand."""
    parser = subparsers.add_parser(
        "clickhouse",
        help="Manage ClickHouse memory provider",
        description="Setup, status, and admin commands for ClickHouse memory provider",
    )
    parser.set_defaults(command="clickhouse")

    sub = parser.add_subparsers(dest="clickhouse_command")

    # Setup
    setup_parser = sub.add_parser(
        "setup",
        help="Configure ClickHouse connection and create schema",
    )
    setup_parser.add_argument("--host", default="localhost", help="ClickHouse host")
    setup_parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    setup_parser.add_argument("--user", default="default", help="ClickHouse user")
    setup_parser.add_argument("--password", help="ClickHouse password")
    setup_parser.add_argument("--database", default="hermes_memory", help="Database name")
    setup_parser.add_argument("--ttl-days", type=int, default=30, help="TTL in days")

    # Status
    status_parser = sub.add_parser(
        "status",
        help="Check ClickHouse connection and table stats",
    )
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Query (admin)
    query_parser = sub.add_parser(
        "query",
        help="Run raw SQL query (admin)",
    )
    query_parser.add_argument("sql", help="SQL query to execute")

    for p in (setup_parser, status_parser, query_parser):
        p.set_defaults(command="clickhouse")


def clickhouse_command(args) -> int:
    """Dispatch ClickHouse CLI commands.

    Called by the CLI framework when ``clickhouse`` subcommand is used.
    """
    cmd = getattr(args, "clickhouse_command", None)
    if cmd == "setup":
        return _cmd_setup(args)
    elif cmd == "status":
        return _cmd_status(args)
    elif cmd == "query":
        return _cmd_query(args)
    else:
        print("Usage: hermes clickhouse <setup|status|query>")
        print("  setup   — Configure connection and create schema")
        print("  status  — Check connection and table stats")
        print("  query   — Run raw SQL (admin)")
        return 1


def _load_provider() -> any:
    """Load ClickHouseMemoryProvider and initialize connection."""
    try:
        from plugins.memory.clickhouse import ClickHouseMemoryProvider
        provider = ClickHouseMemoryProvider()
        provider._load_config = lambda: provider._config
        provider.initialize(session_id="cli")
        return provider
    except Exception as e:
        print(f"Error: {e}")
        return None


def _cmd_setup(args) -> int:
    """Setup ClickHouse connection interactively."""
    print("=== ClickHouse Memory Provider Setup ===\n")

    host = args.host or input(f"Host [localhost]: ") or "localhost"
    port = args.port or int(input(f"Port [8123]: ") or "8123")
    user = args.user or input(f"User [default]: ") or "default"
    password = args.password or input("Password (leave blank for none): ")
    database = args.database or input(f"Database [hermes_memory]: ") or "hermes_memory"
    ttl_days = args.ttl_days or int(input(f"TTL days [30]: ") or "30")

    print(f"\nConnecting to {host}:{port} as {user}...")

    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=host, port=port, username=user,
            password=password, database=database,
            connect_timeout=10,
        )
        client.command("SELECT 1")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return 1

    # Save config
    from hermes_cli.config import load_config, save_config
    config = load_config()
    plugins = config.setdefault("plugins", {})
    plugins["clickhouse"] = {
        "host": host,
        "port": port,
        "user": user,
        "database": database,
        "ttl_days": ttl_days,
    }
    save_config(config)
    print("Config saved.")

    # Create schema
    print("Creating schema...")
    try:
        client.command(f"CREATE DATABASE IF NOT EXISTS {database}")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {database}.events (
            user_id          String,
            session_id       String,
            ts               DateTime,
            turn_number      UInt32,
            role             LowCardinality(String),
            content          String,
            topic            String,
            importance       Float32,
            frequency        UInt32,
            is_repeat        UInt8,
            related_topics   Array(String),
            emotion          String,
            platform         String,
            channel_id       String,
            model            String,
            parent_session_id String
        ) ENGINE = MergeTree()
        PARTITION BY (user_id, toYYYYMM(ts))
        ORDER BY (user_id, importance * frequency DESC, ts DESC)
        TTL ts + INTERVAL {ttl_days} DAY DELETE
        SETTINGS index_granularity = 8192
        """
        client.command(create_sql)
        print("Schema created.")
    except Exception as e:
        print(f"Schema creation failed: {e}")
        return 1

    print("\n✓ ClickHouse provider is ready!")
    print(f"  Database: {database}")
    print(f"  Table: events")
    print(f"  TTL: {ttl_days} days")
    print()
    print("Enable it in config.yaml:")
    print("  memory:")
    print("    provider: clickhouse")
    return 0


def _cmd_status(args) -> int:
    """Show ClickHouse connection status and stats."""
    provider = _load_provider()
    if not provider:
        return 1

    if args.json:
        stats = json.loads(provider._handle_memory_stats())
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print("=== ClickHouse Memory Provider: Status ===\n")
        if not provider._client:
            print("Status: ❌ Not connected")
            return 1

        print("Status: ✅ Connected")
        print(f"  Host: {provider._config.get('host', 'localhost')}")
        print(f"  Database: {provider._config.get('database', 'hermes_memory')}")

        stats = json.loads(provider._handle_memory_stats())
        error = stats.get("error")
        if error:
            print(f"  Stats error: {error}")
        else:
            print(f"  Total turns: {stats['total_turns']}")
            print(f"  Sessions: {stats['sessions']}")
            print(f"  Date range: {stats['first_turn']} → {stats['last_turn']}")
            print(f"  Avg importance: {stats['avg_importance']}")
            if stats['top_topics']:
                print(f"  Top topics:")
                for t in stats['top_topics'][:5]:
                    print(f"    - {t['topic']}: {t['count']}")

    return 0


def _cmd_query(args) -> int:
    """Run raw SQL query (admin)."""
    provider = _load_provider()
    if not provider or not provider._client:
        print("ClickHouse not connected.")
        return 1

    try:
        result = provider._client.query_df(args.sql)
        if result.empty:
            print("Query returned no rows.")
        else:
            print(result.to_string(index=False))
    except Exception as e:
        print(f"Query failed: {e}")
        return 1

    return 0

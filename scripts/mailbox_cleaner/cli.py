from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .clients import ReadAdapter
from .mcp_connector import McpReadOnlyAdapter, StdioJsonRpcSession, load_controlled_stdio_config
from .policy import AccountPolicy
from .runner import run_scheduled
from .state import StateStore


def _fixture_adapter(rows: list[dict[str, str]]) -> ReadAdapter:
    def read(operation: str, account: str) -> list[dict[str, str]]:
        if operation != "list_metadata":
            return []
        return [row for row in rows if row.get("account") == account]

    return read


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixture-safe mailbox read-only report")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path, help="local metadata JSON fixture")
    source.add_argument(
        "--mcp-stdio-config",
        type=Path,
        help="operator-supplied JSON stdio config; disabled unless explicitly allowed",
    )
    parser.add_argument(
        "--allow-controlled-read",
        action="store_true",
        help="acknowledge that the supplied MCP stdio config may reach configured mailboxes",
    )
    parser.add_argument("--state", type=Path, required=True, help="local private dedupe state")
    args = parser.parse_args(argv)
    if args.mcp_stdio_config is not None:
        if not args.allow_controlled_read:
            parser.error("--mcp-stdio-config requires --allow-controlled-read")
        config = load_controlled_stdio_config(args.mcp_stdio_config)
        connectors: dict[str, McpReadOnlyAdapter] = {}
        try:
            for provider, command in config.commands.items():
                connectors[provider] = McpReadOnlyAdapter(StdioJsonRpcSession(command))
        except Exception:
            for connector in connectors.values():
                connector.close()
            raise
        providers = {policy.account: policy.provider for policy in config.policies}

        def read(operation: str, account: str) -> list[dict[str, str]]:
            provider = providers[account]
            return connectors[provider](operation, account, provider=provider)

        try:
            report = run_scheduled(config.policies, read, StateStore(args.state))
        finally:
            for connector in connectors.values():
                connector.close()
        print(report.text)
        return 0
    if args.allow_controlled_read:
        parser.error("--allow-controlled-read requires --mcp-stdio-config")
    assert args.fixture is not None
    rows = json.loads(args.fixture.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        parser.error("fixture must be a JSON list of metadata objects")
    policies = [
        AccountPolicy(account=str(row["account"]), provider=str(row["provider"]))
        for row in rows
    ]
    unique_policies = list(dict.fromkeys(policies))
    report = run_scheduled(unique_policies, _fixture_adapter(rows), StateStore(args.state))
    print(report.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

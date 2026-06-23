"""CLI helpers for the Graphnosis memory provider."""

from __future__ import annotations


def graphnosis_command(args):
    sub = getattr(args, "graphnosis_command", None)
    if sub == "status":
        _cmd_status()
    elif sub == "test-recall":
        _cmd_test_recall(getattr(args, "query", "") or "preferences")
    else:
        print("Usage: hermes graphnosis <status|test-recall> [query]")


def _cmd_status() -> None:
    from pathlib import Path

    from plugins.memory.graphnosis.client import DEFAULT_SOCKET, GraphnosisMcpClient, GraphnosisMcpError

    path = Path(DEFAULT_SOCKET).expanduser()
    print(f"Socket path: {path}")
    if not path.exists():
        print("Status: unavailable — socket not found.")
        print("Install Graphnosis from https://graphnosis.com/download and unlock your cortex.")
        return
    try:
        client = GraphnosisMcpClient(str(path))
        client.connect()
        stats = client.call_tool("stats", {})
        client.close()
        print("Status: connected")
        preview = (stats or "").strip().splitlines()
        for line in preview[:8]:
            print(f"  {line}")
        if len(preview) > 8:
            print("  …")
    except GraphnosisMcpError as exc:
        print(f"Status: socket present but not accepting connections — {exc}")


def _cmd_test_recall(query: str) -> None:
    from plugins.memory.graphnosis.client import GraphnosisMcpClient, GraphnosisMcpError

    try:
        client = GraphnosisMcpClient()
        text = client.call_tool("recall", {"query": query, "maxTokens": 800, "maxNodes": 8})
        client.close()
        print(text or "(no results)")
    except GraphnosisMcpError as exc:
        print(f"Recall failed: {exc}")


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="graphnosis_command")
    subs.add_parser("status", help="Check Graphnosis socket and cortex connectivity")
    test = subs.add_parser("test-recall", help="Run a test recall query")
    test.add_argument("query", nargs="?", default="preferences", help="Recall query")
    subparser.set_defaults(func=graphnosis_command)

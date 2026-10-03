"""Smoke test for the ontology-context-layer MCP server (stdio transport).

Run: python scripts/test_mcp.py
Requires: pip install mcp  (the Hermes venv already has it)
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent / "ontology_mcp_server.py"


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            assert "ontology_ingest_entity" in names, "missing ingest tool"
            assert "ontology_validate" in names, "missing validate tool"
            print(f"[OK] discovered {len(names)} tools: {', '.join(names)}")

            r = await session.call_tool(
                "ontology_ingest_entity",
                {"type": "Company", "name": "GreenLeaf Landscaping",
                 "properties": {"annual_revenue": 85000}, "source": "test", "verified": False},
            )
            assert "created" in r.content[0].text, "ingest failed"
            print("[OK] ingested entity")

            r = await session.call_tool(
                "ontology_add_rule",
                {"name": "Prospect needs call",
                 "if_conditions": [{"property": "verified", "op": "eq", "value": False}],
                 "then": "follow_up_required"},
            )
            assert "added" in r.content[0].text, "rule add failed"
            print("[OK] added rule")

            r = await session.call_tool(
                "ontology_validate", {"entity_id": "greenleaf_landscaping"}
            )
            assert '"overall": "PASS"' in r.content[0].text, f"validate failed: {r.content[0].text}"
            print("[OK] validation PASS")

            r = await session.call_tool("ontology_stats", {})
            assert "entities" in r.content[0].text, "stats failed"
            print("[OK] stats")

            r = await session.call_tool("ontology_export", {})
            assert "entities" in r.content[0].text, "export failed"
            print("[OK] export")

            print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())

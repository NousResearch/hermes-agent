#!/usr/bin/env python3
"""Helper notes for the Hermes Zcash skill.

zcash-mcp is a stdio MCP server, not a direct `--tool` shell CLI. This helper
prints the MCP config and boundary reminders so agents wire the server through
their MCP client instead of trying to call tools as subprocess flags.
"""

from __future__ import annotations

import json
import sys


MCP_CONFIG = {
    "mcpServers": {
        "zcash": {
            "command": "npx",
            "args": ["@frontiercompute/zcash-mcp"],
        }
    }
}


BOUNDARY = {
    "in_scope": [
        "ZAP1 attestation leaves",
        "receipt templates",
        "Merkle proof bundles",
        "anchor status and anchor history",
        "receipt packet validation",
        "agent-eval and external-action receipt requests",
        "Zcash memo decoding",
        "public chain context for interpreting anchors",
    ],
    "out_of_scope": [
        "private key custody",
        "seed handling",
        "balance scanning",
        "PCZT signing",
        "shielded spend construction",
        "wallet synchronization",
        "broadcasting wallet transactions",
    ],
}


WORKFLOW = [
    "Call zcash_capability_manifest to confirm the boundary.",
    "Call zcash_receipt_template for the receipt type.",
    "Convert external evidence into bounded hashes.",
    "Call attest_event or a receipt-request builder.",
    "Check get_anchor_status before making finality claims.",
    "Fetch zap1_prove_receipt when the leaf is ready.",
    "Verify with verify_proof or the receipt verifier tools.",
]


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "config":
        print(json.dumps(MCP_CONFIG, indent=2))
        return 0
    if command == "boundary":
        print(json.dumps(BOUNDARY, indent=2))
        return 0
    if command == "workflow":
        print(json.dumps(WORKFLOW, indent=2))
        return 0

    print("Usage: zcash_client.py [config|boundary|workflow]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

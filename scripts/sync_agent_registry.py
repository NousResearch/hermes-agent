#!/usr/bin/env python3
"""Sync the legacy runtime agent registry from managed agents YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.managed_agents.runtime_mirror import write_runtime_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "configs" / "managed_agents" / "agents.yaml"
DEFAULT_OUTPUT = Path.home() / ".hermes" / "config" / "agent-registry.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    write_runtime_registry(args.source, args.output)
    print(f"synced {args.output} from {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke-test the local Hermes context-governor plugin without enabling it globally.

This script is intentionally non-mutating: it imports the plugin, runs one compact
call, exercises status/tools metadata, and prints a JSON receipt. It does not edit
~/.hermes/config.yaml and does not restart Hermes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.context_engine import discover_context_engines, load_context_engine  # noqa: E402


def main() -> int:
    os.environ.setdefault("HERMES_CONTEXT_GOVERNOR_BUDGET_MODE", "soft_warn")
    os.environ.setdefault("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", "false")
    discovered = discover_context_engines()
    engine = load_context_engine("context_governor")
    assert engine is not None
    engine.update_model("context-governor-live-trial", 100_000, provider="local-smoke")
    engine.on_session_start("context-governor-live-trial")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Build parser. Acceptance gate: cargo test must pass."},
        {"role": "assistant", "content": "Decision: use deterministic JSON parsing."},
        {"role": "tool", "content": ("bulk log\n" * 2000) + "error[E0425] in /src/lib.rs"},
        {"role": "user", "content": "Latest task: summarize what remains."},
    ]
    result = engine.compress(messages, current_tokens=20_000)
    if not result or result[-1].get("role") != "user":
        raise AssertionError("latest user message was not preserved as final message")
    if any(message.get("role") == "tool" for message in result):
        raise AssertionError("raw tool role leaked into compacted output")
    receipt = {
        "ok": True,
        "discovered": discovered,
        "engine": engine.name,
        "available": engine.is_available(),
        "tools": [schema["name"] for schema in engine.get_tool_schemas()],
        "input_messages": len(messages),
        "output_messages": len(result),
        "latest_user": result[-1].get("content"),
        "status": engine.get_status(),
    }
    print(json.dumps(receipt, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

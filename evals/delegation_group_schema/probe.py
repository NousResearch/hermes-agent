"""Offline same-config schema probe. Run in a fresh interpreter for each tree/policy.

python probe.py /path/to/tree /tmp/receipt.json [--independent]
Requires tiktoken; no model calls or model-quality claims.
"""

import json
import os
from pathlib import Path
import socket
import sys
import tempfile

root, destination = sys.argv[1:3]
sys.path.insert(0, root)
os.chdir(root)
with tempfile.TemporaryDirectory(prefix="delegate-schema-") as home:
    os.environ.clear()
    os.environ.update(HOME=home, HERMES_HOME=home, PATH="/usr/bin:/bin", HERMES_PLATFORM="cli")
    config = {"tools": {"tool_search": {"defer": []}}}
    if "--independent" in sys.argv:
        config["delegation"] = {"independent_completions": True}
    Path(home, "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    def deny_network(*args, **kwargs):
        raise RuntimeError("Offline schema probe forbids network access")

    socket.socket.connect = deny_network
    import tiktoken
    from model_tools import get_tool_definitions

    encoder = tiktoken.get_encoding("o200k_base")
    definitions = get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True)
    serialized = json.dumps(definitions, separators=(",", ":"))
    repeated = get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True)
    assert json.dumps(repeated, separators=(",", ":")) == serialized
    counts = {
        definition["function"]["name"]: len(encoder.encode(json.dumps(definition, separators=(",", ":"))))
        for definition in definitions
    }
    delegate = next(d for d in definitions if d["function"]["name"] == "delegate_task")
    receipt = {
        "config": config,
        "tokens": counts,
        "total_tokens": sum(counts.values()),
        "delegate": delegate,
        "repeated_assembly_byte_stable": True,
        "model_calls": 0,
    }
    Path(destination).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"destination": destination, "total_tokens": receipt["total_tokens"], "delegate_tokens": counts["delegate_task"]}))

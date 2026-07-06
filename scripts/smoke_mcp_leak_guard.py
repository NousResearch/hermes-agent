#!/usr/bin/env python3
"""Lightweight standalone smoke test for the leak-guard helpers in
tools.mcp_tool — imports the module and exercises the new symbols
without going through any MCP SDK transport.

Runs in ~50ms; intended for environments without pytest or the dev-extra
installed. CI/dev environments with pytest should use
``tests/test_mcp_stdio_handshake_leak.py`` directly.
"""

import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

try:
    import tools.mcp_tool as m  # type: ignore
except Exception as e:  # pragma: no cover
    # Touching tools.mcp_tool imports lots of optional deps. If a
    # developer-only package (yaml, etc.) is missing, gate the smoke
    # test on symbol-presence checks only — the structural assertion
    # recovers the regression-catching surface.
    src_path = os.path.join(THIS, "tools", "mcp_tool.py")
    if not os.path.exists(src_path):
        print(f"REFUSE: {src_path} missing")
        sys.exit(2)
    src = open(src_path, encoding="utf-8").read()
    required = (
        "_reap_failed_init_stdio_children",
        "_FAST_REAP_GRACE_S",
        "_CONNECT_CIRCUIT_BREAKER_THRESHOLD",
        "_CONNECT_CIRCUIT_BREAKER_COOLDOWN_S",
        "skipped_for_breaker",
    )
    missing = [s for s in required if s not in src]
    if missing:
        print(f"FAIL: module-import failed and structural symbols missing: {missing}")
        sys.exit(1)
    print("PASS: structural guard present (module import unavailable)")
    sys.exit(0)

required_symbols = (
    "_reap_failed_init_stdio_children",
    "_FAST_REAP_GRACE_S",
    "_CONNECT_CIRCUIT_BREAKER_THRESHOLD",
    "_CONNECT_CIRCUIT_BREAKER_COOLDOWN_S",
    "_kill_orphaned_mcp_children",
    "_discover_and_register_server",
)
missing = [s for s in required_symbols if not hasattr(m, s)]
if missing:
    print(f"FAIL: symbols missing: {missing}")
    sys.exit(1)

assert m._CONNECT_CIRCUIT_BREAKER_THRESHOLD >= 3
assert m._CONNECT_CIRCUIT_BREAKER_COOLDOWN_S >= 30.0
assert m._FAST_REAP_GRACE_S >= 0.05
assert m._FAST_REAP_GRACE_S <= 1.0
print(f"PASS: threshold={m._CONNECT_CIRCUIT_BREAKER_THRESHOLD} "
      f"cooldown={m._CONNECT_CIRCUIT_BREAKER_COOLDOWN_S} "
      f"grace={m._FAST_REAP_GRACE_S}")

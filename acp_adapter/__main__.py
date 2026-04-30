"""Allow running the ACP adapter as ``python -m acp_adapter``."""

import os

# ACP mode owns stdout for JSON-RPC and stderr for logs; the controlling editor
# draws its own chrome. Suppress OSC-based tab-title emission from any code path
# that would otherwise write escape sequences to the TTY.
os.environ.setdefault("HERMES_DISABLE_TAB_TITLE", "1")

from .entry import main

main()

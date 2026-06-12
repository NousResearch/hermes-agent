#!/usr/bin/env python3
"""Obsolete Discord bot-to-bot routing smoke test.

Discord bot-to-bot control routing has been decommissioned. The gateway now
ignores Discord messages authored by bots regardless of legacy environment
variables, and the BOT_MSG v1 send/approval tools are not registered.

This script is intentionally kept as a non-destructive tombstone instead of
being deleted, so old references fail clearly.
"""

from __future__ import annotations

import sys


DECOMMISSIONED_MESSAGE = (
    "discord_bot_matrix_smoke.py is obsolete: Discord bot-to-bot control "
    "routing is decommissioned; bot-authored Discord messages are ignored."
)


def main() -> int:
    print(DECOMMISSIONED_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

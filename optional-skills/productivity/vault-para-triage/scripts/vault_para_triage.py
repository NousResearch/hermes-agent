#!/usr/bin/env python3
"""Thin wrapper for the bundled PARA triage helper.

When cron runs this file directly it passes no arguments, so default to the
nightly Slack-friendly triage run. Manual invocations still accept the full CLI.
"""

from __future__ import annotations

import sys

from hermes_cli.vault_para_triage import main


if __name__ == "__main__":
    argv = sys.argv[1:] or ["run", "--format", "slack"]
    raise SystemExit(main(argv))

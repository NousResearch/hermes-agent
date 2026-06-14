"""hermes-cli cron wrapper: 5h ollama-cloud failback check.

Contract (set by the user 2026-06-14):
    - Run every 5 hours, starting now.
    - No output shall be given until backup API key usage was detected
      AND the failback is triggered.

The work is done in ``hermes_cli.failback.run()`` which returns a dict
whose ``action`` field is one of:
    - ``"failback_triggered"``  — primary was reset, we rotated to it.
    - ``"no_failback_needed"`` — already on primary, or primary still
                                  in cooldown.  Silent in cron context.
    - ``"error"``               — something went wrong.  Emit a one-line
                                  warning so the operator sees the
                                  failure (silent cron + broken
                                  watchdog = the worst failure mode).

Exit codes:
    0 — success (including "no_failback_needed")
    1 — error (the operator will see the JSON we emitted to stdout)

This script is registered as a ``--no-agent`` cron job by
``kpi_test_ollama_failover.py`` and runs at "every 5h" under the name
``ollama-failback``.  The cron CLI expects scripts to live under
``~/.hermes/scripts/`` (the user's AppData symlinked to D:\\hermes-app
on this Windows install); the canonical copy at scripts/failback.py
is the versioned source of truth and should be copied to the AppData
location on every deploy.
"""

from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    try:
        from hermes_cli.failback import run
    except Exception as exc:
        print(json.dumps({
            "action": "error",
            "error": f"failed to import hermes_cli.failback: {exc}",
            "traceback": traceback.format_exc(),
        }))
        return 1

    try:
        result = run(provider="ollama-cloud")
    except Exception as exc:
        print(json.dumps({
            "action": "error",
            "error": f"failback.run() raised: {exc}",
            "traceback": traceback.format_exc(),
        }))
        return 1

    action = result.get("action")
    if action == "failback_triggered":
        print(json.dumps(result))
        return 0
    if action == "error":
        print(json.dumps(result))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

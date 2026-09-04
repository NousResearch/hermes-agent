"""Strict JSON bridge for the Desktop's pre-teardown backup policy."""

from __future__ import annotations

import json
import sys

from hermes_cli.update_backup_policy import resolve_pre_update_backup_policy_strict


def main() -> int:
    try:
        policy = resolve_pre_update_backup_policy_strict()
    except Exception as exc:
        print(
            f"pre-update policy resolution failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(policy, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.email_hygiene import apply_hygiene_action


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply one approved Torben email hygiene action.")
    parser.add_argument("--handle", required=True, help="Action handle, e.g. EA-20260625-001")
    parser.add_argument("--approved-by", default="signal", help="Approval source for audit history")
    parser.add_argument("--dry-run", action="store_true", help="Validate without mutating Gmail")
    args = parser.parse_args()

    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    result = apply_hygiene_action(
        ledger=ActionLedger(state_dir / "torben-action-ledger.json"),
        config_path=home / "config" / "google_accounts.yaml",
        handle=args.handle,
        approved_by=args.approved_by,
        dry_run=args.dry_run,
    )
    output_path = state_dir / f"torben-email-hygiene-apply-{args.handle}.json"
    tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.signal_coo.gmail_realtime import (
    DEFAULT_SUBSCRIPTION_NAME,
    run_gmail_realtime_canary,
    write_json,
)

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    home = Path(os.getenv("HERMES_HOME") or DEFAULT_TORBEN_HOME)
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = run_gmail_realtime_canary(
        config_path=home / "config" / "google_accounts.yaml",
        relationship_context_path=home / "config" / "relationship_context.yaml",
        state_path=state_dir / "torben-gmail-watch-state.json",
        account_alias=os.getenv("TORBEN_GMAIL_CANARY_ACCOUNT") or None,
        subscription_name=os.getenv("TORBEN_GMAIL_PUBSUB_SUBSCRIPTION", DEFAULT_SUBSCRIPTION_NAME),
        timeout_seconds=int(os.getenv("TORBEN_GMAIL_CANARY_TIMEOUT_SECONDS", "75")),
        poll_interval_seconds=int(os.getenv("TORBEN_GMAIL_CANARY_POLL_SECONDS", "5")),
        cleanup=not _truthy(os.getenv("TORBEN_GMAIL_CANARY_SKIP_CLEANUP")),
    )
    write_json(state_dir / "torben-gmail-realtime-canary-latest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

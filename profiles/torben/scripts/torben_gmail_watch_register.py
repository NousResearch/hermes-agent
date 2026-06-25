from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.gmail_realtime import DEFAULT_TOPIC_NAME, register_gmail_watches, write_json

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")


def _torben_home() -> Path:
    if os.getenv("HERMES_HOME"):
        return get_hermes_home()
    return DEFAULT_TORBEN_HOME


def main() -> int:
    home = _torben_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = home / "config" / "google_accounts.yaml"
    state_path = state_dir / "torben-gmail-watch-state.json"
    output_path = state_dir / "torben-gmail-watch-register-latest.json"
    topic_name = os.getenv("TORBEN_GMAIL_WATCH_TOPIC", DEFAULT_TOPIC_NAME)

    payload = register_gmail_watches(
        config_path=config_path,
        state_path=state_path,
        topic_name=topic_name,
    )
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("status") != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.signal_coo.gmail_realtime import write_json
from hermes_cli.signal_coo.live_profile_verify import consume_live_profile_investigation_request

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")


def main() -> int:
    profile_home = Path(os.getenv("TORBEN_PROFILE_HOME") or DEFAULT_TORBEN_HOME)
    state_dir = profile_home / "state"
    request_path = state_dir / "torben-live-profile-investigation-request.json"
    handoff_state_path = state_dir / "torben-live-profile-investigation-state.json"
    latest_path = state_dir / "torben-live-profile-investigation-latest.json"

    payload = consume_live_profile_investigation_request(
        request_path=request_path,
        state_path=handoff_state_path,
    )
    write_json(latest_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.signal_coo.gmail_realtime import write_json
from hermes_cli.signal_coo.live_profile_verify import render_verification_failure, verify_torben_live_profile

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")
DEFAULT_REPO_SNAPSHOT_HOME = Path("/Users/ericfreeman/.hermes/hermes-agent/profiles/torben")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    profile_home = Path(os.getenv("TORBEN_PROFILE_HOME") or DEFAULT_TORBEN_HOME)
    repo_snapshot_home = Path(os.getenv("TORBEN_REPO_PROFILE_SNAPSHOT_HOME") or DEFAULT_REPO_SNAPSHOT_HOME)
    output_path = profile_home / "state" / "torben-live-profile-verify-latest.json"
    payload = verify_torben_live_profile(
        profile_home=profile_home,
        repo_snapshot_home=repo_snapshot_home,
        check_snapshot_sync=not _truthy(os.getenv("TORBEN_VERIFY_SKIP_SNAPSHOT_SYNC")),
    )
    write_json(output_path, payload)
    if payload.get("status") == "pass":
        print(json.dumps({"wakeAgent": False, "status": "pass", "task": "torben_live_profile_verify"}))
        return 0
    print(render_verification_failure(payload), end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

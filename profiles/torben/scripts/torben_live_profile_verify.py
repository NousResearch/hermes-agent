from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.signal_coo.gmail_realtime import write_json
from hermes_cli.signal_coo.live_profile_verify import (
    clear_live_profile_investigation_request,
    render_verification_failure,
    stage_live_profile_investigation_request,
    update_live_profile_alert_state,
    verify_torben_live_profile,
)

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")
DEFAULT_REPO_SNAPSHOT_HOME = Path("/Users/ericfreeman/.hermes/hermes-agent/profiles/torben")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    profile_home = Path(os.getenv("TORBEN_PROFILE_HOME") or DEFAULT_TORBEN_HOME)
    repo_snapshot_home = Path(os.getenv("TORBEN_REPO_PROFILE_SNAPSHOT_HOME") or DEFAULT_REPO_SNAPSHOT_HOME)
    output_path = profile_home / "state" / "torben-live-profile-verify-latest.json"
    alert_state_path = profile_home / "state" / "torben-live-profile-verify-alert-state.json"
    investigation_request_path = profile_home / "state" / "torben-live-profile-investigation-request.json"
    payload = verify_torben_live_profile(
        profile_home=profile_home,
        repo_snapshot_home=repo_snapshot_home,
        check_snapshot_sync=not _truthy(os.getenv("TORBEN_VERIFY_SKIP_SNAPSHOT_SYNC")),
    )
    if payload.get("status") == "pass":
        update_live_profile_alert_state(payload=payload, state_path=alert_state_path)
        clear_live_profile_investigation_request(
            payload=payload,
            request_path=investigation_request_path,
        )
        write_json(output_path, payload)
        print(json.dumps({"wakeAgent": False, "status": "pass", "task": "torben_live_profile_verify"}))
        return 0
    duplicate_suppressed = update_live_profile_alert_state(payload=payload, state_path=alert_state_path)
    if not duplicate_suppressed:
        stage_live_profile_investigation_request(
            payload=payload,
            request_path=investigation_request_path,
        )
    write_json(output_path, payload)
    if duplicate_suppressed:
        print(
            json.dumps(
                {
                    "wakeAgent": False,
                    "status": "duplicate_failure_suppressed",
                    "task": "torben_live_profile_verify",
                    "fingerprint": (payload.get("alert_dedupe") or {}).get("fingerprint"),
                }
            )
        )
        return 0
    print(render_verification_failure(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

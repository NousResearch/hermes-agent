from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.signal_coo.gmail_realtime import write_json
from hermes_cli.signal_coo.live_profile_verify import consume_live_profile_investigation_request

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")


LIVE_PROFILE_TOOL_POLICY = {
    "blocked_tools": ["execute_code"],
    "allowed_tools": ["read_file", "terminal"],
    "terminal_rules": [
        "Read-only commands only; no edits, restarts, file deletes, network mutations, email/calendar actions, posts, or trades.",
        "Use /opt/homebrew/bin/uv run python from /Users/ericfreeman/.hermes/hermes-agent when Hermes imports are needed.",
        "Use /opt/homebrew/bin/python3 for standard-library one-liners; do not call bare python or rely on cron PATH.",
    ],
    "agent_instruction": (
        "Never call execute_code for this cron job. Cron cannot approve execute_code, and blocked attempts create alert noise. "
        "Use read_file for file inspection and terminal for bounded read-only checks only."
    ),
}


def attach_live_profile_tool_policy(payload: dict) -> dict:
    enriched = dict(payload)
    enriched["tool_policy"] = LIVE_PROFILE_TOOL_POLICY
    contracts = dict(enriched.get("contracts") or {})
    contracts["tool_policy"] = LIVE_PROFILE_TOOL_POLICY["agent_instruction"]
    enriched["contracts"] = contracts
    return enriched


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
    payload = attach_live_profile_tool_policy(payload)
    write_json(latest_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-live-profile-investigate", main))

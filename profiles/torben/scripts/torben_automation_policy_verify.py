#!/usr/bin/env python3
"""Verify Torben automation policy and write the latest policy artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.signal_coo.automation_policy import write_automation_policy_artifact

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")


def main() -> int:
    profile_home = Path(os.getenv("TORBEN_PROFILE_HOME") or os.getenv("HERMES_HOME") or DEFAULT_TORBEN_HOME)
    artifact = write_automation_policy_artifact(profile_home=profile_home)
    print(
        json.dumps(
            {
                "wakeAgent": artifact.get("status") != "pass",
                "status": artifact.get("status"),
                "task": "torben_automation_policy_verify",
                "artifact": str(profile_home / "state" / "torben-automation-policy-latest.json"),
            }
        )
    )
    return 0 if artifact.get("status") == "pass" else 1


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-automation-policy-verify", main))

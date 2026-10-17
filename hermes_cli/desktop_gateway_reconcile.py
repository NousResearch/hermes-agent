"""Read-only topology/receipt evidence for the Windows update handoff.

No boot preflight, config writes, process termination or service-manager calls.
The caller still verifies runtime identity, PID fingerprint and heartbeat.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import re

from hermes_constants import get_default_hermes_root, get_hermes_home, profile_name_for_home
from hermes_cli.profiles import profile_is_standalone, profiles_to_serve


def evidence(correlation: str, started_at: float) -> dict:
    home = get_hermes_home()
    state = json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))
    served = state.get("served_profiles")
    if not isinstance(served, list) or any(not isinstance(name, str) for name in served):
        raise ValueError("invalid served-profile record")
    launch = profile_name_for_home(home) or "default"
    required = [name for name, _ in profiles_to_serve(True)]
    standalone = launch != "default" and profile_is_standalone(home)
    if standalone:
        required = [launch]
    if not served and (standalone or required == [launch]):
        missing = []  # The real standalone producer publishes [], not [launch].
    else:
        missing = [name for name in required if name not in served]
    if standalone and served:
        missing.append("standalone launch has a host served-profile record")
    directory = get_default_hermes_root() / "logs" / "update_receipts"
    # Archives are rooted even for named launches; latest.json is only a mirror
    # and can have been replaced by another run. Filter EACH archive before ranking.
    candidates = []
    now = datetime.now(UTC).timestamp()
    for path in directory.glob("update_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            sha = record.get("post_update", {}).get("sha", "")
            if (not correlation or record.get("correlation_id") != correlation
                    or record.get("outcome") not in ("success", "partial", "interrupted")
                    or not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha)):
                continue
            finished = datetime.fromisoformat(record["finished_at"])
            if finished.tzinfo is None:
                continue
            stamp = finished.timestamp()
            if started_at - 5 <= stamp <= now + 5:
                candidates.append((stamp, record))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue  # A torn/unrelated archive cannot hide this run's evidence.
    receipt = max(candidates, key=lambda item: item[0])[1] if candidates else {}
    sha = receipt.get("post_update", {}).get("sha", "")
    covered = (set(served) & set(required)) if served else ({launch} if not missing else set())
    # A host's served set does not prove an update-owned standalone sibling
    # (or a service/serve unit) moved to new code. Keep those obligations manual.
    restart = receipt.get("gateway_restart") or {}
    recovery = restart.get("fresh_recovery") or {}
    intended = set()
    opaque_runtime = False
    # Successful restart lists are not the ownership inventory: a failed
    # sibling may never enter them. Preserve the original plan and outcomes.
    for runtimes in ((receipt.get("plan") or {}).get("runtimes", []),
                     receipt.get("runtime_outcomes", [])):
        if not isinstance(runtimes, list):
            raise ValueError("invalid update-owned runtime inventory")
        for runtime in runtimes:
            if not isinstance(runtime, dict):
                raise ValueError("invalid update-owned runtime record")
            profile = runtime.get("profile")
            if runtime.get("kind") == "gateway" and isinstance(profile, str) and profile:
                intended.add(profile)
            else:
                opaque_runtime = True
    for names in (restart.get("relaunched_profiles", []),
                  restart.get("externally_supervised_profiles", []),
                  recovery.get("requested", [])):
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("invalid update-owned profile inventory")
        intended.update(names)
    missing = sorted(set(missing) | (intended - covered))
    if (opaque_runtime or (restart.get("incomplete") and not intended)
            or restart.get("restarted_services") or restart.get("failed_units")
            or recovery.get("serve_units", {}).get("failed")
            or recovery.get("stale_runtimes")):
        missing.append("unverified update-owned service/runtime")
    return {"expected_sha": sha, "missing_profiles": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlation", required=True)
    parser.add_argument("--started-at", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(evidence(args.correlation, args.started_at)))


if __name__ == "__main__":
    main()

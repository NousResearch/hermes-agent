"""Disposable hermetic A/B: pending-restart marker expiry + superseded completed receipt.

Run with the checkout path as ``argv[1]``:

    venv/bin/python evals/update_pending_marker_expiry.py /path/to/hermes-agent

No services, no network, no live fleet: ``collect_fleet_versions`` is stubbed so the
reconciliation sees a controlled fleet, and ``HOME``/``HERMES_HOME`` point at a temp dir
so no live marker or receipt is read or written. The module path is asserted to live under
``argv[1]`` so the A/B cannot silently load the wrong checkout through the editable install.

Exit 1 while either sticky source still fires (base), exit 0 when both settle.
"""

import json
import os
import pathlib
import sys
import tempfile

repo = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
root = pathlib.Path(tempfile.mkdtemp(prefix="marker-expiry-"))
os.environ["HOME"] = str(root)
os.environ["HERMES_HOME"] = str(root / ".hermes")

from hermes_cli import update_cmd_fleet as fleet, update_receipt as receipts  # noqa: E402
import hermes_cli.update_cmd as update_cmd  # noqa: E402

print("MODULE", fleet.__file__)
assert pathlib.Path(fleet.__file__).resolve().is_relative_to(repo), "wrong checkout imported"

# Synthetic code identity: the harness must not depend on a git checkout, and both
# bindings have to move together (the code imports the name from ``update_cmd`` at call
# time, while ``update_cmd`` bound the original object at import time).
sha = "b" * 40
old = "7" * 40
fleet._current_checkout_sha = lambda: sha
update_cmd._current_checkout_sha = lambda: sha
marker = fleet._fleet_restart_pending_marker_path()
live = [{"profile": "default", "pid": 823, "code_sha": sha, "state": "current"}]
receipts.collect_fleet_versions = lambda *a, **k: list(live)

receipt_path = root / ".hermes" / "logs" / "update_receipts" / "latest.json"
receipt_path.parent.mkdir(parents=True, exist_ok=True)


def write_receipt(payload):
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")


results = []


def check(name, got, want):
    results.append((name, got, want))
    print(f"{'ok  ' if got is want else 'FAIL'} {name}: got {got!r} want {want!r}")


# 1. A completed update's receipt records the fleet current AT THE TIME; a later pull
#    moves the checkout past that snapshot. Nothing is owed (the live fleet is current).
write_receipt(
    {
        "outcome": "success",
        "exit_code": 0,
        "post_update": {"sha": old},
        "fleet": [{"profile": "default", "pid": 48579, "code_sha": old, "state": "current"}],
        "plan": {
            "expected_sha": old,
            "runtimes": [
                {"kind": "gateway", "profile": "default", "pid": 1, "code_sha": old},
                # Unvouchable for the gateway matrix, so _live_fleet_covers_receipt cannot
                # rescue the receipt: only the unfinished gate can.
                {"kind": "serve", "profile": "default", "pid": 2, "code_sha": None},
            ],
        },
        "gateway_restart": {"incomplete": False},
    }
)
check("completed-receipt-superseded", fleet._pending_fleet_restart_needed(), False)

# 2. Marker from the pull that produced the code now on disk, and every gateway the
#    marker recorded is live on it: the obligation is discharged (reboot/restart).
fleet._write_fleet_restart_pending_marker(expected_sha=sha)
body = marker.read_text(encoding="utf-8")
check("marker-records-gateway-identities", "gateway_profiles=default" in body, True)
check("marker-discharged-by-current-fleet", fleet._pending_fleet_restart_needed(), False)
check("discharged-marker-expired", marker.is_file(), False)

# 3. Negative control: the same marker with the gateway still on pre-pull code stays
#    pending, and keeps its breadcrumb for the catch-up restart.
fleet._write_fleet_restart_pending_marker(expected_sha=sha)
live[:] = [{"profile": "default", "pid": 823, "code_sha": old, "state": "stale"}]
check("marker-pending-on-stale-gateway", fleet._pending_fleet_restart_needed(), True)
check("stale-marker-kept", marker.is_file(), True)

# 4. Negative control: an UNFINISHED receipt still owes the restart (the gate must not
#    make unfinished updates quiet).
fleet._clear_fleet_restart_pending_marker()
write_receipt(
    {
        "outcome": "failed",
        "exit_code": 1,
        "stop_reason": "KeyboardInterrupt: ",
        "plan": {"runtimes": [{"kind": "gateway", "profile": "default", "code_sha": old}]},
    }
)
check("unfinished-receipt-still-pending", fleet._pending_fleet_restart_needed(), True)

failed = [name for name, got, want in results if got is not want]
print("VERDICT:", "FIXED" if not failed else "REPRODUCED " + ",".join(failed))
sys.exit(1 if failed else 0)

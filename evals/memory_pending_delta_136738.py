"""Drive issue #136738's approval surfaces against a source tree and report what they say.

    python evals/memory_pending_delta_136738.py [source_tree]     # default: this checkout

The issue: `/memory pending` names the entry a staged replace/remove is pinned to, but says
nothing about what the payload does to it — while memory's whole-entry contract means a
clause-level payload deletes every sibling fact it did not re-mention. So a destructive
proposal reads like a small edit in the list a human is approving, and the size of the loss
only shows up afterwards, as a smaller MEMORY.md.

This stages a destructive whole-entry replace plus a remove through the background review's own
staging path (unattended review stages instead of applying), then prints both surfaces verbatim:
`/memory pending` and `/memory approve all`. Exit 0 when the pending view states the char delta
of each staged entry, 1 when it does not (the state before the fix). Nothing is written outside
the scratch HERMES_HOME this script creates.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# One entry packing several unrelated facts — the shape the issue's incident ran into, where the
# payload rewrites the first clause and drops the rest.
ENTRY = ("laptop drop-off = DELIBERATE KDE AC auto-suspend 1h (not a fault), no WoL; "
         "wifi = WPA3 only on 5GHz, 2.4GHz band disabled at the AP; "
         "backup = nightly restic to nas-2 at 03:30, retention 14d; "
         "printer = cups queue hp-m428, duplex default")
NEW_CLAUSE = "drop-off = deliberate KDE AC suspend 1h, no WoL"
TAILNET = "tailnet = 100.64.0.0/10, exit node = nas-2"


def main(argv):
    tree = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(tree))
    os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="eval-136738-")
    os.environ.pop("HERMES_CRON_SESSION", None)

    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool
    from tools.skill_provenance import (reset_current_write_origin, reset_review_attended,
                                        set_current_write_origin, set_review_attended)

    store = MemoryStore()
    store.load_from_disk()
    assert store.add("memory", ENTRY)["success"], "could not seed the multi-fact entry"
    assert store.add("memory", TAILNET)["success"]

    operations = [
        {"action": "replace", "old_text": "drop-off = DELIBERATE KDE", "content": NEW_CLAUSE},
        {"action": "add", "content": "nas-2 = restic target + tailscale exit node"},
        {"action": "remove", "old_text": "tailnet = 100.64.0.0/10"},
    ]
    origin, attended = set_current_write_origin("background_review"), set_review_attended(False)
    try:
        staged = json.loads(memory_tool(target="memory", store=store, operations=operations))
    finally:
        reset_review_attended(attended)
        reset_current_write_origin(origin)
    print(f"staged by the review fork: {staged.get('staged')} "
          f"(pending id {staged.get('pending_id')})")

    pending = handle_pending_subcommand(wa.MEMORY, ["pending"]) or ""
    print("\n--- /memory pending ---")
    print(pending)
    print("--- /memory approve all ---")
    print(handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store))

    delta = f"{len(ENTRY)} -> {len(NEW_CLAUSE)} chars, {len(NEW_CLAUSE) - len(ENTRY):+d}"
    remove_delta = f"{len(TAILNET)} -> 0 chars, {-len(TAILNET):+d}"
    checks = [
        (f"pending view states the replace delta ({delta})", delta in pending),
        ("pending view flags the gutted entry", "shrinks it below half" in pending),
        (f"pending view states the remove delta ({remove_delta})", remove_delta in pending),
        ("pending view still shows the FULL pinned entry", ENTRY in pending),
        ("the unquantified shape is gone", "replaces entry: laptop drop-off" not in pending),
    ]
    print("\n--- checks ---")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [label for label, ok in checks if not ok]
    print(f"\n{'PASS' if not failed else 'FAIL'}: {len(checks) - len(failed)}/{len(checks)} checks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

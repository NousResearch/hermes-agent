"""Drive issue #136738's approval surfaces against a source tree and report what they say.

    python evals/memory_pending_delta_136738.py [source_tree]     # default: this checkout

Current Hermes releases use span-preserving ``patch`` instead of whole-entry ``replace``, but the
approval queue still needs to state what each patch or removal will do before a human approves it.
Older pinned replace records remain supported during upgrade and are covered by the regression
suite; this eval drives the current background-review staging path end to end.

The scratch HERMES_HOME is a ``TemporaryDirectory``, so nothing the run writes outlives it. The
run also checks that the directory is gone before it reports success.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ENTRY = ("laptop drop-off = DELIBERATE KDE AC auto-suspend 1h (not a fault), no WoL; "
         "wifi = WPA3 only on 5GHz, 2.4GHz band disabled at the AP; "
         "backup = nightly restic to nas-2 at 03:30, retention 14d; "
         "printer = cups queue hp-m428, duplex default")
OLD_CLAUSE = "laptop drop-off = DELIBERATE KDE AC auto-suspend 1h (not a fault), no WoL"
NEW_CLAUSE = "laptop drop-off = deliberate KDE AC suspend 1h, no WoL"
TAILNET = "tailnet = 100.64.0.0/10, exit node = nas-2"


def _drive(tree):
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
        {"action": "patch", "old_text": OLD_CLAUSE, "content": NEW_CLAUSE},
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
    approval = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store) or ""
    print("--- /memory approve all ---")
    print(approval)

    patched_entry = ENTRY.replace(OLD_CLAUSE, NEW_CLAUSE, 1)
    patch_delta = f"{len(ENTRY)} -> {len(patched_entry)} chars, {len(patched_entry) - len(ENTRY):+d}"
    remove_delta = f"{len(TAILNET)} -> 0 chars, {-len(TAILNET):+d}"
    checks = [
        (f"pending view states the patch delta ({patch_delta})", patch_delta in pending),
        (f"pending view states the remove delta ({remove_delta})", remove_delta in pending),
        ("pending view names the patch correctly", "patches entry:" in pending and "patchs entry:" not in pending),
        ("pending view still shows the full pinned patch entry", ENTRY in pending),
        ("pending view still shows the full pinned removed entry", TAILNET in pending),
        ("approval preserved sibling facts while applying the patch",
         patched_entry in store.memory_entries and TAILNET not in store.memory_entries),
    ]
    print("\n--- checks ---")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [label for label, ok in checks if not ok]
    print(f"\n{'PASS' if not failed else 'FAIL'}: {len(checks) - len(failed)}/{len(checks)} checks")
    return 1 if failed else 0


def main(argv):
    tree = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(tree))
    with tempfile.TemporaryDirectory(prefix="eval-136738-") as home:
        os.environ["HERMES_HOME"] = home
        os.environ.pop("HERMES_CRON_SESSION", None)
        code = _drive(tree)
    if os.path.exists(home):
        print(f"\nFAIL: the scratch HERMES_HOME was left behind at {home}")
        return 1
    print(f"\nscratch HERMES_HOME removed: {home}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))

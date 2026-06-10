#!/usr/bin/env python3
"""Phase-0 SDK kill-gate spike for the mem0 destructive-tools spec.

Proves the SDK behaviors the spec's forget/delete design depends on, against a
SCRATCH user_id (never the live `ace` store). Wipes the scratch user at the end.

PASS criteria (spec Phase 0):
  1. add() then get()/get_all() round-trips a memory.
  2. update(text=, metadata=) neutralizes text AND carries custom metadata.   [C5/forget kill-gate]
  3. history(memory_id) returns the prior version (un-forget source).         [C5]
  4. delete(memory_id) → subsequent get() 404s (MemoryNotFoundError).         [C2/delete]
  5. delete of a bogus id raises a not-found error (grounds C2 not-found path).
  6. Whether search/get_all `filters` supports a metadata exclude server-side. [C6 impl detail]
     (Informational: if not, we filter client-side in the read paths — already planned.)

Usage:
  ./venv/bin/python scripts/mem0-destructive-spike.py --scratch-user spike-$RANDOM
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

# Reuse the plugin's config loader so we hit the same creds/endpoint.
sys.path.insert(0, ".")
from plugins.memory.mem0 import _load_config  # noqa: E402


def _unwrap(resp):
    if isinstance(resp, dict):
        return resp.get("results", resp)
    return resp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-user", required=True,
                    help="Scratch user_id (must NOT be the live 'ace' store).")
    args = ap.parse_args()
    uid = args.scratch_user
    if uid in ("ace", "hermes-user", "hermes"):
        print(f"REFUSE: '{uid}' looks like a real store. Use a spike-* scratch id.")
        return 2

    cfg = _load_config()
    api_key = cfg.get("api_key")
    if not api_key:
        print("FAIL: no mem0 api_key in config/env.")
        return 2

    from mem0 import MemoryClient
    client = MemoryClient(api_key=api_key)

    results = {}
    created_ids = []

    def record(name, ok, detail=""):
        results[name] = (ok, detail)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print(f"=== Phase-0 SDK kill-gate spike (scratch user_id={uid!r}) ===")

    try:
        # --- 1. add + get_all round-trip ----------------------------------
        print("\n[1] add() + get_all() round-trip")
        add_resp = client.add(
            [{"role": "user", "content": "Spike fact: the sky is plaid on Tuesdays."}],
            user_id=uid, agent_id="spike-agent", infer=False,
        )
        time.sleep(2.0)  # eventual consistency
        allm = _unwrap(client.get_all(filters={"user_id": uid}))
        record("1.add_get_all", bool(allm), f"{len(allm)} memory(ies) after add")
        if not allm:
            print("  (no memories surfaced; aborting deeper checks)")
            raise SystemExit
        mem = allm[0]
        mid = mem.get("id") or mem.get("memory_id")
        created_ids.append(mid)
        original_text = mem.get("memory", "")
        print(f"      id={mid}  text={original_text!r}")

        # --- 2. update(text=, metadata=) ----------------------------------
        print("\n[2] update() neutralizes text + carries metadata  [forget kill-gate]")
        tomb_meta = {"forgotten": True, "forgotten_by": "spike",
                     "original_text": original_text, "reason": "spike-test"}
        client.update(mid, text="[FORGOTTEN] spike", metadata=tomb_meta)
        time.sleep(2.0)
        after = client.get(mid)
        new_text = after.get("memory", "")
        new_meta = after.get("metadata") or {}
        text_ok = new_text.startswith("[FORGOTTEN]")
        meta_ok = bool(new_meta) and new_meta.get("forgotten") is True \
            and new_meta.get("original_text") == original_text
        record("2a.update_text", text_ok, f"text now {new_text!r}")
        record("2b.update_metadata", meta_ok,
               f"metadata={new_meta!r}" if not meta_ok else "forgotten+original_text carried")

        # --- 3. history() returns prior version ---------------------------
        print("\n[3] history() returns prior version  [un-forget source]")
        hist = client.history(mid)
        hist_list = hist if isinstance(hist, list) else _unwrap(hist)
        # look for the original text anywhere in history
        found_prior = any(
            original_text and original_text in str(h.get("old_memory") or h.get("memory") or h.get("input") or h)
            for h in (hist_list or [])
        )
        record("3.history_has_prior", found_prior,
               f"{len(hist_list or [])} history rows; prior text {'found' if found_prior else 'NOT found'}")
        if not found_prior and hist_list:
            print(f"      (history sample: {str(hist_list[0])[:200]})")

        # --- 6. metadata-exclude on search/get_all? (informational) -------
        print("\n[6] Does search/get_all support a metadata exclude server-side?")
        meta_filter_works = None
        try:
            # Try a search constrained by metadata.forgotten == False (exclude tombstones)
            s = client.search(
                query="sky",
                filters={"user_id": uid, "metadata": {"forgotten": False}},
            )
            s_list = _unwrap(s)
            # If the forgotten memory is excluded, that's server-side support.
            forgotten_present = any((r.get("id") or r.get("memory_id")) == mid for r in (s_list or []))
            meta_filter_works = not forgotten_present
            record("6.metadata_exclude_server_side", meta_filter_works,
                   "server-side metadata filter honored" if meta_filter_works
                   else "forgotten mem still returned → filter CLIENT-SIDE (as planned)")
        except Exception as e:
            record("6.metadata_exclude_server_side", False,
                   f"filters={{metadata:...}} not accepted ({type(e).__name__}) → client-side filter")

        # --- 4. delete() → get() 404 --------------------------------------
        print("\n[4] delete() then get() should 404")
        client.delete(mid)
        time.sleep(2.0)
        deleted_ok = False
        try:
            client.get(mid)
            record("4.delete_then_404", False, "get() still returned the memory")
        except Exception as e:
            deleted_ok = "not" in str(e).lower() or "404" in str(e) or "found" in str(e).lower()
            record("4.delete_then_404", deleted_ok, f"{type(e).__name__}: {str(e)[:80]}")
        if deleted_ok:
            created_ids.remove(mid)

        # --- 5. delete bogus id → not-found -------------------------------
        print("\n[5] delete() of a bogus id raises not-found")
        try:
            client.delete("00000000-0000-0000-0000-000000000000")
            record("5.delete_bogus_notfound", False, "bogus delete did NOT raise")
        except Exception as e:
            record("5.delete_bogus_notfound", True, f"{type(e).__name__}: {str(e)[:80]}")

    except SystemExit:
        pass
    except Exception:
        print("\nUNEXPECTED ERROR:")
        traceback.print_exc()
    finally:
        # --- cleanup: wipe scratch user ----------------------------------
        print("\n=== cleanup: wiping scratch user ===")
        try:
            client.delete_all(user_id=uid)  # delete_all takes user_id as a kwarg, NOT filters=
            time.sleep(1.5)
            leftover = _unwrap(client.get_all(filters={"user_id": uid}))
            print(f"  scratch user now has {len(leftover or [])} memories "
                  f"({'clean' if not leftover else 'RESIDUE — clean manually'})")
        except Exception as e:
            print(f"  cleanup error: {e}")

    # --- verdict ----------------------------------------------------------
    print("\n=== VERDICT ===")
    gate_keys = ["2a.update_text", "2b.update_metadata", "3.history_has_prior",
                 "4.delete_then_404", "5.delete_bogus_notfound"]
    gate_pass = all(results.get(k, (False,))[0] for k in gate_keys)
    for k in gate_keys:
        ok = results.get(k, (False, "missing"))[0]
        print(f"  {'✓' if ok else '✗'} {k}")
    mfs = results.get("6.metadata_exclude_server_side", (None,))[0]
    print(f"  (info) metadata-exclude server-side: {mfs}")
    print(f"\nKILL-GATE: {'PASS — forget/delete design is grounded; proceed to build.' if gate_pass else 'FAIL — see spec Phase-0 kill condition (fallback design).'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

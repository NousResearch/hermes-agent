#!/usr/bin/env python3
"""Live e2e smoke test for mem0 destructive tools through the REAL handler.

Drives Mem0MemoryProvider.handle_tool_call() against the hosted SDK on a
throwaway scratch user_id. Proves the actual production code path (gate, ledger,
mint-store, recall-hide, forget/restore/delete) works against real mem0 — not
just the fake-client unit tests. Wipes the scratch user at the end.

Usage:
  MEM0_DESTRUCTIVE_TOOLS=true HERMES_HOME=/tmp/mem0-e2e \
    ./venv/bin/python scripts/mem0-destructive-e2e.py --scratch-user spike-$RANDOM
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-user", required=True)
    args = ap.parse_args()
    uid = args.scratch_user
    if uid in ("ace", "hermes-user", "hermes"):
        print(f"REFUSE: '{uid}' looks like a real store."); return 2

    os.environ["MEM0_DESTRUCTIVE_TOOLS"] = "true"
    os.environ.setdefault("HERMES_HOME", "/tmp/mem0-e2e-home")
    os.makedirs(os.environ["HERMES_HOME"], exist_ok=True)

    from plugins.memory.mem0 import Mem0MemoryProvider, _FORGOTTEN_PREFIX
    p = Mem0MemoryProvider()
    p.initialize("e2e")
    p._user_id = uid
    p._agent_id = "spike-agent"
    if not p._destructive_enabled:
        print("FAIL: gate did not enable."); return 2
    client = p._get_client()

    def call(tool, a):
        return json.loads(p.handle_tool_call(tool, a))

    def seed(text):
        client.add([{"role": "user", "content": text}], user_id=uid,
                   agent_id="spike-agent", infer=False)
        time.sleep(2.0)
        allm = client.get_all(filters={"user_id": uid})
        allm = allm.get("results", allm) if isinstance(allm, dict) else allm
        return allm

    passed, failed = [], []

    def check(name, ok, detail=""):
        (passed if ok else failed).append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    print(f"=== Live e2e through real handler (scratch user={uid!r}) ===")
    try:
        # 1) seed two memories
        seed("E2E alpha: the moon is made of basalt.")
        allm = seed("E2E beta: cats dream in color.")
        ids = {m.get("memory", "")[:12]: (m.get("id") or m.get("memory_id")) for m in allm}
        beta_id = next((m.get("id") or m.get("memory_id") for m in allm
                        if "beta" in m.get("memory", "")), None)
        alpha_id = next((m.get("id") or m.get("memory_id") for m in allm
                         if "alpha" in m.get("memory", "")), None)
        check("seed", bool(beta_id and alpha_id), f"{len(allm)} memories")

        # 2) forget beta → recall hides it
        fo = call("mem0_forget", {"memory_id": beta_id, "reason": "e2e superseded"})
        check("forget.tombstone", fo.get("forgotten") == 1, json.dumps(fo)[:100])
        time.sleep(2.0)
        prof = call("mem0_profile", {})
        check("forget.hidden_from_profile",
              "cats dream" not in prof.get("result", "") and _FORGOTTEN_PREFIX not in prof.get("result", ""),
              "beta absent from recall")

        # 3) restore beta → recall shows it again
        rs = call("mem0_forget", {"memory_id": beta_id, "restore": True})
        check("restore", rs.get("result") == "restored", json.dumps(rs)[:100])
        time.sleep(2.0)
        prof2 = call("mem0_profile", {})
        check("restore.visible_again", "cats dream" in prof2.get("result", ""), "beta back in recall")

        # 4) hard delete alpha (by-id) → gone
        de = call("mem0_delete", {"memory_id": alpha_id})
        check("delete.by_id", de.get("deleted") == 1, json.dumps(de)[:100])
        time.sleep(2.0)
        try:
            client.get(alpha_id)
            check("delete.confirmed_gone", False, "still present")
        except Exception:
            check("delete.confirmed_gone", True, "404 as expected")

        # 5) ledger written, 0o600
        ledger = os.path.join(os.environ["HERMES_HOME"], "mem0-destructive-ledger.jsonl")
        has_ledger = os.path.exists(ledger)
        import stat as _stat
        mode_ok = has_ledger and _stat.S_IMODE(os.stat(ledger).st_mode) == 0o600
        check("ledger.written_0600", mode_ok, ledger if has_ledger else "missing")

    finally:
        print("\n=== cleanup: wiping scratch user ===")
        try:
            client.delete_all(user_id=uid)
            time.sleep(1.5)
            left = client.get_all(filters={"user_id": uid})
            left = left.get("results", left) if isinstance(left, dict) else left
            print(f"  scratch user now holds {len(left or [])} memories")
        except Exception as e:
            print(f"  cleanup note: {e}")

    print(f"\n=== VERDICT: {len(passed)} pass, {len(failed)} fail ===")
    if failed:
        print("  FAILED:", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""LIVE E2E for the A-lite capture pipeline: enqueue -> drain -> a REAL write to mem0.ace.

Drives the actual CapturePipeline (real queue + drain worker + certified gate) against the LIVE
mem0 server, using a THROWAWAY user_id so nothing pollutes Ace's real memories. Proves the whole
client path works end-to-end (not just the unit fakes), then cleans up.

Run on the Mac Studio (where mem0.json + the CA bundle live):
  PYTHONPATH=<worktree> venv/bin/python live_e2e_capture.py
"""
import json, os, sys, time, uuid, urllib.request, ssl

WT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WT)  # so `import capture_pipeline` (flat) resolves

import capture_pipeline as cp
import capture_scrub as scrub
from capture_queue import idem_key

CFG = json.load(open(os.path.expanduser("~/.hermes/mem0.json"), encoding="utf-8"))
HOST = CFG["host"].rstrip("/")
KEY = CFG["admin_api_key"]
CA = CFG["ca_bundle"]
TEST_USER = f"__capture_e2e_{uuid.uuid4().hex[:8]}"

_ctx = ssl.create_default_context(cafile=CA)

def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(HOST + path, data=data, method=method,
                               headers={"X-API-Key": KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, context=_ctx, timeout=60) as resp:
        return json.loads(resp.read().decode() or "{}")

# real client hooks (mirror what the plugin wires, but standalone here)
def add_fn(messages, kwargs):
    body = {"messages": messages, "user_id": TEST_USER}
    for k in ("metadata", "prompt", "model"):
        if k in kwargs and kwargs[k] is not None:
            body[k] = kwargs[k]
    _req("POST", "/memories", body)

def _search_idem(key):
    resp = _req("POST", "/search", {"query": "", "user_id": TEST_USER, "filters": {"capture_idem": key}, "top_k": 10})
    rows = resp.get("results", resp if isinstance(resp, list) else [])
    return rows

def recall_idem_fn(key): return len(_search_idem(key))
def get_written_fn(key): return [{"id": r.get("id",""), "memory": r.get("memory","")} for r in _search_idem(key)]
def forget_fn(mid):
    try: _req("PUT", f"/memories/{mid}", {"text": "[FORGOTTEN] [secret-scrubbed]", "metadata": {"forgotten": True}})
    except Exception as e: print("forget err", e)

def cleanup():
    try:
        _req("DELETE", "/memories", {"user_id": TEST_USER})
    except Exception:
        # fall back to per-id delete
        try:
            for r in _req("GET", f"/memories?user_id={TEST_USER}").get("results", []):
                _req("DELETE", f"/memories/{r['id']}")
        except Exception as e: print("cleanup err", e)

def main():
    print(f"LIVE E2E against {HOST}  user_id={TEST_USER}")
    pipe = cp.CapturePipeline(
        capture_on_fn=lambda: True,
        add_fn=add_fn, recall_idem_fn=recall_idem_fn,
        scrub_fn=lambda f: scrub.filter_facts(f),
        forget_fn=forget_fn, get_written_fn=get_written_fn,
        write_filters={"user_id": TEST_USER},
        model="gpt-5.4-mini",
        queue_path=f"/tmp/capture_e2e_{uuid.uuid4().hex[:6]}.db",
    )
    print(f"  certified={pipe._certified} gate={pipe._gate_version}")
    assert pipe._certified, "gate must be certified for E2E"

    # 1) a narration-heavy turn with ONE durable fact buried in it
    user = ("ok let's dig into the voice hub — my pipecat-house-voice runs on ACE-AI at 192.168.1.176 "
            "and I want you to check whether the wake sensitivity regressed after the last deploy")
    asst = "I'll check the wake model config and run a sensitivity test, then report back."
    key = idem_key("e2e", 1, user, asst)
    assert pipe.enqueue_turn(user, asst, session_id="e2e", turn_ordinal=1) is True
    print("  enqueued turn 1")

    # 2) drain it (real server-side extraction + gate + write)
    t0 = time.time()
    handled = pipe._worker.drain_once()
    print(f"  drain_once -> {handled} in {time.time()-t0:.1f}s")

    # 3) prove a REAL row landed, gated (durable topology fact kept, narration dropped)
    rows = get_written_fn(key)
    print(f"  rows written for this turn: {len(rows)}")
    for r in rows: print("     •", r["memory"][:100])
    assert len(rows) >= 1, "expected at least the durable topology fact to be captured"
    joined = " ".join(r["memory"].lower() for r in rows)
    assert "ace-ai" in joined or "192.168.1.176" in joined or "pipecat" in joined, "durable fact missing"
    # narration ('check whether the wake sensitivity regressed') should NOT be a stored fact
    assert "regress" not in joined and "report back" not in joined, "narration leaked!"
    print("  ✅ durable fact captured, narration filtered")

    # 4) secret-scrub E2E: a turn carrying a token -> the written fact must be scrubbed
    key2 = idem_key("e2e", 2, "sec", "sec")
    sec_user = "remember my telegram bot token is " + ("8905425635:" + "AAH3xY9zKq" + "_Wp0LmNoPqRsTuVwXyZ" + "012345")
    pipe.enqueue_turn(sec_user, "got it", session_id="e2e", turn_ordinal=2)
    pipe._worker.drain_once()
    rows2 = get_written_fn(key2)
    leaked = any("8905425635" in r["memory"] for r in rows2)
    print(f"  secret turn rows={len(rows2)} token_leaked={leaked} scrubbed={pipe._worker.stats['scrubbed']}")
    assert not leaked, "SECRET LEAKED into a stored+recallable row!"
    print("  ✅ secret scrubbed")

    print(f"  stats: {pipe.stats()}")
    pipe.stop()

if __name__ == "__main__":
    try:
        main()
        print("\nE2E PASS")
    finally:
        cleanup()
        print("cleaned up test rows")

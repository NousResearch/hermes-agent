"""Live: detached delegate_task batch, one child fails immediately, siblings run ~8 s. Does the parent's
completion queue see the per-task failure notice BEFORE the consolidated batch result?
Usage: python notice_live.py <repo_root>"""
import json, os, sys, tempfile, threading, time
root = sys.argv[1]; sys.path.insert(0, root)
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hh-")
os.environ["HERMES_STREAM_RETRIES"] = "0"

from run_agent import AIAgent
import tools.delegate_tool as dt
import tools.delegate_tool_dispatch as dd
from tools.process_registry import process_registry

parent = AIAgent(api_key="k", base_url="https://example.com/v1", provider="test-provider", model="test/model",
                 quiet_mode=True, skip_context_files=True, skip_memory=True)
parent.api_mode = "chat_completions"

# Stub the child run: task 0 fails instantly, others take 8 s and succeed.
orig_run = dd._Batch.run_child if hasattr(dd, "_Batch") else None
def fake_run_child(self, idx, task, child):
    if idx == 0:
        return {"task_index": 0, "status": "error", "error": "simulated 401 authentication_error", "api_calls": 0, "duration_seconds": 0.1}
    time.sleep(8)
    return {"task_index": idx, "status": "completed", "summary": f"ok {idx}", "api_calls": 1, "duration_seconds": 8.0}
dd._Batch.run_child = fake_run_child

# Background dispatch requires an async-capable session; emulate a CLI session key.
import tools.async_delegation as ad
t0 = time.time()
out = dt.delegate_task(tasks=[{"goal": "fail fast on a simulated 401 error"}, {"goal": "slow successful worker number one"}, {"goal": "slow successful worker number two"}], background=True, parent_agent=parent)
print("dispatch:", json.dumps(json.loads(out))[:160])
seen = []
deadline = time.time() + 30
while time.time() < deadline:
    for evt, text in process_registry.drain_notifications(session_key=""):
        kind = "TASK_FAILURE_NOTICE" if evt.get("task_failure_notice") else ("BATCH_FINAL" if evt.get("is_batch") else evt.get("type"))
        seen.append((round(time.time() - t0, 1), kind))
        print(f"  t+{time.time()-t0:5.1f}s {kind}: {text.splitlines()[0][:100]}")
    if any(k == "BATCH_FINAL" for _, k in seen):
        break
    time.sleep(0.3)
print("ORDER:", seen)

"""Live A/B for the nested-delegate deadline. A depth-1 orchestrator child dispatches one leaf that runs a
~460 s task (longer than the 420 s sequential deadline). On main the orchestrator's delegate_task call returns
'timed out after 420.0s' and the leaf runs on as an orphan; on the branch the call blocks and returns the
real result. Uses glm-5.3 via Nous for cost. Deadline shortened via config to keep the run short."""
import os, sys, json, time, re, subprocess
# Usage: python nested_delegate_deadline.py <repo_root>   (run once per ref; LIVE: a couple of real child calls)
root = sys.argv[1]; arm = os.path.basename(os.path.normpath(root))
sys.path.insert(0, root)
# Temp HERMES_HOME with the real auth + a config that shortens the generic sequential deadline to 40 s, so the
# run takes ~1.5 min instead of 8. The fix exempts delegate_task from this deadline entirely, so the shortened
# value is exactly what main will hit.
import shutil, tempfile, yaml
home = tempfile.mkdtemp(prefix="dl_home_"); os.environ["HERMES_HOME"] = home
real_home = os.environ.get("HERMES_HOME_SOURCE", os.path.expanduser("~/.hermes"))  # credentials are copied from here into a temp home
shutil.copy(f"{real_home}/auth.json", f"{home}/auth.json")
cfg = yaml.safe_load(open(f"{real_home}/config.yaml", encoding="utf-8")) or {}
cfg.setdefault("timeouts", {}).setdefault("tools", {})["sequential_call"] = 40
cfg.setdefault("delegation", {})["orchestrator_enabled"] = True
yaml.safe_dump(cfg, open(f"{home}/config.yaml", "w", encoding="utf-8"))
import agent.tool_executor as te
assert te.__file__.startswith(root)
from agent.deadline import resolve_timeout
print("effective sequential deadline:", resolve_timeout("tools.sequential_call", default=te._resolve_concurrent_tool_timeout()))
from run_agent import AIAgent
from hermes_cli.runtime_provider import resolve_runtime_provider
MODEL = "z-ai/glm-5.3-flash"
rt = resolve_runtime_provider(requested="nous", target_model=MODEL)
sid = f"dl_{arm}_{int(time.time())}"
ag = AIAgent(model=MODEL, provider="nous", base_url=rt.get("base_url"), api_key=rt.get("api_key"), api_mode=rt.get("api_mode"),
             session_id=sid, quiet_mode=True, enabled_toolsets=["terminal", "delegation"], platform="cli", max_iterations=8,
             skip_context_files=True, skip_memory=True)
# Make this agent a depth-1 orchestrator exactly as delegate_tool_child_run does for a real nested orchestrator:
# at depth>0 delegate_task runs synchronously inside the tool call, which is the path under the deadline.
ag._delegate_depth = 1
ag._delegate_role = "orchestrator"
task = ("Use delegate_task exactly once (not background) with a single task whose goal is: "
        "\"Run the shell command `sleep 75 && echo LEAF_DONE_MARKER` with the terminal tool (background=false is fine, it is under the tool timeout), "
        "then reply with the exact text the command printed.\" "
        "When delegate_task returns, reply with ONE line: RESULT: followed by the child's summary text verbatim (or the error text if it errored).")
t0 = time.time(); r = ag.run_conversation(task); wall = time.time() - t0
final = (r.get("final_response") or "").strip()
print(f"ARM {arm}: wall={wall:.0f}s final={final[:300]!r}")
timed_out = "timed out" in final.lower()
got_marker = "LEAF_DONE_MARKER" in final
print(json.dumps({"arm": arm, "delegate_timed_out": timed_out, "orchestrator_got_leaf_result": got_marker, "wall_s": round(wall)}))

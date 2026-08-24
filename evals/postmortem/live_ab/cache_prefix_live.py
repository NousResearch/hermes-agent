"""Live A/B for the thinking-strip cache miss (F0). Runs a short real tool loop through AIAgent on
Fable 5.1 via Nous and prints per-call cache hit ratios from agent.log. ~10 calls, well under $1.
Arm A = current code. Arm B = HERMES_KEEP_ALL_THINKING=1 monkeypatch of _manage_thinking_signatures
that passes thinking blocks back unchanged for the Nous/Anthropic route."""
import os, sys, re, time, subprocess, json
# LIVE: real provider calls (cents). Usage: python cache_prefix_live.py <repo_root> <A|B>
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))
sys.path.insert(0, sys.argv[1])
arm = sys.argv[2] if len(sys.argv) > 2 else "A"
import agent.anthropic_message_convert as amc
if arm == "B":
    _orig = amc._manage_thinking_signatures
    def _keep_all(result, base_url, model):
        # keep-all model on a signature-validating route: pass every thinking block back unchanged,
        # only drop cache_control on thinking blocks and the internal flag.
        for idx, m in amc._assistant_block_lists(result):
            for b in m["content"]:
                if amc._block_type(b) in amc._THINKING_TYPES:
                    b.pop("cache_control", None)
            m.pop("_thinking_signature_invalidated", None)
    amc._manage_thinking_signatures = _keep_all
    # the adapter imports the name at call time via module attribute? verify binding
    import agent.anthropic_adapter as ad
    if hasattr(ad, "_manage_thinking_signatures"):
        ad._manage_thinking_signatures = _keep_all
from run_agent import AIAgent
from hermes_cli.runtime_provider import resolve_runtime_provider
rt = resolve_runtime_provider(requested="nous", target_model="anthropic/claude-fable-5.1")
sid = f"f0ab_{arm}_{int(time.time())}"
ag = AIAgent(model="anthropic/claude-fable-5.1", provider="nous", base_url=rt.get("base_url"), api_key=rt.get("api_key"),
             api_mode=rt.get("api_mode"), session_id=sid, quiet_mode=True,
             enabled_toolsets=["file", "terminal"], platform="cli", max_iterations=12,
             skip_context_files=True, skip_memory=True, reasoning_config={"enabled": True, "effort": "medium"})
task = ("In /tmp/f0ab_work (create it), do these steps ONE tool call at a time, no parallel calls: "
        "1) write a.txt with 'alpha', 2) write b.txt with 'beta', 3) read a.txt, 4) read b.txt, "
        "5) run `ls -la /tmp/f0ab_work`, 6) run `wc -c /tmp/f0ab_work/*`, 7) run `cat /tmp/f0ab_work/a.txt`, "
        "then reply with one line: DONE.")
t0 = time.time()
r = ag.run_conversation(task)
print("final:", (r.get("final_response") or "")[:80], "| wall", round(time.time() - t0, 1), "s")
time.sleep(1)
log = subprocess.run(f"grep -h '\\[{sid}\\]' ~/.hermes/logs/agent.log | grep 'API call #'", shell=True, capture_output=True).stdout.decode("utf-8", "replace")
rows = re.findall(r"API call #(\d+): .*in=(\d+) out=(\d+) .*cache=(\d+)/(\d+) \((\d+)%\)", log)
tot_in = tot_c = 0
for n, i, o, c, ct, p in rows:
    i, c = int(i), int(c); tot_in += i; tot_c += c
    print(f"  call {n:>2} in={i:>7} out={o:>5} cached={c:>7} ({p}%)  uncached={i-c}")
print(f"ARM {arm}: calls={len(rows)} input={tot_in} cached={tot_c} uncached={tot_in-tot_c} hit={100*tot_c/max(tot_in,1):.1f}%")
print(json.dumps({"arm": arm, "calls": len(rows), "input": tot_in, "cached": tot_c}))

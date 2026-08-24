"""Definitive F0 test: capture consecutive wire payloads on a real Fable 5.1 tool loop and diff the
message prefix between call N and N+1. If Hermes strips prior-turn thinking, call N+1's messages[:k]
will NOT equal call N's messages (prefix divergence) even though the conversation only grew.
Also reports cache hit per call. Cost: a handful of calls."""
import os, sys, re, time, json, copy, subprocess
# LIVE: makes ~6 real calls to the configured provider (a few cents). Usage:
#   python cache_prefix_wire.py <repo_root> <A|B> [--hermes-home DIR]   (default HERMES_HOME: the real one, for credentials)
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))
if "--hermes-home" in sys.argv:
    os.environ["HERMES_HOME"] = sys.argv[sys.argv.index("--hermes-home") + 1]
arm = sys.argv[2] if len(sys.argv) > 2 else "A"
import agent.anthropic_message_convert as amc
if arm == "B":
    def _keep_all(result, base_url, model):
        for idx, m in amc._assistant_block_lists(result):
            for b in m["content"]:
                if amc._block_type(b) in amc._THINKING_TYPES: b.pop("cache_control", None)
            m.pop("_thinking_signature_invalidated", None)
    amc._manage_thinking_signatures = _keep_all
# Capture every outbound Anthropic-format payload
captured = []
import agent.anthropic_adapter as ad
_orig_convert = None
for name in ("convert_messages_to_anthropic", "to_anthropic_messages", "convert_to_anthropic", "build_anthropic_messages"):
    if hasattr(amc, name): _orig_convert = (name, getattr(amc, name)); break
if _orig_convert is None:
    # find the public converter: the function that calls _manage_thinking_signatures at line ~703
    import inspect
    for name, fn in inspect.getmembers(amc, inspect.isfunction):
        try:
            if "_manage_thinking_signatures(result" in inspect.getsource(fn): _orig_convert = (name, fn); break
        except Exception: pass
name, fn = _orig_convert
def _wrapped(*a, **k):
    out = fn(*a, **k)
    try:
        msgs = out[1] if isinstance(out, tuple) else out   # (system, messages)
        captured.append(copy.deepcopy(msgs))
    except Exception: pass
    return out
setattr(amc, name, _wrapped)
if hasattr(ad, name): setattr(ad, name, _wrapped)
print("hooked converter:", name)
from run_agent import AIAgent
from hermes_cli.runtime_provider import resolve_runtime_provider
rt = resolve_runtime_provider(requested="nous", target_model="anthropic/claude-fable-5.1")
sid = f"f0wire_{arm}_{int(time.time())}"
ag = AIAgent(model="anthropic/claude-fable-5.1", provider="nous", base_url=rt.get("base_url"), api_key=rt.get("api_key"),
             api_mode=rt.get("api_mode"), session_id=sid, quiet_mode=True, enabled_toolsets=["file", "terminal"],
             platform="cli", max_iterations=10, skip_context_files=True, skip_memory=True,
             reasoning_config={"enabled": True, "effort": "medium"})
task = ("Work in /tmp/f0wire (create it). Before EACH tool call, think carefully for a moment about edge cases. "
        "Steps, one tool call each: 1) write notes.md with a 5-line summary of what a Python context manager is; "
        "2) read it back; 3) run `wc -l /tmp/f0wire/notes.md`; 4) append one more line to notes.md explaining __exit__ return values; "
        "5) read it back; then reply DONE.")
ag.run_conversation(task)
time.sleep(1)
log = subprocess.run(f"grep -h '\\[{sid}\\]' ~/.hermes/logs/agent.log | grep 'API call #'", shell=True, capture_output=True).stdout.decode("utf-8", "replace")
rows = re.findall(r"API call #(\d+): .*in=(\d+) out=(\d+) .*cache=(\d+)/(\d+) \((\d+)%\)", log)
for r in rows: print(f"  call {r[0]:>2} in={int(r[1]):>6} out={int(r[2]):>5} cached={int(r[3]):>6} ({r[5]}%) uncached={int(r[1])-int(r[3])}")
print(f"ARM {arm}: captured {len(captured)} payloads")
# prefix divergence check
def sig(m):  # message signature: role + block types + text/thinking lengths + signature presence
    c = m.get("content")
    if isinstance(c, list):
        return (m.get("role"), tuple((b.get("type"), len(str(b.get("text", b.get("thinking", b.get("input", ""))))), bool(b.get("signature"))) for b in c))
    return (m.get("role"), str(c)[:200])
div = 0
for i in range(1, len(captured)):
    prev, cur = captured[i-1], captured[i]
    k = len(prev)
    same = [sig(a) == sig(b) for a, b in zip(prev, cur[:k])]
    if not all(same):
        j = same.index(False); div += 1
        print(f"  payload {i}: prefix DIVERGED at message {j}/{k}: prev={sig(prev[j])}  cur={sig(cur[j])}")
print(f"ARM {arm}: {div} of {len(captured)-1} consecutive payloads had a mutated prefix (thinking blocks in prev assistant msgs: "
      f"{sum(1 for m in captured[-1] if m.get('role')=='assistant' and isinstance(m.get('content'),list) and any(b.get('type')=='thinking' for b in m['content']))} of "
      f"{sum(1 for m in captured[-1] if m.get('role')=='assistant')} assistant msgs in final payload)")

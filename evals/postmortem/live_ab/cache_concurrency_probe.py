"""Live prompt-cache concurrency probe: does a route keep cache routing sticky under a fan-out?

Runs N independent Hermes ``AIAgent`` sessions concurrently, each the same ~8-step tool loop whose
context grows 6K -> ~240K, and records for EVERY API call: prompt / cache_read / cache_creation /
output tokens, the response id, the upstream provider when the route reports one, and a sha of the
system prompt, tools and every message, so client-side prefix mutation can be ruled in or out.

A consecutive pair (call k, k+1) in one session is classified:
  ideal     cache_read(k+1) == prompt(k)            the previous call's write was read back
  stuck     cache_read(k+1) == cache_read(k)        the previous write was NOT visible: routing
  collapse  cache_read(k+1) <  50% of cache_read(k) the whole conversation was re-written
  (pairs whose system sha changed are compaction/aux calls and are excluded)

Results that motivated hermes-agent #104284 / #104421 and NousResearch/api#227 (2026-09-05/06,
Fable 5.1, 20 sessions x 6 calls unless noted):
  nous, native /v1/messages         13.9% stuck (4 runs 14-20%)     -> chat is the Nous default
  nous, /v1/chat/completions         0 / 320 pairs
  openrouter direct, pinned anthropic 0 / 161 pairs
  openrouter direct, unpinned (40x8)  9.8% collapse
  2 s settle before every call        no change (not a race)

Usage:
  python -m evals.postmortem.live_ab.cache_concurrency_probe --repo . --provider nous \
      --workers 20 --calls 6 --out /tmp/probe.jsonl [--wire chat|native] [--model ID] \
      [--pin anthropic] [--settle 2] [--ttl 5m]
  providers: nous (Portal creds from HERMES_HOME), openrouter (OPENROUTER_API_KEY or --api-key),
             anthropic (ANTHROPIC_API_KEY or --api-key)
Cost: ~$50 per 20x6 arm on Fable 5.1 at the 5m tier. Every run starts from the real Hermes request
path; the only patch is a read-only wrapper on the SDK stream that records usage and headers.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback


def _parse():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="hermes-agent checkout to import from")
    ap.add_argument("--provider", required=True, choices=["nous", "openrouter", "anthropic"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--calls", type=int, default=6, help="tool calls per session")
    ap.add_argument("--out", required=True, help="JSONL of every call; summary written next to it")
    ap.add_argument("--wire", choices=["chat", "native"], default=None,
                    help="force the wire (default: what Hermes would pick for the provider/model)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--pin", default=None, help="OpenRouter provider slug to pin (providers_allowed)")
    ap.add_argument("--settle", type=float, default=0.0, help="seconds to sleep before every request")
    ap.add_argument("--ttl", choices=["5m", "1h"], default="5m", help="cache_control TTL on the wire")
    ap.add_argument("--api-key", default=None)
    return ap.parse_args()


ARGS = _parse()
REPO, PROVIDER, N, CALLS, OUT = ARGS.repo, ARGS.provider, ARGS.workers, ARGS.calls, ARGS.out
SETTLE_S = ARGS.settle
sys.path.insert(0, os.path.abspath(REPO))
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))
import anthropic
from anthropic.resources.messages import Messages
_orig_stream = Messages.stream
LOCK = threading.Lock()
TLS = threading.local()

def _write(rec):
    with LOCK:
        with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps(rec, default=str) + "\n")

class _Ctx:
    def __init__(self, inner, rec): self.inner, self.rec = inner, rec
    def __enter__(self):
        self.mgr = self.inner.__enter__()
        try:
            resp = getattr(self.mgr, "response", None)
            hdrs = dict(getattr(resp, "headers", {}) or {})
            self.rec["request_id"] = hdrs.get("request-id") or hdrs.get("x-request-id")
            self.rec["nous_headers"] = {k: v for k, v in hdrs.items() if k.lower().startswith("x-nous") or k.lower() in ("cf-ray", "x-request-id", "request-id")}
        except Exception as e: self.rec["hdr_err"] = repr(e)[:120]
        return self
    def __iter__(self): return iter(self.mgr)
    def get_final_message(self):
        m = self.mgr.get_final_message()
        try:
            u = m.usage
            self.rec.update(prompt_tokens=(u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
                            uncached_input=u.input_tokens, cache_read=u.cache_read_input_tokens, cache_creation=u.cache_creation_input_tokens,
                            output=u.output_tokens, message_id=m.id, t_end=time.time())
        except Exception as e: self.rec["usage_err"] = repr(e)[:120]
        _write(self.rec); return m
    def __exit__(self, *a): return self.mgr.__exit__(*a)
    def __getattr__(self, n): return getattr(self.mgr, n)

def patched_stream(self, **kw):
    msgs = kw.get("messages") or []
    rec = dict(worker=int(m.group(1)) if m else None, call=len(msgs), t_start=time.time(), model=kw.get("model"),
               n_msgs=len(msgs), system_sha=sys_sha, tools_sha=tools_sha, msg_shas=msg_shas)
    if SETTLE_S > 0 and len(msgs) > 1:
        time.sleep(SETTLE_S); rec["settle_s"] = SETTLE_S
    return _Ctx(_orig_stream(self, **kw), rec)
Messages.stream = patched_stream

from openai.resources.chat.completions import Completions
_orig_create = Completions.create
class _OAStream:
    def __init__(self, inner, rec): self.inner, self.rec, self.usage, self._id, self._prov = inner, rec, None, None, None
    def __iter__(self):
        for ch in self.inner:
            if self._id is None: self._id = getattr(ch, "id", None)
            if self._prov is None: self._prov = getattr(ch, "provider", None)
            u = getattr(ch, "usage", None)
            if u is not None: self.usage = u
            yield ch
        try:
            u = self.usage; d = getattr(u, "prompt_tokens_details", None)
            d = d.model_dump() if hasattr(d, "model_dump") else (dict(d.__dict__) if d else {})
            cached = d.get("cached_tokens") or 0; cw = d.get("cache_write_tokens") or d.get("cache_creation_input_tokens") or 0
            self.rec.update(prompt_tokens=u.prompt_tokens, cache_read=cached, cache_creation=cw, uncached_input=(u.prompt_tokens or 0) - cached - cw,
                            output=u.completion_tokens, message_id=getattr(self, "_id", None), t_end=time.time(), details=d,
                            cost=float(getattr(u, "cost", 0) or 0), cost_details=getattr(u, "cost_details", None), upstream_provider=getattr(self, "_prov", None))
        except Exception as e: self.rec["usage_err"] = repr(e)[:160]
        _write(self.rec)
    def __getattr__(self, n): return getattr(self.inner, n)
def patched_create(self, *a, **kw):
    msgs = kw.get("messages") or []
    import re as _re
    first = msgs[0] if msgs else {}
    sysm = next((m for m in msgs if m.get("role") == "system"), None)
    nonsys = [m for m in msgs if m.get("role") != "system"]
    f0 = nonsys[0] if nonsys else {}
    ftxt = f0.get("content") if isinstance(f0.get("content"), str) else "".join(b.get("text", "") for b in (f0.get("content") or []) if isinstance(b, dict))
    m = _re.search(r"\[probe-session (\d+)\]", ftxt or "")
    sha = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:10]
    rec = dict(worker=int(m.group(1)) if m else None, call=len(nonsys), t_start=time.time(), model=kw.get("model"), n_msgs=len(nonsys),
               system_sha=sha(sysm), tools_sha=sha(kw.get("tools")), msg_shas=[sha(x) for x in nonsys], extra_body=kw.get("extra_body"))
    if SETTLE_S > 0 and len(nonsys) > 1: time.sleep(SETTLE_S); rec["settle_s"] = SETTLE_S
    if kw.get("stream"):
        kw["stream_options"] = {**(kw.get("stream_options") or {}), "include_usage": True}
    resp = _orig_create(self, *a, **kw)
    if kw.get("stream"):
        return _OAStream(resp, rec)
    try:
        u = resp.usage; d = getattr(u, "prompt_tokens_details", None); d = d.model_dump() if hasattr(d, "model_dump") else {}
        cached = d.get("cached_tokens") or 0; cw = d.get("cache_write_tokens") or 0
        rec.update(prompt_tokens=u.prompt_tokens, cache_read=cached, cache_creation=cw, uncached_input=u.prompt_tokens - cached - cw, output=u.completion_tokens,
                   message_id=resp.id, t_end=time.time(), details=d, upstream_provider=getattr(resp, "provider", None))
    except Exception as e: rec["usage_err"] = repr(e)[:160]
    _write(rec); return resp
MODEL = ARGS.model or ("claude-fable-5.1" if PROVIDER == "anthropic" else "anthropic/claude-fable-5.1")
if ARGS.wire:
    API_MODE = "chat_completions" if ARGS.wire == "chat" else "anthropic_messages"
elif PROVIDER == "nous":
    from hermes_cli.providers import nous_api_mode
    API_MODE = nous_api_mode(MODEL)
else:
    API_MODE = "chat_completions" if PROVIDER == "openrouter" else "anthropic_messages"
if API_MODE == "chat_completions": Completions.create = patched_create

import agent.prompt_caching as _pc
_pc.effective_cache_ttl = lambda ttl, *, model="", provider="": ARGS.ttl  # the probe pins the TTL; user config must not leak in
from run_agent import AIAgent
def creds():
    if PROVIDER == "openrouter":
        key = ARGS.api_key or os.environ.get("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY or --api-key required")
        # native wire: the SDK appends /v1/messages itself
        base = "https://openrouter.ai/api" if ARGS.wire == "native" else "https://openrouter.ai/api/v1"
        return dict(api_key=key, base_url=base, provider="openrouter")
    if PROVIDER == "anthropic":
        key = ARGS.api_key or os.environ.get("ANTHROPIC_API_KEY") or sys.exit("ANTHROPIC_API_KEY or --api-key required")
        return dict(api_key=key, base_url="https://api.anthropic.com", provider="anthropic")
    from hermes_cli.auth_nous import resolve_nous_runtime_credentials
    c = resolve_nous_runtime_credentials()
    return dict(api_key=c["api_key"], base_url=c.get("base_url") or "https://inference-api.nousresearch.com/v1", provider="nous")
CRED = creds()
WORKDIR = tempfile.mkdtemp(prefix="cacheprobe-")
# seed ~35K tokens of file content so the loop's context grows fast and realistically (tool results, not user text)
for i in range(6):
    with open(f"{WORKDIR}/part{i}.txt", "w", encoding="utf-8") as f:
        f.write("".join(f"line {j:05d} of part {i}: {hashlib.sha256(f'{i}-{j}'.encode()).hexdigest()}\n" for j in range(700)))
TASK_PREFIX = "[probe-session {w}] "
TASK = (f"You are testing a tool loop. In {WORKDIR} there are part0.txt..part5.txt. Read each file fully with read_file, ONE per turn, "
        f"then run `wc -l {WORKDIR}/*.txt` with terminal, then read part0.txt again, then reply with the total line count and the first 8 chars "
        f"of the sha on line 00042 of part3. Do not summarize the files; just call the tools, one at a time. Total tool calls: {CALLS}.")

def worker(w):
    TLS.worker = w
    try:
        extra = dict(providers_allowed=[ARGS.pin], provider_sort=None) if (ARGS.pin and PROVIDER == "openrouter") else {}
        a = AIAgent(api_key=CRED["api_key"], base_url=CRED["base_url"], provider=CRED["provider"], **extra,
                    api_mode=API_MODE, model=MODEL, session_id=f"cacheprobe-{PROVIDER}-{ARGS.wire or API_MODE}-{N}-{w}",
                    platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                    enabled_toolsets=["file", "terminal"], max_iterations=CALLS + 4)
        a.run_conversation(TASK_PREFIX.format(w=w) + TASK)
    except Exception:
        _write(dict(worker=w, error=traceback.format_exc()[-600:]))

open(OUT, "w", encoding="utf-8").close()
t0 = time.time(); th = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(N)]
for t in th: t.start()
for t in th: t.join(timeout=900)
print(f"done in {time.time()-t0:.0f}s")

# ---- summary: classify consecutive pairs per session
rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
errs = [r for r in rows if "error" in r]; rows = [r for r in rows if "cache_read" in r]
byw = collections.defaultdict(list)
for r in rows: byw[r["worker"]].append(r)
for v in byw.values(): v.sort(key=lambda r: r["t_start"])
cat = collections.Counter(); stuck_tok = 0; bad = []
for w, cs in byw.items():
    for p, c in zip(cs, cs[1:]):
        if c["system_sha"] != p["system_sha"]: cat["compaction"] += 1; continue
        if abs(c["cache_read"] - p["prompt_tokens"]) <= 0.01 * p["prompt_tokens"] + 50: cat["ideal"] += 1
        elif abs(c["cache_read"] - p["cache_read"]) <= 50: cat["stuck"] += 1; stuck_tok += p["prompt_tokens"] - p["cache_read"]; bad.append((w, p, c, "stuck"))
        elif c["cache_read"] < 0.5 * p["cache_read"]: cat["collapse"] += 1; bad.append((w, p, c, "collapse"))
        else: cat["other"] += 1
pairs = sum(cat.values()) - cat["compaction"]
tot_in = sum(r["prompt_tokens"] for r in rows); tot_read = sum(r["cache_read"] for r in rows); tot_cre = sum(r["cache_creation"] for r in rows)
cost = sum(float(r.get("cost") or 0) for r in rows)
print(f"\n== {PROVIDER} wire={API_MODE} N={N} x {CALLS}: sessions={len(byw)} calls={len(rows)} errors={len(errs)} hit={tot_read/max(tot_in,1)*100:.1f}% "
      f"cache_creation={tot_cre:,}" + (f" cost=${cost:.2f}" if cost else ""))
print(f"   pairs={pairs}  ideal={cat['ideal']}  stuck={cat['stuck']} ({cat['stuck']/max(pairs,1)*100:.1f}%)  collapse={cat['collapse']} ({cat['collapse']/max(pairs,1)*100:.1f}%)  other={cat['other']}"
      f"  | tokens re-written by stuck pairs: {stuck_tok:,}")
for w, p, c, kind in bad[:10]:
    print(f"  {kind.upper():8} w{w}: read {p['cache_read']:,}->{c['cache_read']:,} prompt {p['prompt_tokens']:,}->{c['prompt_tokens']:,} write {c['cache_creation']:,} "
          f"ids {p.get('message_id')} -> {c.get('message_id')} upstream {p.get('upstream_provider')}->{c.get('upstream_provider')}")
json.dump(dict(provider=PROVIDER, wire=API_MODE, model=MODEL, workers=N, calls_per_session=CALLS, ttl=ARGS.ttl, pin=ARGS.pin, settle_s=SETTLE_S,
               calls=len(rows), errors=len(errs), hit=tot_read / max(tot_in, 1), pairs=pairs, **{k: v for k, v in cat.items()}, stuck_tokens=stuck_tok, cost_usd=cost,
               bad_pairs=[dict(kind=k, session=f"cacheprobe-{PROVIDER}-{ARGS.wire or API_MODE}-{N}-{w}", ok_id=p.get("message_id"), ok_cf_ray=(p.get("nous_headers") or {}).get("cf-ray"),
                              bad_id=c.get("message_id"), bad_cf_ray=(c.get("nous_headers") or {}).get("cf-ray"), read_before=p["cache_read"], read_after=c["cache_read"],
                              expected_read=p["prompt_tokens"], write_after=c["cache_creation"]) for w, p, c, k in bad]),
          open(OUT.replace(".jsonl", ".summary.json"), "w", encoding="utf-8"), indent=1)

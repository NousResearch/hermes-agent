#!/usr/bin/env python3
"""Phase-2 save-decision eval (the GATE).

Runs the REAL memory-review prompt + the mem0_remember salience clause through a model
over held-out multi-turn fixtures, and records whether the model decides to save
(call mem0_remember) or not. Gate (S10 holdout precedent):
  - save-recall Wilson 95% LB >= 0.75  (genuine facts that get saved)
  - false-save rate            <= 0.10 (narration/ambiguous that get saved)

The model is asked to emit a strict JSON verdict so the decision is mechanically graded
(no model-grades-model). Fixtures were authored BEFORE the clause wording and label the
SAVE DECISION, not the wording.
"""
import json, os, sys, urllib.request, urllib.error, math

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "bgr_save_fixtures.jsonl")

# Import the REAL clause + rubric the fork uses. Resolve the repo root from THIS file
# (eval/ lives at the repo root), so it runs from any checkout, not an author-local path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from agent.background_review import _MEMORY_REVIEW_PROMPT, _MEMORY_REVIEW_MEM0_CLAUSE

def _key():
    # Environment first (works on any machine / CI); the personal .env is an
    # optional fallback for the author's box and must never crash when absent.
    k = os.environ.get("OPENAI_API_KEY", "")
    if k:
        return k
    env_path = os.path.expanduser("~/.hermes/.env")
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI" + "_API_" + "KEY="):
                    return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return ""

SYSTEM = (
    "You are the background self-improvement review pass of an AI assistant. You read a short "
    "conversation excerpt and decide whether it contains a DURABLE fact about the user worth "
    "saving to long-term memory via the mem0_remember tool.\n\n"
    + _MEMORY_REVIEW_PROMPT + _MEMORY_REVIEW_MEM0_CLAUSE +
    "\n\nFor THIS eval, do not call a tool. Instead output STRICT JSON only: "
    '{\"save\": true|false, \"fact\": \"<the one durable fact, or empty>\"}. '
    "save=true ONLY if you would call mem0_remember on this excerpt."
)

def decide(transcript, k, model="gpt-5-nano", url="https://api.openai.com/v1/chat/completions", auth=True):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Conversation excerpt:\n" + transcript},
        ],
    }
    if "openai.com" in url:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {k}"
    import time as _t
    last = None
    for attempt in range(8):
        try:
            r = urllib.request.Request(url, data=body, method="POST", headers=headers)
            resp = json.loads(urllib.request.urlopen(r, timeout=60).read())
            txt = resp["choices"][0]["message"]["content"]
            import re
            m = re.search(r"\{[^}]*\"save\"[^}]*\}", txt)
            if m:
                try:
                    return bool(json.loads(m.group(0)).get("save"))
                except Exception:
                    pass
            return "\"save\": true" in txt.lower() or '"save":true' in txt.lower()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                _t.sleep(6 * (attempt + 1)); continue
            raise
        except Exception as e:
            last = e
            _t.sleep(3); continue
    raise last


CLAUDE_URL = os.environ.get("SAVE_EVAL_CLAUDE_URL", "http://localhost:18810/anthropic/v1/messages")

def decide_claude(transcript, model="claude-opus-4-8"):
    """Decide using the REAL fork model (Apollo runs claude-opus-4-8). This is the
    honest measure — the review fork inherits agent.model, not gpt-5-nano."""
    body = json.dumps({
        "model": model,
        "max_tokens": 200,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": "Conversation excerpt:\n" + transcript +
                      "\n\nOutput ONLY the strict JSON verdict."}],
    }).encode()
    import time as _t
    last = None
    for attempt in range(8):
        try:
            r = urllib.request.Request(CLAUDE_URL, data=body, method="POST",
                                       headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
            resp = json.loads(urllib.request.urlopen(r, timeout=60).read())
            txt = resp["content"][0]["text"]
            import re
            m = re.search(r"\{[^}]*\"save\"[^}]*\}", txt)
            if m:
                try:
                    return bool(json.loads(m.group(0)).get("save"))
                except Exception:
                    pass
            return "\"save\": true" in txt.lower() or '"save":true' in txt.lower()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                _t.sleep(8 * (attempt + 1)); continue
            raise
        except Exception as e:
            last = e
            _t.sleep(4); continue
    raise last

def wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c - m) / d

def main():
    k = _key()
    backend = os.environ.get("SAVE_EVAL_BACKEND", "bpp")
    if backend == "bpp":
        decide_fn = lambda t: decide(t, k, model="claude-opus-4-8",
                                     url="http://localhost:18811/v1/chat/completions", auth=False)
        label = "claude-opus-4-8 via claude-bpp (real fork model)"
    elif backend == "claude":
        decide_fn = lambda t: decide_claude(t)
        label = "claude-opus-4-8 via claude-pool (real fork model)"
    else:
        decide_fn = lambda t: decide(t, k)
        label = "gpt-5-nano"
    print(f"backend: {backend} ({label})")

    # Preflight: fail EARLY with a clear, actionable message instead of crash-looping
    # on connection-refused (CI / a checkout without the local claude relay). The
    # default backend uses the fork's real model (claude-opus-4-8) via a relay; a
    # bare OPENAI_API_KEY does NOT reach it — that's the gpt5nano backend (weak,
    # non-representative; see BGR-SAVE-EVAL-RESULT.md).
    if backend in ("bpp", "claude"):
        probe_url = ("http://localhost:18811/v1/chat/completions" if backend == "bpp"
                     else CLAUDE_URL)
        try:
            import socket
            from urllib.parse import urlparse as _up
            u = _up(probe_url)
            socket.create_connection((u.hostname, u.port or 80), timeout=3).close()
        except Exception:
            print(f"ERROR: backend '{backend}' needs a reachable claude relay at "
                  f"{probe_url}, which is not up. This gate runs against the fork's "
                  f"REAL model (claude-opus-4-8); a bare OPENAI_API_KEY does not reach "
                  f"it. Options: point SAVE_EVAL_CLAUDE_URL at a reachable relay, or "
                  f"run the non-representative gpt-5-nano path with "
                  f"SAVE_EVAL_BACKEND=gpt5nano (requires OPENAI_API_KEY).")
            sys.exit(2)
    elif not k:
        print("ERROR: gpt5nano backend requires OPENAI_API_KEY in the environment.")
        sys.exit(2)

    rows = [json.loads(l) for l in open(FIX, encoding="utf-8") if l.strip()]
    genuine = [r for r in rows if r["expect"] == "save"]
    nosave = [r for r in rows if r["expect"] == "no_save"]

    saved_genuine = 0
    false_saves = 0
    misses, fps = [], []
    for r in rows:
        d = decide_fn(r["transcript"])
        import time as _t; _t.sleep(1.0)  # gentle pacing for the shared Opus relay
        if r["expect"] == "save":
            if d: saved_genuine += 1
            else: misses.append(r["id"])
        else:
            if d:
                false_saves += 1
                fps.append(r["id"])

    n_g = len(genuine); n_ns = len(nosave)
    recall = saved_genuine / n_g if n_g else 0
    recall_lb = wilson_lb(saved_genuine, n_g)
    fsave = false_saves / n_ns if n_ns else 0

    print(f"genuine: {saved_genuine}/{n_g} saved (recall {recall:.1%}, Wilson95 LB {recall_lb:.3f})")
    print(f"  misses: {misses}")
    print(f"no-save: {false_saves}/{n_ns} wrongly saved (false-save {fsave:.1%})")
    print(f"  false-saves: {fps}")
    ok = recall_lb >= 0.75 and fsave <= 0.10
    print(f"\nGATE: save-recall LB {recall_lb:.3f} >= 0.75 AND false-save {fsave:.1%} <= 10%")
    print(f"SAVE-EVAL {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

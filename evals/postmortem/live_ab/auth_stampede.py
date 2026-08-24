"""Live A/B of the Nous hourly-expiry stampede, no real network.

Local server: accepts bearer FRESH, returns 401 {"type":"authentication_error", "Your API key is invalid,
blocked or out of funds..."} for any other bearer. resolve_nous_runtime_credentials is patched to return
FRESH (standing in for the auth store the keepalive/peers have refreshed). N agents are built holding a
STALE JWT that expires in 30 s and each fires one API call concurrently. Count 401s the server saw.

Usage: python stampede_ab.py <repo_root> <n_agents>
"""
import base64, json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

root, n = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, root)
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hh-")
os.environ["HERMES_STREAM_RETRIES"] = "0"


def jwt(exp, sub="acct-A"):
    # `sub` matters: pre-expiry adoption (#103526 round 2) only swaps to a key for the SAME account.
    b = lambda o: base64.urlsafe_b64encode(json.dumps(o).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{b({'alg':'none'})}.{b({'exp':exp,'sub':sub})}.s"


STALE, FRESH = jwt(time.time() + 30), jwt(time.time() + 3600)
hits = {"401": 0, "200": 0}
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        auth = self.headers.get("authorization", "")
        if not self.path.endswith("/chat/completions"):
            self.send_response(404); self.send_header("content-length", "0"); self.end_headers(); return
        if auth == f"Bearer {FRESH}":
            body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            with lock: hits["200"] += 1
            self.send_response(200)
        else:
            if os.environ.get("TRACE401"):
                import traceback; sys.stderr.write("401 path: "+self.path+"\n")
            body = json.dumps({"error": {"type": "authentication_error", "message": "Your API key is invalid, blocked or out of funds. Please go visit the portal to sort that out: https://portal.nousresearch.com "}}).encode()
            with lock: hits["401"] += 1
            self.send_response(401)
        self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}/v1"

import hermes_cli.auth as auth_mod
auth_mod.resolve_nous_runtime_credentials = lambda **kw: {"api_key": FRESH, "base_url": base}
import hermes_cli.nous_auth_keepalive as ka
ka.start_nous_auth_keepalive = lambda **kw: None  # thread itself is out of scope here; we test adoption

from run_agent import AIAgent

agents = []
for i in range(n):
    a = AIAgent(api_key=STALE, base_url=base, provider="nous", model="test/model", quiet_mode=True,
                skip_context_files=True, skip_memory=True)
    a.api_mode = "chat_completions"
    a._interrupt_requested = False
    agents.append(a)

results = []
def go(a):
    try:
        results.append(a.chat("hi")[:20]); results.append("KEY:"+a.api_key[:5])
    except Exception as e:
        results.append(f"ERR {type(e).__name__}")

ts = [threading.Thread(target=go, args=(a,)) for a in agents]
t0 = time.time(); [t.start() for t in ts]; [t.join(120) for t in ts]
print("keys:", sorted(set(r for r in results if r.startswith("KEY"))));print(f"agents={n} server_401={hits['401']} server_200={hits['200']} errors={sum(r.startswith('ERR') for r in results)} wall={time.time()-t0:.1f}s")

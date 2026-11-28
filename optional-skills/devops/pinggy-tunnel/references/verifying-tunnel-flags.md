# Pinggy remote-arg verification

How to prove a remote arg (after `--`) actually does what you claim, and how to
read the flags back. Use this when the SKILL.md table is not enough — e.g. a new
flag, a suspicious gate, or a report that "the URL is protected".

## Probe origin: echo the received request headers

Most flag effects are invisible from the response alone. Serve an origin that
returns the headers it received, then read them back.

```python
# echo_server.py — run with terminal(background=True) on 127.0.0.1:18001
import http.server, json
class H(http.server.BaseHTTPRequestHandler):
    def _echo(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        payload = json.dumps({"ok": True, "headers": dict(self.headers), "body": body}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self): self._echo()
    def do_POST(self): self._echo()
    def do_OPTIONS(self): self._echo()
    def log_message(self, *a, **k): pass
http.server.HTTPServer(("127.0.0.1", 18001), H).serve_forever()
```

Read the result without a parser dependency:

```bash
curl -s "https://<url>/hdr" -H "User-Agent: echo-client/1.0" \
  | python -c "import sys,json;print(json.dumps(json.load(sys.stdin)['headers'],indent=1))"
```

## Per-flag observed result

| Flag | What the origin sees | Client sees |
|---|---|---|
| `a:X-Test:v` | `X-Test: v` added | — |
| `u:Host:example.com` | `Host: example.com` | — |
| `r:Accept` | no `Accept` header | — |
| `x:fullurl` | `X-Pinggy-Url: https://<url>/<path>` | — |
| `x:noreverseproxy` | only the client's own headers | — |
| `x:xff` (alone) | identical to default — reverse-proxy mode already adds it | — |
| `k:key` | `Authorization: Bearer key` passed through | `401` + `WWW-Authenticate: Bearer realm="Pinggy Key Auth Failed"` without it |
| `b:u:p` | — | `401` + `WWW-Authenticate: Basic realm="Pinggy Basic Auth Failed"` |

## Read the flags back through the web debugger

The debugger reports the effective configuration, not just the URLs. It is the
fastest way to confirm a flag was accepted:

```bash
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L4301:localhost:4300 -R0:localhost:18001 free@a.pinggy.io -T -- "w:104.28.235.104,192.168.0.0/16"
```

```bash
curl -s http://127.0.0.1:4301/urls          # {"urls": [...]}
curl -s http://127.0.0.1:4301/ipwhitelist   # ["104.28.235.104/32", "192.168.0.0/16"]
```

A single-IP whitelist is normalised to a `/32` entry. An empty list means no
`w:` flag is active.

## Concurrent-tunnel probe

Free tier concurrency is not capped at one tunnel per source IP. To measure it,
launch N tunnels in parallel and probe each URL — a plain sequential loop gives
false negatives because each short-lived session closes before you probe:

```python
import subprocess, concurrent.futures as cf, re, time

def launch(i):
    p = subprocess.Popen(
        ["ssh", "-p", "443", "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=30",
         "-o", "ConnectTimeout=8", "-R0:localhost:18000", "free@a.pinggy.io", "-T"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url, t0 = None, time.time()
    while time.time() - t0 < 12:
        line = p.stdout.readline()
        if not line:
            break
        m = re.search(r'https://[a-z0-9-]+\.free\.pinggy\.net', line)
        if m:
            url = m.group(0)
            break
    return url, p

N = 8
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    res = list(ex.map(launch, range(N)))
urls = [u for u, _ in res if u]
alive = 0
for u in urls:
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                        "--max-time", "6", "-H", "User-Agent: probe/1.0", u + "/"],
                       capture_output=True, text=True, timeout=10)
    alive += (r.stdout.strip() == "200")
for _, p in res:
    p.kill()
print(f"created {len(urls)}/{N}, serving 200: {alive}/{len(urls)}")
```

Observed: 8/8. Note `ps -W | grep -c ssh` over-reports — count responding URLs,
not processes.

## Traps that produce a false "protected"

- **Host parity.** Pinggy terminates TLS and forwards by Host. The public URL and
  the local origin can both return `200` while talking to different backends.
  After a tunnel is torn down, re-probe the URL: a dead tunnel returns `000`
  (connection failure), not `200`. Never conclude from a single probe.
- **Wrong host, wrong session.** A `Permission denied (publickey,password)`
  means the tunnel never existed — usually a flag placed in the username
  instead of after `--`. Do not treat it as an auth-gate success.
- **`Unsupported argument provided.` + immediate close.** An unknown remote arg
  is fatal, not ignored. Check the arg spelling before retrying.
- **Screening page vs. origin.** Free tunnels inject a one-time browser
  interstitial. Compare body sizes, not just status codes: the interstitial is
  much larger than the origin's response. Send `X-Pinggy-No-Screen: 1` or a
  custom `User-Agent` to bypass.

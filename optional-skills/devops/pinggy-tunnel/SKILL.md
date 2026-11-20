---
name: pinggy-tunnel
description: Expose localhost via Pinggy SSH tunnel; flags after `--`.
version: 0.2.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Pinggy, Tunnel, Networking, SSH, Webhook, Localhost]
    related_skills: []
---

# Pinggy Tunnel Skill

Expose a local service (dev server, webhook receiver, MCP endpoint, demo) to the public internet using a Pinggy SSH reverse tunnel. No daemon to install — the user's stock SSH client connects to `a.pinggy.io:443` and Pinggy hands back a public HTTP/HTTPS URL.

Free tier: 60-minute tunnels, random subdomain, no signup. Pro tier is an opt-in with a token.

## Access-control syntax — READ THIS FIRST

**There are two different places flags can go, and only one of them works.**

| Mechanism | Where | Works? | Use for |
|---|---|---|---|
| **Remote args** | after `--` on the ssh command line | YES — this is the real API | `b:`, `k:`, `w:`, `a:`, `r:`, `u:`, `x:*` |
| Username keywords | inside the SSH username, joined with `+` | NO — silently ignored or `Permission denied` | tunnel TYPE only (`tcp@`, `tls@`, `tlstcp@`, `qr@`, `force@`) and the Pro token |

An earlier version of this skill (and the Hermes optional-skill page) documented `"k:token+free@a.pinggy.io"`. **That syntax does not work.** Verified: `k:…+free@a.pinggy.io`, `b:…+free@a.pinggy.io`, `co+free@a.pinggy.io`, `qr@free.pinggy.io` all fail — either the gate is silently not applied (request returns 200 with no auth challenge) or the connection is refused with `Permission denied (publickey,password)`. Do not use it.

Correct form:

```bash
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "b:admin:secret" "x:https"
```

`-T` is not optional in practice — it disables the interactive TUI so the URL lines land cleanly on stdout. Without it the session stays in the rich TUI (the tunnel still works, but the output is not parseable).

## When to Use

- User asks to "expose this locally", "share my dev server", "make this URL public", "tunnel port N", "get a public URL for a webhook"
- Need to receive a webhook callback during a local task (Stripe, GitHub, Discord, AgentMail)
- Sharing a one-off HTTP demo (MCP server, Ollama/vLLM endpoint, dashboard) with a remote party
- The host has SSH but no `cloudflared` / `ngrok` binary, and installing one would be overkill

If the host already has `cloudflared` configured, prefer the `cloudflared-quick-tunnel` skill — Cloudflare quick tunnels don't expire after 60 minutes.

## Prerequisites

- `ssh` on PATH (`ssh -V`). Default on Linux, macOS, and Windows 10+. No other install.
- A local service listening on `127.0.0.1:<port>` before the tunnel starts. Pinggy will return URLs but they'll 502 until the local origin is up.

Optional:

- `PINGGY_TOKEN` env var for paid Pro features (persistent subdomain, custom domain, multiple tunnels, no 60-minute cap). Free tier needs no credentials.

## Quick Reference

```bash
# Plain HTTP/HTTPS tunnel for port 8000 (free tier)
ssh -p 443 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R0:localhost:8000 free@a.pinggy.io -T

# Basic auth gate (b:user:pass)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "b:admin:secret"

# Bearer token gate (visitor sends: Authorization: Bearer <token>)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "k:mysecrettoken"

# IP whitelist (single IP or CIDR, comma-separated)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "w:203.0.113.0/24"

# Add / update / remove a request header
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "a:X-Request-Tag:pinggy"
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "u:Host:example.com"
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "r:Accept"

# Advanced x: options
#   x:https          force HTTPS (301 redirect from http://)
#   x:xff            add X-Forwarded-For (reverse-proxy mode already adds it by default)
#   x:fullurl        add X-Pinggy-Url with the original request URL
#   x:passpreflight  let CORS preflight (OPTIONS) through unmodified
#   x:noreverseproxy strip X-Forwarded-* / Forwarded headers
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 \
    free@a.pinggy.io -T -- "k:mykey" "x:passpreflight"

# TCP tunnel (databases, raw SSH, etc.) — returns tcp://host:port
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:5432 tcp@a.pinggy.io -T

# TLS tunnel (end-to-end: Pinggy forwards TLS opaque, origin's own cert is served)
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:443 tls@a.pinggy.io -T

# Pro tier (persistent URL, no 60-min cap) — token is the username, host is pro.pinggy.io
ssh -p 443 -o StrictHostKeyChecking=no -R0:localhost:8000 "$PINGGY_TOKEN@pro.pinggy.io" -T
```

Every example passes `-T`. Omit it only for an interactive human session that wants the rich TUI.

## Procedure — Start a Tunnel and Get the URL

The model SHOULD use the `terminal` tool. The tunnel must stay alive for the duration of the share, so run it as a background process and parse the public URL from stdout.

### 1. Confirm a local origin is up

```bash
curl -sI http://127.0.0.1:8000/ | head -1
# expect HTTP/1.x 200 (or any non-connection-refused response)
```

If nothing is listening yet, start it first (e.g. `python -m http.server 8000 --bind 127.0.0.1`). Pinggy will happily return a URL pointed at nothing — the user will see 502 until the origin comes up.

### 2. Launch the tunnel as a background process

Use `terminal(background=True)`. Do NOT wrap it in `nohup … &` — the Hermes terminal tool rejects shell-level backgrounding and you lose process tracking. Read the URL from `process(action='log', session_id=…)`.

```bash
ssh -p 443 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R0:localhost:8000 free@a.pinggy.io -T
```

`StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null` skips the first-run host-key prompt. `ServerAliveInterval=30` keeps the SSH session from getting torn down by an idle NAT.

### 3. Parse the URL out of the log

```bash
sleep 6
process(action='log', session_id='<id>')
```

Expected output:

```
Allocated port 9 for remote forward to localhost:8000
You are not authenticated.
Your tunnel will expire in 60 minutes. Upgrade to Pinggy Pro to get unrestricted tunnels.
https://lilnv-104-28-235-104.free.pinggy.net
https://ndejf-104-28-235-104.run.pinggy-free.link
```

Two URLs are printed. **Both are `https://` and both work.** The `.free.pinggy.net` one is canonical; `.run.pinggy-free.link` is the alternate. Parse with a regex covering both:

```bash
grep -oE 'https://[a-z0-9-]+\.(free\.pinggy\.net|run\.pinggy-free\.link)' <log> | head -1
```

Older docs show `*.pinggy.link`; current free-tier tunnels are on the two domains above. Handle both if you are reading someone else's log.

### 4. Verify

```bash
curl -sI https://<the-url>/ | head -3
# expect 200/302/whatever the local origin actually returns
```

If you get `502 Bad Gateway`, the SSH session is up but the local origin isn't listening — fix step 1 first.

### 5. Teardown

```
process(action='kill', session_id='<id>')
```

If the session handle is lost, kill by PID. **`pkill` does not exist in the git-bash/MSYS shell on Windows** — it fails with `command not found`, so the tunnel stays up:

```bash
# Windows / git-bash (MSYS)
ps -W | grep '/usr/bin/ssh' | awk '{print $1}' | xargs -r kill -9

# Linux / macOS
pkill -f 'ssh -p 443 .* free@a\.pinggy\.io'
```

## Access Control — Verified Behaviour

All flags are **remote args** (after `--`), quoted individually.

To *prove* a flag's effect rather than trusting the table — serve a header-echo origin and inspect what arrives, and read the effective config back from the web debugger (`/urls`, `/ipwhitelist`). Recipes and the observed per-flag results: `references/verifying-tunnel-flags.md`.

| Flag | Effect | Verified response |
|---|---|---|
| `b:user:pass` | HTTP Basic auth gate | `401` + `WWW-Authenticate: Basic realm="Pinggy Basic Auth Failed"` without credentials; `200` with correct ones |
| `k:token` | Bearer-token gate | `401` + `WWW-Authenticate: Bearer realm="Pinggy Key Auth Failed"` without/with wrong token; `200` with `Authorization: Bearer <token>` |
| `w:IP[,IP…]` | IP whitelist; requests from other IPs get **no response at all** (connection hangs / TLS handshake fails), not a 403 | silences the tunnel entirely for non-listed sources |
| `a:Name:Value` | append header to every request | header visible at the origin |
| `u:Name:Value` | remove then add header | `Host` is special: updatable, never removable |
| `r:Name` | remove header | header absent at the origin |
| `x:https` | force HTTPS | `301` from `http://` to `https://` |
| `x:passpreflight` | pass CORS preflight through unmodified | OPTIONS reaches the origin even under `k:` |
| `x:fullurl` | add `X-Pinggy-Url` with the original URL | verified present |
| `x:noreverseproxy` | strip `X-Forwarded-*` / `Forwarded` | verified absent |

Notes that matter:

- **Reverse-proxy mode is ON by default.** Every request already carries `X-Forwarded-For`, `X-Forwarded-Proto`, `X-Forwarded-Host`, and `Forwarded`. `x:xff` is a no-op unless you also pass `x:noreverseproxy`.
- **`b:` and `k:` can be combined, but they are OR — either gate alone is sufficient.** Verified: with both set, a request carrying only the bearer token returns `200`, and one carrying only the Basic credential also returns `200`; only a request with neither gets `401`. Stack multiple values of the same kind by repeating the flag (`k:key1 k:key2`, `b:u1:p1 b:u2:p2`, `w:IP1,IP2`).
- **`w:` evaluates the real client IP**, not a header the client controls. Sending `X-Forwarded-For: <listed IP>` does NOT get you in.
- **There is no `co` flag.** Neither as a remote arg (`Unsupported argument provided.` → connection closed) nor as a username keyword (ignored). Pinggy does not add `Access-Control-Allow-Origin`; if the origin needs CORS, set the header on the local server itself.
- **An unrecognised remote arg is fatal**, not ignored: the server prints `Unsupported argument provided.` and closes the connection. If a tunnel dies instantly, check the args.
- **Do not tunnel anything sensitive without `b:`, `k:`, or `w:`.** A bare HTTP tunnel is reachable by anyone who has the URL.
- **Prove a gate fires before reporting a URL as protected.** A `200` from the URL is not evidence of protection. Check the negative case: no credentials must return `401` with a `WWW-Authenticate` header, correct credentials `200`. If the negative case is also `200`, the flag never took effect — most often because it was put in the username instead of after `--`.
- **Re-probe a URL after tearing its tunnel down.** A genuinely dead tunnel returns `000` (connection failure), never `200`. Do not conclude anything about reachability from a single probe on a URL you have just killed.

## Web Debugger (optional)

Add a local forward to the SSH command:

```bash
ssh -p 443 -L4300:localhost:4300 -R0:localhost:8000 free@a.pinggy.io -T
```

Then open `http://localhost:4300` for live request/response pairs, or query the API:

```bash
curl http://localhost:4300/urls          # {"urls": ["https://…", "https://…"]}
curl http://localhost:4300/ipwhitelist   # ["104.28.235.104/32", "192.168.0.0/16"]
```

If `4300` is taken, map another local port: `-L9999:localhost:4300` and browse to that port. Use `-o ExitOnForwardFailure=yes` so a busy port fails loudly instead of silently dropping the debugger.

## Pitfalls

- **60-minute hard cap on the free tier.** The SSH session terminates at the 60-minute mark; the URL goes dead. For longer shares use a Pro token, or auto-restart with a shell loop. Note the URL changes on every restart.
- **Free-tier URL is random and changes on restart.** Don't bookmark it, don't paste it into a config file. Re-parse from the log each time.
- **Free tier does NOT limit you to one concurrent tunnel.** Verified 8/8 simultaneous free tunnels from the same source IP, all serving 200. The old "one tunnel per IP" claim is stale — but keep tunnel count sane anyway.
- **Free tunnels show a one-time browser screening page.** Only for real browser User-Agents. `curl`, webhook senders, and any custom `User-Agent` pass straight through; browsers can be sent `X-Pinggy-No-Screen: 1`. Verify with a size diff: the screening page is much larger than the origin's response.
- **`-T` for parseable output.** Without it the rich TUI takes over and the URL lines are not cleanly greppable.
- **Never use `nohup … &` with the Hermes terminal tool.** Pass `background=True` instead; the tool rejects shell-level backgrounding.
- **`process(action='log')` is the reliable way to read the URL.** Parsing a redirected logfile also works, but the session log is already captured for you.
- **TCP and TLS tunnels return a different shape.** TCP gives `tcp://<subdomain>.run.pinggy-free.link:<port>`. TLS gives `tls://<subdomain>.run.pinggy-free.link` with **no port**, and it is end-to-end: Pinggy forwards TLS opaque, so the origin's own certificate is what the client sees and Pinggy cannot read the traffic. Verified with a self-signed origin cert served through the tunnel. Don't assume every tunnel is an https URL.
- **Pro mode requires the token as the username**, and the host is `pro.pinggy.io`: `"$PINGGY_TOKEN@pro.pinggy.io"`. A persistent subdomain is assigned to the token in the dashboard, not via a flag.
- **Pinggy needs no SSH key.** A password prompt means the client fell back to password auth; press enter, or use `auth` as a username keyword if a client insists.

## Recipes

Composite patterns combining a local origin with a Pinggy tunnel. Each recipe is self-contained — start the origin, start the tunnel, parse the URL, hand it back to the user.

### Recipe 1 — Receive a webhook callback

Use this when an external service (Stripe, GitHub, Discord, AgentMail, etc.) needs to POST to a publicly reachable URL during a local task.

```bash
# 1. Tiny capturing server: every request gets appended to a hits log next to this file.
#    Write it to the Hermes scratch dir, then start with terminal(background=True).
cat >~/.hermes/cache/scratch/webhook-server.py <<'PY'
import http.server, json, datetime, pathlib
LOG = pathlib.Path("~/.hermes/cache/scratch/webhook-hits.log").expanduser()
class H(http.server.BaseHTTPRequestHandler):
    def _capture(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        rec = {"t": datetime.datetime.now().astimezone().isoformat(), "path": self.path,
               "method": self.command, "headers": dict(self.headers), "body": body}
        with LOG.open("a") as f: f.write(json.dumps(rec) + "\n")
        self.send_response(200); self.send_header("content-type","application/json")
        self.end_headers(); self.wfile.write(b'{"ok":true}\n')
    def do_GET(self): self._capture()
    def do_POST(self): self._capture()
    def log_message(self,*a,**k): pass
http.server.HTTPServer(("127.0.0.1", 18080), H).serve_forever()
PY
python ~/.hermes/cache/scratch/webhook-server.py    # terminal(background=True)

# 2. Tunnel — bearer-token-gate so randos can't pollute the capture log
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:18080 free@a.pinggy.io -T -- "k:$(openssl rand -hex 12)" \
    >~/.hermes/cache/scratch/webhook-pinggy.log 2>&1
# run with terminal(background=True), then read the URL from process(action='log')
sleep 6
URL=$(grep -oE 'https://[a-z0-9-]+\.(free\.pinggy\.net|run\.pinggy-free\.link)' ~/.hermes/cache/scratch/webhook-pinggy.log | head -1)
echo "Webhook URL: $URL"

# 3. While the agent works, watch hits land
tail -f ~/.hermes/cache/scratch/webhook-hits.log
```

Hand `$URL` to the service that needs to call you, plus `Authorization: Bearer <token>`. Teardown: `process(action='kill', …)` on both the tunnel and the capture server.

### Recipe 2 — Expose an MCP server over HTTP/SSE

Use when a remote MCP client (Claude Desktop on another machine, a teammate's editor, etc.) needs to reach an MCP server running on the local box. Only works for MCP servers that speak HTTP transport — stdio-mode servers can't be tunneled.

```bash
# 1. Start the MCP server in HTTP mode (example: a FastMCP server on port 8765)
python my_mcp_server.py --transport http --port 8765    # terminal(background=True)

# 2. Tunnel with a bearer token — MCP traffic should not be open to the internet
TOKEN=$(openssl rand -hex 16)
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:8765 free@a.pinggy.io -T -- "k:$TOKEN" \
    >~/.hermes/cache/scratch/mcp-pinggy.log 2>&1
# terminal(background=True); URL from process(action='log')
sleep 6
URL=$(grep -oE 'https://[a-z0-9-]+\.(free\.pinggy\.net|run\.pinggy-free\.link)' ~/.hermes/cache/scratch/mcp-pinggy.log | head -1)
echo "MCP URL: $URL"
echo "Bearer token: $TOKEN"
```

The remote client connects to `$URL` with `Authorization: Bearer $TOKEN`. Hermes' own native MCP client config: `{"transport": "http", "url": "<URL>", "headers": {"Authorization": "Bearer <TOKEN>"}}`.

Pinggy forwards the `Authorization` header unchanged, so the local server sees Pinggy's token. Let Pinggy do the gating and keep the local server on `127.0.0.1`.

### Recipe 3 — Expose a local LLM endpoint (Ollama / vLLM / llama.cpp)

Share a local model with a remote caller (another agent, a phone, a teammate). Ollama listens on `:11434`, vLLM and llama.cpp typically on `:8000`.

```bash
# Pre-req: the model server is already running on 127.0.0.1:11434 (Ollama default)
TOKEN=$(openssl rand -hex 16)
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:11434 free@a.pinggy.io -T -- "k:$TOKEN" \
    >~/.hermes/cache/scratch/llm-pinggy.log 2>&1
# terminal(background=True); URL from process(action='log')
sleep 6
URL=$(grep -oE 'https://[a-z0-9-]+\.(free\.pinggy\.net|run\.pinggy-free\.link)' ~/.hermes/cache/scratch/llm-pinggy.log | head -1)
echo "Endpoint: $URL"
echo "Token:    $TOKEN"

# Verify
curl -s "$URL/api/tags" -H "Authorization: Bearer $TOKEN" | head
```

For an OpenAI-compatible vLLM/llama.cpp endpoint, callers use base URL `$URL/v1` with `Authorization: Bearer $TOKEN`. Pinggy does not touch the body, and **it does not add CORS headers** — a browser caller needs the origin to emit `Access-Control-Allow-Origin` itself.

### Recipe 4 — Share a dev server with a one-shot password

The fastest "let a teammate poke at my running app" pattern. Random password, prints once, dies when you Ctrl-C.

```bash
PASS=$(openssl rand -base64 12 | tr -d '+/=' | head -c 12)
echo "Dev server password: $PASS"
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    -R0:localhost:3000 free@a.pinggy.io -T -- "b:dev:$PASS" "x:https"
# URL from process(action='log'). Share URL + password. Kill the session to tear down.
```

`b:dev:$PASS` gates the URL with HTTP Basic auth. `x:https` forces TLS. If the app is a browser SPA, set `Access-Control-Allow-Origin` on the local server — Pinggy won't add it.

## Verification

```bash
# origin — terminal(background=True)
python -m http.server 18000 --bind 127.0.0.1

# tunnel — terminal(background=True)
ssh -p 443 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -R0:localhost:18000 free@a.pinggy.io -T

sleep 6
# URL from process(action='log'); then:
curl -sI "$URL/" | head -1

# teardown
process(action='kill', session_id='<tunnel>')
process(action='kill', session_id='<origin>')
```

Expected: a `https://<sub>.free.pinggy.net` (or `.run.pinggy-free.link`) URL and `HTTP/1.1 200` on the curl head.

Stronger check that an auth gate actually fires — do this before reporting a URL as protected:

```bash
curl -s -o /dev/null -w "%{http_code}\n" "$URL/"                                  # expect 401
curl -s -o /dev/null -w "%{http_code}\n" "$URL/" -H "Authorization: Bearer $TOKEN"  # expect 200
```

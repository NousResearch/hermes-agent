---
name: mcp-oauth-remote-gateway
description: Manual OAuth for remote MCP servers on headless gateways.
version: 1.0.0
author: Ben Barclay (benbarclay), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [MCP, OAuth, PKCE, Remote-Deployment]
    related_skills: [hermes-agent, mcporter, fastmcp]
---

# MCP OAuth on a Remote Hermes Gateway

## Overview

Hermes' built-in MCP OAuth client runs a one-shot HTTP listener on `127.0.0.1:<port>`
inside the Hermes process and registers that loopback address as the OAuth
`redirect_uri`. That works perfectly for a local CLI on the user's own machine.
It breaks completely when Hermes runs as a remote gateway (container, VPS,
messaging bot), because the user's browser resolves `127.0.0.1` to the user's own
laptop, not the remote container — so the authorization code never reaches Hermes.

This skill does the OAuth dance by hand and writes the resulting tokens into the
exact files Hermes' token storage expects, so a subsequent `/reload-mcp` finds
cached tokens and skips the browser flow entirely.

## When to Use

Use this skill when **all** of the following are true:

1. The user wants to add a remote HTTP MCP server that requires OAuth (not a static Bearer token).
2. Hermes is running as a **remote gateway** (container, VPS, Docker, managed service) — NOT a local CLI on the user's laptop.
3. The server supports OAuth 2.1 with PKCE and RFC 7591 Dynamic Client Registration (most modern MCP servers do — Better Stack, Linear, Cloudflare, Datadog, etc.). If it doesn't support DCR (GitHub is the notable exception), this skill does not apply — use a pre-registered OAuth App or a Personal Access Token instead.

Do NOT use this for:
- **Local CLI Hermes** — just set `auth: oauth` in `mcp_servers.<name>` and `/reload-mcp`. The built-in flow opens a browser and captures the callback on localhost. Works perfectly.
- **Servers that accept a static Bearer token (API key)** — always prefer `headers.Authorization: "Bearer <token>"` when the user is willing. Simpler, no refresh dance.
- **GitHub Copilot MCP** (`api.githubcopilot.com/mcp/`) — GitHub does not expose DCR. Use a PAT or a pre-registered OAuth App (see pitfall 12).

## Why the Built-in OAuth Flow Fails on a Remote Gateway

Hermes' native MCP OAuth client (`tools/mcp_oauth.py`):

1. Picks a free local port `P`.
2. Registers a dynamic OAuth client with the AS, sending `redirect_uri = http://127.0.0.1:P/callback`.
3. Starts an HTTP server on `127.0.0.1:P` **inside the Hermes process**.
4. Prints the authorize URL and waits for the code at its local endpoint.

When Hermes runs remotely, the `127.0.0.1` in the `redirect_uri` is the remote
container's loopback, not the user's. After authorizing, the user's browser 302s
to `http://127.0.0.1:P/callback?code=...`, which resolves to the user's own
laptop and fails to connect. The callback never reaches the Hermes process, the
flow times out, and `/reload-mcp` returns "No MCP tools available" with no detail.

Symptoms to recognize: `[xdg-open] <defunct>` processes under the hermes user, an
empty or missing tokens directory (`$HERMES_HOME/mcp-tokens/`), and a reload that
responds without any "Added/Reconnected: X" line in `change_detail`.

## Cheap First Fallbacks: the Built-in Flow's Own Escape Hatches

Before any manual token surgery, check whether the built-in flow's fallbacks
already cover the deployment. When Hermes detects a remote session it prints two
options alongside the authorize URL (`tools/mcp_oauth.py`):

1. **Paste-back** — on an interactive TTY, a stdin reader races the HTTP
   listener. The user authorizes, the browser fails to connect to
   `127.0.0.1:<port>`, and they paste the full address-bar URL
   (`?code=...&state=...`) back at the prompt. Works for SSH'd-in CLI sessions.
2. **SSH port-forward** — `ssh -N -L <port>:127.0.0.1:<port> <user>@<host>`
   makes the redirect reach the remote listener normally.

Both require an interactive terminal to the Hermes host. The rest of this skill
is for when there is NO interactive TTY — Hermes running purely as a messaging
gateway/bot where `/reload-mcp` triggers the flow with nobody at a prompt.

## Preferred Front Door: the Hermes Dashboard (try this BEFORE manual token surgery)

A remote Hermes gateway often also runs the **dashboard** web UI as a SEPARATE
process (e.g. `hermes dashboard --host 0.0.0.0 --port <port>`; check with
`ps aux | grep 'hermes dashboard'`). It exposes a connector/MCP console —
endpoints like `/api/mcp/servers`, `/api/mcp/status`, and `/connectors` (all
login-gated; a cookieless curl returning 401/302 confirms they exist).

**Why the dashboard solves the core problem:** when the user drives OAuth from
the dashboard *in their own browser*, the redirect lands in a context the
dashboard can capture — sidestepping the `127.0.0.1`-callback failure that breaks
the CLI/manual flow. So the correct escalation order for "add or re-auth an OAuth
MCP server on a remote gateway" is:

1. **Dashboard, in the user's browser** — the intended front door. Add servers, run OAuth, reload, all authenticated as the user. No copy-paste-callback dance, no hand-writing token files.
2. **Manual token surgery (the rest of this skill)** — the FALLBACK for when there's no browser session to the dashboard (pure-chat/headless context).

**Finding the dashboard's PUBLIC URL.** The dashboard binds internally to
`0.0.0.0:<port>`, but the user needs the externally-reachable URL. Most deploy
platforms inject it into the environment — grep for it rather than making the
user hunt:

```bash
env | grep -iE "HERMES_DASHBOARD_PUBLIC_URL|RAILWAY_PUBLIC_DOMAIN|RAILWAY_STATIC_URL|RAILWAY_SERVICE_.*_URL|PUBLIC_URL|BASE_URL|DOMAIN" \
  | sed -E 's/(TOKEN|SECRET|KEY|PASSWORD)=.*/\1=***REDACTED***/I'
```

`HERMES_DASHBOARD_PUBLIC_URL` is authoritative when present. On Railway also check
`RAILWAY_PUBLIC_DOMAIN` / `RAILWAY_STATIC_URL` (the `*.up.railway.app` host) and
`RAILWAY_SERVICE_*_URL` vars, which sometimes carry a friendlier custom domain.
Hand the user the full `https://` URL and point them at the Connectors/MCP
section. ALWAYS pipe through the `sed` redaction above — these env greps sit next
to `*_TOKEN`/`*_SECRET` vars.

**What the dashboard does NOT fix (still host-side / shell):** stdio servers that
need shell auth state (a CLI `login` command whose credentials may not persist
across restarts) and anything reading credentials from `$HERMES_HOME/.env`. Those
are out of the dashboard's scope regardless.

## The Workaround

Do the OAuth dance manually, then write the resulting tokens into the exact files
Hermes' `HermesTokenStorage` would have written, so on `/reload-mcp` Hermes finds
cached tokens and skips the browser flow entirely.

Run the shell commands below through the `terminal` tool on the gateway host and
do the Python steps (PKCE generation, token exchange, file writes) via
`execute_code` or a `terminal` python3 invocation — file writes must happen in
the SAME code block as the token exchange (see pitfall 16).

### 1. Confirm it's a remote gateway

```bash
env | grep -iE "HERMES|RAILWAY|CONTAINER"
echo "$DISPLAY $WAYLAND_DISPLAY $SSH_CLIENT"
```

No display + a remote indicator = remote gateway. `tools/mcp_oauth.py::_can_open_browser()`
uses these same env vars, so if Hermes' own auto-detect says "headless", the
built-in flow won't work.

### 2. Find HERMES_HOME and the config path

```bash
HERMES_HOME=$(python3 -c 'from hermes_constants import get_hermes_home; print(get_hermes_home())')
echo "config: $HERMES_HOME/config.yaml"
echo "tokens: $HERMES_HOME/mcp-tokens/"
```

### 3. Discover OAuth metadata from the MCP server

MCP servers advertise their OAuth setup via RFC 9728 (OAuth 2.0 Protected
Resource Metadata). The `WWW-Authenticate` header on a 401 tells you where to look:

```bash
curl -sI https://mcp.example.com | grep -i www-authenticate
# → Bearer realm="mcp", resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"
```

**Not every server returns `WWW-Authenticate`.** Some return a bare
`{"errors":["Unauthorized"]}` 401 with no auth-discovery hint. When that happens,
probe well-known paths directly:

```bash
for p in \
  /.well-known/oauth-protected-resource \
  /.well-known/oauth-authorization-server \
  /.well-known/openid-configuration ; do
  echo "=== $p ==="
  curl -s -A "python-httpx/0.27" "https://mcp.example.com$p" | head -c 400; echo
done
```

Fetch the resource metadata to get `authorization_servers`, then fetch the AS's
`/.well-known/oauth-authorization-server` to get `authorization_endpoint`,
`token_endpoint`, and `registration_endpoint`.

Pitfall: many servers sit behind Cloudflare and 403 bare `urllib` user agents.
Always set `User-Agent: python-httpx/0.27` (or similar) on requests in this flow.

### 4. Dynamic Client Registration (RFC 7591)

POST to the `registration_endpoint` with:

```json
{
  "client_name": "Hermes Agent (manual OAuth)",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "<scopes_from_resource_metadata>"
}
```

Omit `scope` entirely if the AS's `scopes_supported` is empty — see step 5
pitfall. Use port `8765` (or any port — nothing will listen).
`token_endpoint_auth_method: none` marks this as a public PKCE client. Save the
returned `client_id`.

### 5. Build the authorize URL with PKCE

Generate:
- `code_verifier`: `secrets.token_urlsafe(64)[:128]`
- `code_challenge`: `base64url(sha256(code_verifier))` (no padding)
- `state`: `secrets.token_urlsafe(24)`

Query params: `response_type=code`, `client_id`, `redirect_uri`, `code_challenge`,
`code_challenge_method=S256`, `state`, plus `resource=<mcp_server_url>` (RFC 8707 —
many servers require this to bind the token to the specific MCP resource). Include
`scope=<space-separated>` ONLY if the AS metadata's `scopes_supported` is a
non-empty array AND/OR the resource metadata declares specific scopes. If
`scopes_supported: []`, omit the `scope` parameter — the server grants its full
default set on its own. Fabricating scope strings against an empty
`scopes_supported` can cause `invalid_scope` errors on some ASes.

**Stash `code_verifier` and `state` to disk** (e.g. `/tmp/.mcp-oauth-work/<server>.json`,
0600 perms). You need them for step 7, possibly across multiple chat turns.

### 6. Give the user the authorize URL

```
Open this URL in your browser:
<authorize_url>

After approving, your browser will try to load http://127.0.0.1:8765/callback
and fail to connect — THAT'S EXPECTED. Just copy the entire URL from the
address bar (it will contain ?code=...&state=...) and paste it back here.
```

### 7. Exchange the code for tokens

When the user pastes the callback URL:

1. Parse `code` and `state` from the query string.
2. **Verify `state` matches the stashed value** (CSRF check — do not skip).
3. POST `application/x-www-form-urlencoded` to the `token_endpoint`:
   - `grant_type=authorization_code`
   - `code=<from callback>`
   - `redirect_uri=<same as step 4>`
   - `client_id=<from step 4>`
   - `code_verifier=<stashed>`
   - `resource=<mcp_server_url>` (if the AS required it in step 5, include here too)
4. Response contains `access_token`, `refresh_token`, `token_type`, `expires_in`, `scope`.

### 8. Write tokens in Hermes' exact schema

`tools/mcp_oauth.py::HermesTokenStorage` expects two files under
`$HERMES_HOME/mcp-tokens/` (create dir with `0o700`, files with `0o600`):

**`<server_name>.json`** — the `OAuthToken` pydantic model:
```json
{
  "access_token": "...",
  "token_type": "Bearer",
  "expires_in": 7200,
  "refresh_token": "...",
  "scope": "read write"
}
```

**`<server_name>.client.json`** — the `OAuthClientInformationFull` model:
```json
{
  "client_id": "...",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "read write",
  "client_name": "..."
}
```

Write each file via `json.dumps(..., indent=2)`. Sanitize the filename with
`re.sub(r'[^\w\-]', '_', server_name)[:128]` — this matches `_safe_filename()` in
Hermes' token storage.

### 9. Add the server to config.yaml

```yaml
mcp_servers:
  <name>:
    url: "https://mcp.example.com"
    auth: oauth
    timeout: 180
    connect_timeout: 60
```

### 10. Smoke-test the token BEFORE asking the user to reload

Manually POST an MCP `initialize` request to confirm the token works end-to-end —
this catches scope misconfigurations, wrong `resource` values, and CF blocks
before the user is confused by another "No MCP tools available" reload:

```python
body = json.dumps({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "hermes-debug", "version": "1.0"},
    },
}).encode()
# POST to the MCP URL with:
#   Authorization: Bearer <access_token>
#   Accept: application/json, text/event-stream
#   Content-Type: application/json
#   MCP-Protocol-Version: 2025-06-18
#   User-Agent: python-httpx/0.27
```

Expect HTTP 200 with `Content-Type: text/event-stream` and a JSON-RPC result
containing `serverInfo` and `capabilities`. **Do not use `urllib` with its default
UA** — Cloudflare will 403 you even though Hermes (which uses httpx) will succeed.
`scripts/diagnose-oauth-mcp.py` automates this smoke test.

### 11. Tell the user to run `/reload-mcp`

On reload, Hermes sees `auth: oauth`, calls `HermesTokenStorage.get_tokens()`,
finds your cached tokens, skips the browser flow, and registers `mcp_<name>_*`
tools. Refresh happens automatically before `expires_in` elapses.

## Pitfalls & Lessons Learned

1. **Do not assume "headless" means "OAuth impossible."** The built-in flow works fine for local CLI; the issue is strictly remote deployments where the user's browser and the Hermes process are on different machines. Check the execution environment before claiming OAuth isn't an option.

2. **Read the source, not just the skill docs.** `tools/mcp_oauth.py` and the MCP config reference in `website/docs/` are the authoritative references. Grep the tree before telling the user a feature "doesn't exist."

3. **Cloudflare UA filter.** Many MCP/OAuth providers front their infra with Cloudflare, which 403s `python-urllib/*` user agents on metadata endpoints even though those endpoints are public. Set `User-Agent: python-httpx/0.27` (or any browser-like string) on every request in this flow. Hermes itself uses httpx, so this is never a problem in the real connection path.

4. **Include `resource` in both authorize and token requests.** RFC 8707 resource indicators are not optional for most modern MCP servers — they bind the issued token to the specific MCP resource URL. Leaving it out sometimes still works but may yield a token that later fails at the MCP server with a scope/audience error.

5. **Trailing slash matters.** Some servers advertise the resource as `https://mcp.example.com/` with a trailing slash and reject tokens issued against the no-slash variant. Copy the `resource` value verbatim from the `.well-known/oauth-protected-resource` response.

6. **`/reload-mcp` is silent on failure.** If the reload shows "No MCP tools available" with no `change_detail` line, a server is in config but failed to connect and no error bubbled up. Tail the error log, smoke-test the token directly with a manual `initialize` POST, and — if everything looks good — ask for a full process restart.

7. **Circuit breaker can survive `/reload-mcp`.** `tools/mcp_tool.py` keeps a module-level error-count dict with a small threshold. Once tripped (e.g. after token expiry produces several consecutive failures), the tool handler can short-circuit before calling the server, so no successful call resets the counter. Symptom: reload says "Reconnected: X" but subsequent calls still fail with "server unreachable" in the same conversation. Recovery order: try `/reload-mcp` FIRST (cheap, no chat-process blip) — on current builds it can clear the counter; only escalate to a full gateway process restart if a live call STILL short-circuits after reload. Do not lead with "you must restart."

8. **Refresh on an expired access_token + a tripped breaker is a deadlock.** The auto-refresh logic runs inside the MCP call path, which the breaker short-circuits once tripped. Manually refreshing the token on disk does not help by itself — pair a manual token refresh with a full restart, not a `/reload-mcp`.

9. **`invalid_grant` on a manual refresh means the refresh token is DEAD — re-auth is the only fix, do not loop.** When the access_token has been expired long enough, the refresh_token can also be revoked/expired server-side. A `grant_type=refresh_token` POST then returns HTTP 400 `{"error":"invalid_grant",...}` (wording varies: "Grant not found", "Token expired", "refresh token is invalid"). There is NO recovery from the gateway side. Hand back to the user with two options: (a) re-run the full manual OAuth dance (steps 3–10), or (b) if the provider offers a static personal API key, switch to that — no refresh/expiry cycle, more durable for an unattended remote gateway. Detect early: before any create/update operation against an OAuth MCP, check `expires_at` vs `time.time()`; if already expired, attempt the refresh first and surface `invalid_grant` immediately rather than failing mid-task.

10. **A successful refresh that STILL yields a rejected token = server-side SESSION revocation; only a fresh authorization_code flow fixes it.** Distinct from pitfall 9. The stored token file can look healthy (`expires_at` well out, refresh_token present), yet a live `initialize` POST returns `401 invalid_token` with a JSON-RPC body like `{"error":{"code":-32002,"message":"Session expired. Please re-authenticate."}}`. The `grant_type=refresh_token` POST may **succeed** (HTTP 200, new access_token) — yet the brand-new token gets the SAME `-32002`. The provider revoked the underlying MCP *session* server-side; the OAuth refresh chain re-mints credentials but cannot re-establish a revoked session. Decision rule when an OAuth MCP reports "not connected": (1) smoke-test the stored access_token with a manual `initialize` POST; (2) if `401 invalid_token`, attempt a refresh and smoke-test the NEW token; (3a) new token works → write it + restart to clear the breaker; (3b) new token STILL gets `-32002`/"Session expired" → stop, this is session revocation, hand the user the authorize URL for a full re-auth. `scripts/diagnose-oauth-mcp.py` automates steps 1–2 and prints which branch you're in. For an unattended gateway whose session keeps getting revoked, prefer a static Personal API key. See `references/stripe-mcp-oauth-revocation.md` for a worked example of a provider that revokes weekly.

11. **Client info file is NOT optional.** Hermes needs `<server>.client.json` to know the `client_id` for refresh grants. Skipping it means the first refresh fails and the user has to re-auth — writing both files is the whole point of this skill.

12. **Never hand-type the redirect URL for the user to open.** Generate the authorize URL programmatically with `urllib.parse.urlencode()`. Spaces in scopes and special chars in `state` break string-concatenated URLs.

13. **Security: the stash file contains the `code_verifier`.** Delete `/tmp/.mcp-oauth-work/<server>.json` immediately after successful token exchange. There's no reason to keep a proof-of-identity secret around once it's consumed.

14. **Write what the token endpoint actually returned.** The AS may grant a narrower (or wider) scope than requested. Write the `scope` from the token-exchange response to `<server>.json`, not what you asked for in step 5. When `scopes_supported: []`, the explicit scope list you send IS authoritative both ways: some servers grant exactly what you list (pass narrow scopes for least-privilege, or enumerate the full set if the user needs everything), and some won't echo the granted scope back at registration time — only the token-exchange response is authoritative.

15. **OAuth tokens often double as Bearer tokens against the provider's public REST API.** The access_token in `<server>.json` is frequently not "MCP-only" — `Authorization: Bearer <token>` against the provider's documented REST API succeeds whenever the corresponding resource scope was granted. This is the OAuth 2.0 spec, not a provider quirk. When the MCP server is read-only but you need a write operation, check whether the OAuth token can hit the provider's REST API directly before suggesting a separate API key.

16. **Secret redaction can mask tokens in tool output.** If secret redaction is enabled, tokens and long opaque strings render as `***` in tool-result output, so you cannot `print(response)` to keep the access_token visible across turns. Combined with single-use `code` values from authorization_code grants: if you print the token-exchange response, you may lose the token AND consume the code, forcing a restart with a fresh authorize URL. **Always write the access_token directly to its final destination file in the SAME code block that performs the token exchange.** If you must print for debugging, print only `len(access_token)`, `token_type`, `scope`, `expires_in` — never the secret.

17. **GitHub MCP (`api.githubcopilot.com/mcp/`) uses a pre-registered confidential OAuth App, not DCR + PKCE-public.** Its client info ships with a real `client_secret` and `token_endpoint_auth_method: client_secret_post`. The token-exchange POST to `https://github.com/login/oauth/access_token` must include `client_secret` as a form field alongside `client_id`, `code`, `code_verifier`, and `redirect_uri` (PKCE is still honored on top of the secret). The redirect URI is **fixed** in the OAuth App config — you cannot change it, so the manual listener-port trick doesn't apply; the user just lets the browser fail to connect on that port and pastes the address-bar URL back.

## What NOT to do

- **Don't use `mcp-remote` as a fallback.** It runs an npx subprocess whose OAuth callback server ALSO sits on the remote container's localhost — same problem. `mcp-remote` only helps when the MCP client doesn't speak remote HTTP at all (Hermes does natively).
- **Don't push "paste your API token and I'll add headers"** if the user explicitly asked for OAuth. Offer the static-token shortcut only after explaining why the native OAuth flow fails in remote deployments. Respect the user's choice to do the extra legwork for rotation-free, scope-limited access.
- **Don't claim Hermes doesn't support a feature without reading the source.** Grep the source tree before making capability claims.

## Quick Reference Files

- `scripts/diagnose-oauth-mcp.py` — re-runnable, read-only-by-default diagnostic. Given a server name, it smoke-tests the stored access_token, attempts a refresh, smoke-tests the new token, and prints exactly which recovery branch you're in (`TOKEN_OK` = breaker/restart, `REFRESH_FIXED` = persist+restart, `SESSION_REVOKED` = full re-auth, `REFRESH_DEAD` = full re-auth/API key). Pass `--write` to persist a working refreshed token atomically. Never prints secret values. **Run this FIRST when an OAuth MCP server reports "not connected"** — it encodes the pitfall 7/9/10 decision tree.
- `references/stripe-mcp-oauth-revocation.md` — a worked example (Stripe) of a provider that revokes its OAuth session on a recurring basis, and the durable fix: switch to a static restricted API key.

## Related

- `native-mcp` — general guide to configuring MCP in Hermes. Authoritative config reference lives there.
- `mcporter` — the external CLI bridge, for ad-hoc MCP calls outside of Hermes' config.

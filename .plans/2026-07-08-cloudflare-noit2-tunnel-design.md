# Cloudflare Tunnel exposure for the noit2.com platform — design

**Date:** 2026-07-08
**Status:** Approved (design), pending implementation plan
**Owner:** platform / hermes-agent

## 1. Purpose

A user who builds something on the hermes-agent platform — an app, a service, an API —
can expose what they built to the internet on their own `noit2.com` subdomain, either to
show it to the world or to consume the service themselves. The exposure is **ephemeral test
by default**, guarded by a mandatory auto-close protocol so that a forgotten test
environment can never stay exposed to the internet indefinitely. Longer exposure requires
explicit admin approval.

The hermes **dashboard is not the focus**. It is one possible local origin among many. The
thing that matters is a safe protocol around exposing arbitrary user-built origins to the
internet, with a dead-man's switch as the core feature rather than an afterthought.

## 2. Requirements

- Expose **arbitrary local origins** (any `host:port` the user's built service listens on)
  to the internet, not a fixed dashboard port.
- **Multiple services per user at once** on separate subdomains, served by one `cloudflared`
  with multiple ingress rules.
- HTTPS on port **443 at Cloudflare's edge**; the origin listens on `127.0.0.1` only. **No
  port opens on the host firewall.** The tunnel connects outbound to Cloudflare.
- **Dead-man's switch (centerpiece):** idle-reset 30-minute timer. The timer resets on
  incoming traffic; 30 minutes with no traffic closes the tunnel. This is the safety
  protocol for "exposed test env must not leak forever."
- **Hold-open requires admin approval.** A user cannot hold a tunnel open indefinitely on
  their own. They file a live hold request; an admin approves (with an expiry) or denies;
  the running tunnel extends to the approved expiry and then falls back to the idle timer.
- Per-user subdomains under the shared `noit2.com` Cloudflare zone.
- `noit2.com` root remains the Cloudflare Pages brand site (already configured by the
  operator; out of scope for code).

## 3. Hostname layout

| Hostname | Served by | Origin |
|---|---|---|
| `noit2.com` (root) | Cloudflare Pages | brand/landing site (operator-managed, out of scope) |
| `<user>.noit2.com` | Cloudflare Tunnel | user's built app/API on `127.0.0.1:<port>` |
| `<user>-<thing>.noit2.com` | Cloudflare Tunnel | a second service the same user is exposing |

All tunnel subdomains live in the single `noit2.com` Cloudflare zone. Each user runs their
own `cloudflared` with their own tunnel credentials; ingress rules map each exposed
subdomain to a local origin port.

## 4. Approach

First-class `hermes tunnel` subcommand, in-tree, following the existing
`hermes_cli/subcommands/` pattern (dashboard.py, gateway.py, webhook.py). The dead-man's
switch and the approvals store are small focused modules so they can be unit-tested
independently.

Rejected alternatives:
- **Standalone wrapper scripts** (`scripts/tunnel-up.ps1` / `.sh`) — minimal but not
  integrated, duplicated across shells, untestable, doesn't feel like part of the product.
- **Cloudflared-agnostic tunnel abstraction** (could also drive pinggy/ngrok later) —
  YAGNI for now; more abstraction than the task needs. The repo already ships a
  pinggy-tunnel *skill*, which remains orthogonal.

## 5. Components

### 5.1 `hermes_cli/subcommands/tunnel.py` — CLI surface

Commands:

- `hermes tunnel up [--origin <sub>=<host:port>]... [--hold-request [--reason <text>] [--until <time>]]`
  - Validate config (zone, tunnel_name, credentials file exists, `cloudflared` on PATH).
  - For each `--origin` mapping (and/or the `tunnel.routes` config list, CLI wins on
    conflict), ensure a DNS CNAME exists (`cloudflared tunnel route dns <name>
    <sub>.<zone>`) and generate a per-run cloudflared ingress config mapping
    `<sub>.<zone> -> http://localhost:<port>` with a `404` catchall.
  - If a dashboard origin is among the routes, set `dashboard.public_url =
    https://<sub>.<zone>` (and `HERMES_DASHBOARD_PUBLIC_URL`) so OAuth callback + WebSocket
    URLs build from the public hostname. Otherwise leave dashboard config untouched.
  - Launch `cloudflared tunnel --config <generated> run <tunnel_name> --metrics
    127.0.0.1:<metrics_port>` as a supervised child process.
  - Arm the dead-man's switch (see 5.2). Optionally file a hold request immediately.

- `hermes tunnel down [--kill-origins]` — graceful drain + kill cloudflared for this
  profile. Origins keep running locally unless `--kill-origins`.

- `hermes tunnel status` — running state, active origins, idle-since, time-to-close, public
  URLs, pending hold request (if any).

- `hermes tunnel doctor` — `cloudflared` present + reachable, credentials file valid,
  origins up, DNS CNAMEs resolve.

- `hermes tunnel hold [--reason <text>] [--until <time>]` — file a live hold request for
  the running tunnel.

- `hermes tunnel requests` — list pending hold requests (operator/admin view).

- `hermes tunnel approve <id> --until <time>` / `hermes tunnel deny <id> [--reason]` —
  admin resolution. Gated to identities in `tunnel.admin`.

### 5.2 `hermes_cli/tunnel_supervisor.py` — dead-man's switch (centerpiece)

Responsibilities:
- Launch and own the `cloudflared` child process.
- Poll `http://127.0.0.1:<metrics_port>/metrics` every `poll_interval_seconds` (default 5)
  and read cloudflared's request counter.
- **Idle-reset timer:** any increase in the counter since the last poll resets the
  `idle_timeout_seconds` (default 1800) idle clock. If `idle_timeout_seconds` elapse with
  no counter increase, gracefully drain then kill cloudflared.
- **Hold extension:** also poll the approvals store; when this tunnel's hold request is
  approved with an expiry, switch to "hold until <approved-until>" (timer disabled until
  then). When the approval expiry passes, fall back to the idle timer (do not hard-kill).
- **Graceful drain:** on close (idle expiry, Ctrl+C, or `down`), stop accepting new
  connections and let in-flight requests finish up to `drain_seconds` (default 15), then
  kill cloudflared. Origins are left running locally.
- Expose `status` state for the `hermes tunnel status` command.

**Policy hook (user-contributed):** the module exposes a pure function
`should_close_now(state) -> bool` (and a `reset_idle_on(state, prev_counter, cur_counter)
-> bool` helper) that encodes the close policy — poll cadence interaction, what counts as
activity, the hold-vs-idle precedence, and the drain trigger. This is the 5–10-line
business-logic decision that shapes daily behavior; it is scaffolded with a clear
signature and left for the user to implement during the build.

### 5.3 `hermes_cli/tunnel_approvals.py` — hold-request / approval store

- JSONL at `~/.hermes/tunnel/hold_requests.jsonl`.
- Record shape: `id`, `user`, `subdomains`, `reason`, `requested_until`, `status`
  (`pending` / `approved` / `denied`), `approved_until`, `decided_by`, `created_at`,
  `decided_at`.
- API: `file_request(...)`, `list_pending()`, `approve(id, until, by)`, `deny(id, reason,
  by)`, `is_approved(id)`, `approved_until(id)`. All mutating admin operations are gated to
  `tunnel.admin` identities.
- Append-only writes; status transitions validated (`pending -> approved|denied` only).
- No external service. The dashboard may render this store later (out of scope here).

### 5.4 Config + env

New `tunnel:` block in `cli-config.yaml.example`, honored by the config loader, with env
overrides that win (matching the repo's established precedence):

```yaml
tunnel:
  enabled: false
  zone: "noit2.com"
  tunnel_name: ""              # cloudflared named tunnel
  credentials_file: ""         # path to <uuid>.json
  metrics_port: 0              # 0 = pick a free port
  idle_timeout_seconds: 1800   # 30 min
  drain_seconds: 15
  poll_interval_seconds: 5
  admin: []                    # identities permitted to approve/deny hold requests
  routes: []                   # list of {subdomain: str, host: str, port: int}
```

Env: `HERMES_TUNNEL_ZONE`, `HERMES_TUNNEL_NAME`, `HERMES_TUNNEL_CREDS`,
`HERMES_TUNNEL_METRICS_PORT`, `HERMES_TUNNEL_IDLE_TIMEOUT`, `HERMES_TUNNEL_DRAIN_SECONDS`,
`HERMES_TUNNEL_POLL_INTERVAL`, `HERMES_TUNNEL_ADMIN` (comma-separated),
`HERMES_TUNNEL_HOLD_REQUEST` (for `up --hold-request`).

### 5.5 Testing

- `tests/hermes_cli/test_tunnel_supervisor.py` — pure-policy tests with a mocked counter
  sequence: active traffic resets the idle clock; sustained idle expires; hold-approved
  extends past the idle window; hold-denied closes on schedule; approval expiry falls back
  to idle (no hard kill); drain triggers on close. No real cloudflared.
- `tests/hermes_cli/test_tunnel_approvals.py` — store round-trips, status transition
  validation, expiry enforcement, admin-only mutation (non-admin approve/deny rejected).
- `tests/hermes_cli/test_tunnel_config.py` — config/env precedence, route merging
  (CLI over config), defaults.
- Integration with a real `cloudflared` + credentials is out of scope for CI; documented as
  a manual runbook in the docs.

### 5.6 Docs

`website/docs/user-guide/tunnel.md` — prerequisites (install `cloudflared`, create a named
tunnel, `cloudflared tunnel route dns` per subdomain, credentials file location), the
`tunnel:` config block, the `hermes tunnel up/down/status/doctor/hold/requests/approve/deny`
reference, the idle-reset behavior + admin-approved hold flow, and the Pages-on-root /
services-on-subdomains layout.

## 6. Data flow

```
user: hermes tunnel up --origin alice=127.0.0.1:3000 --origin alice-api=127.0.0.1:8080
  -> tunnel.py validates config + cloudflared present + creds file
  -> ensures DNS CNAMEs for alice.noit2.com, alice-api.noit2.com
  -> generates per-run cloudflared config (ingress -> 127.0.0.1 ports, 404 catchall)
  -> [if dashboard origin present] sets dashboard.public_url
  -> tunnel_supervisor launches cloudflared ... run --metrics 127.0.0.1:<p>
  -> supervisor polls metrics counter every 5s:
       counter increased -> reset 30-min idle clock
       30 min no increase -> drain 15s -> kill cloudflared (origins keep running)
  -> supervisor polls approvals store:
       hold request approved with expiry -> hold until expiry, then resume idle timer
       hold denied or unapproved -> close on idle schedule

user (mid-demo): hermes tunnel hold --reason "demo running long" --until 4h
  -> tunnel_approvals.file_request(...) -> record pending in hold_requests.jsonl

admin: hermes tunnel requests  -> sees pending
       hermes tunnel approve <id> --until 4h
  -> tunnel_approvals.approve(...) -> status=approved, approved_until set
  -> supervisor picks up approval on next poll -> extends

internet -> https://alice.noit2.com (443 at edge) -> cloudflared -> 127.0.0.1:3000
```

## 7. Security posture

- Origins bind to `127.0.0.1` only; the host firewall opens no port. The only public path
  is through Cloudflare's edge, which terminates TLS on 443.
- The idle-reset 30-min timer is the default safety net: a forgotten test env cannot leak
  indefinitely.
- Hold-open is not a user right — it requires admin approval with a bounded expiry, after
  which the idle timer resumes. There is no unbounded user-controlled exposure.
- Cloudflare Access (WAF / service tokens / zero-trust policies on the tunnel hostname) is
  operator-configured in the Cloudflare dashboard and orthogonal to this code; `hermes
  tunnel doctor` checks that the CNAME resolves but does not manage Access policies.

## 8. Out of scope

- The Pages brand site on `noit2.com` root (operator-managed in Cloudflare).
- Cloudflare Access / WAF / service-token policy configuration (operator dashboard).
- Per-user tunnel *provisioning* automation (creating tunnels + issuing credentials). The
  operator has already configured Cloudflare; this design assumes credentials exist.
- Dashboard UI for approvals (CLI-first now; dashboard render of the store is a later
  follow-up).
- A hard wall-clock cap on approved holds (e.g. 24h backstop) — not requested; can be a
  future `tunnel.max_hold_seconds` knob.

## 9. Open questions for implementation plan

- Exact cloudflared metrics counter name(s) to read (verify against the installed
  `cloudflared` version during build; the supervisor will probe `/metrics` and tolerate
  either the per-request or per-connection counter).
- How `tunnel.admin` identities are determined (profile name? explicit list?) — default to
  the explicit `admin:` list, resolved against the current profile.
- Whether `hermes tunnel up` should auto-start an origin that isn't running yet (no —
  origins are user-built services; the user starts them; `up` only reports the origin as
  down via `doctor`/`status`).
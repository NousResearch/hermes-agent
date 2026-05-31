# Hetzner Infrastructure — Status & TODO

**Server:** `178.105.246.8` (CX33, `alyosha`)
**Last updated:** May 30, 2026

## Services

| Service | Port | Process | Status | Auth |
|---------|------|---------|--------|------|
| Hermes Gateway | — | systemd `hermes-gateway.service` | ✅ Running | Telegram bot token |
| OpenClaw | 18789 | nohup (`openclaw gateway`) | ✅ Running | Bot token |
| Research Substrate v0.7.2 | 4020 (tunnel only) | systemd `research-substrate.service` | ✅ Running | API key + rate limit (demo: 30 req/hr) |
| Camofox Browser | 9377 | nohup (`camofox-browser`) | ✅ Running | `X-Api-Key` header |
| SearXNG | 8080 | internal only | ✅ Running | `X-Api-Key` header |
| LLM Ask Shim | 4010 | uvicorn | ✅ Running | Internal only (127.0.0.1) |

## Firewall

**Hetzner Cloud Firewall** `hermes-fw-v3` (ID 11053687)

| Direction | Protocol | Port | Source |
|-----------|----------|------|--------|
| in | tcp | 22 | 0.0.0.0/0, ::/0 |
| in | tcp | 80 | 0.0.0.0/0, ::/0 |
| in | tcp | 443 | 0.0.0.0/0, ::/0 |
| in | icmp | — | 0.0.0.0/0, ::/0 |

**Cloudflare Tunnel:** `research.alyoechosys.dev` → `localhost:4020` (systemd `cloudflared.service`)
- Tunnel ID: `50d10034-c7ac-4493-9de9-c196e2a29721`
- Config: `/etc/cloudflared/config.yml` + `/home/alyosha/.cloudflared/config.yml`
- Credentials: `/etc/cloudflared/50d10034-c7ac-4493-9de9-c196e2a29721.json`

**Notes:**
- No UFW installed. Hetzner cloud firewall is the only filter.
- Port 4020 closed externally — only reachable via Cloudflare Tunnel or localhost.
- Port 9377 (camofox), 18789 (OpenClaw) — blocked at cloud level, safe.
- **API token:** `HCLOUD_TOKEN` in `.env` on Hetzner — required for firewall management.

## Completed

### May 30, 2026

- [x] **Research substrate patches (v0.5 → v0.6 → v0.7)**
  - Option-chain auto-fetch nearest expiration date (no `expiration_date` required)
  - Stock-index auto-reroute (SPX, ^GSPC, DJI, etc. → Yahoo stock-info)
  - Search fallback chain: MiniMax → Serper → DDG → Brave → SearXNG (triggers on empty results + errors)
  - 77 search queries stress-tested across 16 categories — 0 unresolved
  - 39 endpoints tested: 34 working, 3 empty (expected), 1 schema issue
  - Docs: `docs/research-substrate-endpoints.md`, `docs/web-search-fallback.md`

- [x] **API key auth on substrate** — `X-API-Key` support
  - `SUBSTRATE_API_KEY` in `/home/alyosha/workspace/research-substrate/.env`
  - Live public without key: `/docs`, `/openapi.json`, `/healthz`, `/search`, `/providers`, `/v1/models`
  - Keep private/rate-limited endpoint examples keyed unless live checks prove public access

- [x] **Hetzner cloud firewall** — replaced `hermes-basic-firewall` with `hermes-fw-v3`
  - Old firewall silently dropped port 4020 on PUT — created new one instead
  - Applied to server ID 133859693

- [x] **Research substrate service** — running under systemd as `research-substrate.service` behind Cloudflare Tunnel

- [x] **File permissions** — `chmod 600` on all `.env` files and `config.yaml`

- [x] **Serper plugin** — `plugins/web/serper/` (provider.py, plugin.yaml, __init__.py)
  - Local only (not deployed to Hetzner yet — Hetzner uses SearXNG as fallback)

- [x] **Docs synced** — both doc files on Hetzner at `~/workspace/hermes-agent/docs/`

## TODO

### Cloudflare Tunnel + Domain (priority: ~~high~~ DONE)

- [x] Buy domain (Cloudflare Registrar recommended — at-cost pricing)
  - `.dev` ~$12/yr, `.com` ~$10/yr
  - Something neutral, not project-specific
- [x] Add site to Cloudflare (free plan)
- [x] Install `cloudflared` on Hetzner
- [x] Create tunnel: `research.alyoechosys.dev` → `localhost:4020`
- [x] Configure systemd service for `cloudflared`
- [x] Close port 4020 on Hetzner firewall
- [ ] Verify substrate binds to `127.0.0.1` only (belt-and-suspenders behind tunnel)
- [ ] Optional: Cloudflare Access policy (Zero Trust, email-based, free ≤50 users)

### Service Hardening (priority: medium)

- [ ] Create systemd unit for OpenClaw (currently nohup)
- [ ] Create systemd unit for camofox (currently nohup)
- [ ] Bind camofox to `127.0.0.1` instead of `0.0.0.0` (belt-and-suspenders, already firewalled)
- [ ] Install UFW as second-layer defense
- [ ] Set up logrotate for substrate logs

### Hermes Agent (priority: medium)

- [ ] Deploy serper plugin to Hetzner
- [ ] Verify Hermes search fallback chain works end-to-end on Hetzner
- [ ] Wire substrate API key into Hermes config for internal calls

### Monitoring (priority: low)

- [ ] Health check endpoints for each service
- [ ] Simple uptime monitoring (cron + curl + Telegram alert?)
- [ ] Disk/memory alerts (Hetzner CX33 has 8GB RAM)

## Key Paths on Hetzner

```
/home/alyosha/
├── .hermes/
│   ├── config.yaml          # Hermes config (600)
│   └── .env                 # API keys (600)
├── workspace/
│   ├── hermes-agent/        # Hermes repo
│   │   └── docs/            # This doc + endpoint references
│   └── research-substrate/  # Substrate code
│       ├── .env             # SUBSTRATE_API_KEY, SUPADATA_API_KEY, KIMI_API_KEY (600)
│       └── research_substrate/
│           └── app.py       # Main app (v0.7.2, ~1400 LOC)
└── .local/share/hermes-node/  # Camofox browser
```

## API Token Access

- **Hetzner Cloud API:** token in `.env` on Hetzner as `HCLOUD_TOKEN`
  - Manage firewall: `curl -H "Authorization: Bearer $HCLOUD_TOKEN" https://api.hetzner.cloud/v1/firewalls`
  - Server ID: 133859693
  - Firewall ID: 11053613
- **Substrate API key:** `SUBSTRATE_API_KEY` in substrate `.env`
  - Header: `X-API-Key: <key>`
  - Docs are open: `http://178.105.246.8:4020/docs`

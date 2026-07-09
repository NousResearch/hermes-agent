# IX Agency

IX Agency is the Intelliverse X build of the Hermes Agent desktop app: the
stock `apps/desktop` shell rebranded for the org, with an agency workspace
(clients, engagements, billing, org skill/tool catalogs), shipping the
`ivx-*` agent skills, wired to the self-hosted Firecrawl instance for web
research, and connected to the company WireGuard VPN (`usa-vpn`) for
reaching org backends privately.

This directory holds the org-specific pieces. Everything else is tracked
upstream (`NousResearch/hermes-agent`); keep the fork current with:

```bash
gh repo sync intelli-verse-x/hermes-agent --source NousResearch/hermes-agent --branch main
```

## What is rebranded (and what deliberately is not)

Display identity is IX Agency; internal mechanics stay `Hermes` so the
updater, relaunch, and uninstall flows keep working across upstream syncs:

| Surface | Value |
| --- | --- |
| Window titles, menu bar, About panel, notifications | IX Agency (`APP_NAME` in `apps/desktop/electron/main.ts`) |
| macOS Finder display name (`CFBundleDisplayName`), DMG title | IX Agency |
| Windows Start Menu shortcut, uninstaller display name | IX Agency |
| Installer artifacts | `IX-Agency-<version>-<os>-<arch>.*` |
| Locale strings ("Hermes Desktop") in en/ja/zh/zh-Hant | IX Agency |
| Bundle/executable (`Hermes.app`, `Hermes.exe`), `productName`, `appId` | unchanged — update/uninstall machinery globs these paths |
| References to the Hermes backend/CLI (`hermes update`, "Hermes backend") | unchanged — that is the runtime's real name |

## Build the desktop app

```bash
npm run install:desktop          # from the repo root
cd apps/desktop
npm run dev                      # develop against a local backend
npm run dist:mac                 # or dist:win / dist:linux for installers
```

## Org skills

The five org skills live in `skills/` alongside the bundled ones and are
seeded into `~/.hermes/skills/` automatically by the installer and
`hermes update` (via `tools/skills_sync.py`):

- `ivx-agent-vault` — runtime secrets from Infisical
- `ivx-content-factory` — Content Factory MCP usage
- `ivx-gastown-bridge` — talking to Gas Town from outside the Mayor session
- `ivx-products-tour` — product portfolio inventory
- `ivx-stack-tour` — the 5-layer intelli-verse-x stack

They use the open Agent Skills format (SKILL.md), so the same folders work in
Cursor, Claude Code, Codex, and Goose.

## Web research: self-hosted Firecrawl

Two integration paths, both configured in `config.example.yaml`:

1. Native `web` toolset — set `web.backend: firecrawl` in
   `~/.hermes/config.yaml` and `FIRECRAWL_API_URL` in `~/.hermes/.env`
   (see the repo root `.env.example`). The bundled `plugins/web/firecrawl`
   plugin then routes `web_search` / `web_extract` to the org instance.
2. MCP — the `firecrawl` entry under `mcp_servers` runs the MIT-licensed
   `firecrawl-mcp` server for crawl/deep-research tools.

Do not fork Firecrawl's core into this repo: it is AGPL-3.0. Run it as a
separate service and talk to it over the API, which keeps this fork's
licensing MIT-clean.

## The IX Agency workspace (sidebar → IX Agency)

`apps/desktop/src/app/ix-agency/` adds an agency workspace to the app:

| Tab | What it holds |
| --- | --- |
| Copilot | Native admin copilot — LiteLLM chat with admin-mcp tools, skill pills, and a write-action confirmation gate |
| Clients | Accounts the agency serves — status (lead/active/paused/churned), contacts, notes |
| Engagements | Projects and retainers per client — billing model, value, dates |
| Billing | Invoices with outstanding/overdue/paid-this-year totals and mark-paid |
| Org skills | The Intelliverse portal's admin-copilot playbooks plus your own drafts, injectable into Copilot chats |
| Org tools | The admin-mcp MCP tile directory (bundled snapshot, live via gateway token) |
| Connect | One-click company VPN (WireGuard) + portal / admin-mcp gateway settings |

Clients, engagements, and invoices are local-first: they persist on the
machine (localStorage, `hermes.desktop.ixAgency.book`) with no server
dependency. The gateway bearer token entered under Connect is encrypted at
rest with Electron `safeStorage` (`userData/ix-agency.json`).

`apps/desktop/scripts/generate-ix-agency-data.mjs` regenerates the bundled
skill/tile snapshots from sibling checkouts of `Intelliverse-X-Webfrontend`
and `intelli-verse-kube-infra` (override with `IX_FRONTEND_DIR` /
`IX_INFRA_DIR`).

## VPN: the company WireGuard (`usa-vpn`)

The org's VPN is the existing `usa-vpn` deployment (see
`intelli-verse-kube-infra/usa-vpn/`): WireGuard via wg-easy v15 on
Lightsail, endpoint `3.224.15.124:51820`, full tunnel, one `.conf` profile
per employee. The EKS-hosted admin surfaces (Cloudflare Access / IP
allowlists pinned to the VPN exit IP) become reachable once connected.

Desktop integration — the Connect tab drives the tunnel directly:

1. Get your profile: admins mint one per employee with
   `usa-vpn/create-clients.sh` (or the wg-easy admin UI) and hand you
   `<name>/usa-vpn.conf` — see `usa-vpn/EMPLOYEE-GUIDE.md`.
2. Install WireGuard tooling: `brew install wireguard-tools` (macOS),
   `apt install wireguard-tools` (Linux), or WireGuard for Windows.
3. In IX Agency → Connect, browse to the `.conf` and hit Connect. macOS
   prompts for your admin password (wg-quick needs root); Windows raises a
   real UAC prompt; Linux needs a passwordless sudoers rule for wg-quick.
   Status polls without elevation and shows connected/disconnected live.

### Remote Hermes backend over the VPN

The desktop app runs a local backend by default and works fully offline.
For a shared org backend (k8s via `intelli-verse-x/hermes-deployment` on
EKS, namespace `aicart`, host `hermes.intelli-verse-x.ai`):

1. Backend — pin a session token so desktop Remote Gateway connections
   survive restarts:

   ```bash
   # in ~/.hermes/.env on the backend host
   HERMES_DASHBOARD_SESSION_TOKEN=<long-random-token>

   hermes dashboard --host 0.0.0.0 --port 9119 --insecure --tui --no-open
   ```

   `--insecure` here means session-token auth instead of the Nous Portal
   OAuth gate; keep the ingress restricted (Cloudflare Access or an IP
   allowlist pinned to the VPN exit IP `3.224.15.124`) — the dashboard
   must never be reachable from the open internet.

2. Desktop — connect the VPN (Connect tab), then Settings → Gateway →
   Remote gateway, enter the backend URL and the session token.

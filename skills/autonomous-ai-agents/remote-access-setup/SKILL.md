---
name: remote-access-setup
description: "Use when setting up Hermes remote access securely."
version: 1.1.0
author: max-bitnode, Hermes Agent
license: MIT
platforms: [linux, windows]
metadata:
  hermes:
    tags: [hermes, remote-access, android, dashboard]
    category: autonomous-ai-agents
    related_skills: [hermes-agent]
---

# Hermes Remote Access Setup

Configures a password-gated Hermes Dashboard for trusted LAN/VPN access or an HTTPS reverse proxy. It does not configure a third-party proxy or DNS provider unless the corresponding management tools are available.

## When to Use

Use for a fresh or existing Hermes installation that must be reachable from Hy4ri/Hermes Mobile, Hermes Desktop, or a browser over LAN, VPN, or an authenticated HTTPS reverse proxy.

## Prerequisites

- Load the `hermes-agent` skill and current Dashboard documentation.
- Confirm `hermes --version` and `hermes doctor` succeed before making changes.
- For HTTPS, obtain the complete public URL and the exact reverse-proxy source IP/CIDR.

## How to Run

Use `terminal` for prerequisite checks and launch the skill-relative helper in a user-visible terminal. On Windows, keep the console open so the exit code remains visible.

## Quick Reference

- Confirm the exact client. `Hy4ri/hermes-mobile` connects to the Dashboard origin (normally port 9119), not the OpenAI-compatible API on 8642.
- Ask only for non-secret choices in chat. Never request a password in chat. Password entry must happen locally through the interactive helper using `getpass`.
- Never print passwords, hashes, signing secrets, cookies, or WebSocket tickets.
- Use HTTPS for Internet-facing access. Basic Auth alone is suitable only behind a trusted LAN/VPN or an additional access layer; recommend OIDC/Nous OAuth for direct public exposure.
- Do not rotate an existing `HERMES_DASHBOARD_BASIC_AUTH_SECRET` unless explicitly requested; rotation invalidates sessions.
- Back up changed files and append a concise entry to `~/.hermes/CHANGELOG.md`.
- One machine dashboard manages all normal Hermes profiles. Do not create one connection or port per profile unless the user explicitly requests isolated dashboards.
- Do not claim success until both the public/LAN `/api/status` response and an authenticated WebSocket-ticket request have been verified. If the reverse proxy is not configured or inaccessible, report backend-ready rather than remote-ready.

## Information to collect

Ask in one `clarify` call when possible:

1. Access topology:
   - Private VPN/LAN (recommended for Basic Auth)
   - Local HTTPS reverse proxy on this host
   - Remote HTTPS reverse proxy on another host
2. Dashboard username (default `admin`).
3. Dashboard port (default `9119`).
4. For HTTPS: complete public URL, including `https://` and any path prefix.
5. For a remote proxy: exact proxy source IP/CIDR.
6. Whether to install/replace the platform autostart mechanism. Default yes for a fresh instance; inspect an existing systemd user unit on Linux or per-user Windows startup entry first.

Do not ask which Hermes profile should be exposed: the machine dashboard supplies the profile switcher.

## Procedure

1. Load the `hermes-agent` skill and current official Dashboard docs before changing anything.
2. Inspect live prerequisites: `command -v hermes`, `hermes --version`, `hermes doctor`, listener on selected port, existing dashboard process/service, and current `/api/status` if any. Confirm the machine being configured from the Desktop connection registry/current selection rather than inferring it from open sockets or the previously active gateway.
3. Run the helper in a user-visible interactive terminal and keep its completion observable. On Windows, use an attached console or a wrapper that displays the exit code and pauses; never launch a detached password prompt and treat the user's “finished” message as evidence that the helper succeeded. Invoke the helper with Hermes' own venv interpreter (the `python` sibling of the resolved `hermes` launcher), not an arbitrary system/tool Python, because importing the bundled auth plugin also imports Hermes runtime dependencies.

```bash
python ~/.hermes/skills/hermes/remote-access-setup/scripts/setup_remote.py \
  --mode vpn \
  --username admin \
  --port 9119
```

Modes:

- `vpn`: bind `0.0.0.0`; URL is supplied after launch from reachable host/IP.
- `https-local`: bind `127.0.0.1`; requires `--public-url https://host[/prefix]`.
- `https-remote`: bind `0.0.0.0`; requires `--public-url` and `--trusted-proxy IP_OR_CIDR`.

The helper prompts twice for the password, hashes it with Hermes' bundled scrypt implementation, preserves unrelated `.env` entries, generates a stable signing secret only if absent, configures `dashboard.public_url` and `dashboard.trusted_proxies` through `hermes config set`, installs the supported per-user autostart mechanism for the current platform when requested, and records backups/change notes. Immediately read back the listener, autostart entry, config, and `/api/status`; helper launch or password entry alone proves nothing.

4. For an HTTPS topology, configure or inspect the actual reverse proxy. It must preserve/forward Host, `X-Forwarded-Host`, `X-Forwarded-Proto`, optional `X-Forwarded-Prefix`, cookies, and WebSocket upgrades for `/api/ws` and `/api/pty`. Never expose 8642 merely for Hy4ri.
5. Verify the exact phone-reachable origin:

```bash
curl -fsS "$ORIGIN/api/status" | jq '.auth_required, .auth_providers'
```

Expected: `true` and `basic` (or the selected OAuth/OIDC provider).
6. Verify login and WebSocket ticket without exposing credentials. Prefer a local temporary cookie jar and secure password input. Delete the jar afterward. A status-only check is insufficient.
7. Report:
   - exact origin for the app, including scheme and optional port/path;
   - username only;
   - that the password is the one entered locally;
   - profiles are selected after login;
   - service/auth/status/ticket/proxy verification results;
   - anything still pending.

## Pitfalls

- Do not run the helper with an unrelated standalone Python. It re-execs through the interpreter beside the resolved `hermes` launcher before importing the bundled auth plugin.
- Do not interpret password entry or helper launch as success. Read back the listener, autostart mechanism, status response, authenticated session, and WebSocket ticket.
- A healthy LAN endpoint is only backend-ready when the public DNS name or reverse proxy is unavailable.

## Hy4ri client fields

- Server: full Dashboard origin, e.g. `https://hermes.example.com` or `http://100.x.y.z:9119`.
- Login: Dashboard username/password.
- Do not enter `API_SERVER_KEY`, `/api`, `/login`, or `/api/ws`.
- The app requests session cookies and one-time WebSocket tickets itself.

## Verification

- [ ] Exact client-facing `/api/status` reports `auth_required: true` and the intended provider.
- [ ] Authenticated `/api/auth/me` succeeds without exposing credentials or cookies.
- [ ] Authenticated `POST /api/auth/ws-ticket` returns a ticket; never print its value.
- [ ] Autostart, listener, public URL, trusted proxy, backup, and changelog were read back.
- [ ] For HTTPS, DNS, forwarded headers, cookies, and `/api/ws` upgrades work through the proxy.

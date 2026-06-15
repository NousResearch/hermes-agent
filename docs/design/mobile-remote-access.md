# Mobile Remote Access

## Goal

Hermes Desktop should be able to act as a local host for first-class remote
clients, including a future Flutter mobile app. The mobile client should drive
the user's own Hermes instance directly, keep sessions cleanly scoped to that
instance, and support multiple saved Hermes hosts without requiring a hosted
relay.

This document describes the upstream-friendly direction and the first backend
contract now exposed by the dashboard:

```http
GET /api/mobile/bootstrap
```

The endpoint is read-only and public by design. It lets a client confirm that a
reachable HTTP server is a Hermes dashboard host and choose the correct auth
flow before attempting authenticated API or WebSocket calls.

## Host and Client Model

- **Host:** a `hermes dashboard` process. Hermes Desktop can launch this process
  locally, and server installs can run it under `systemd`, `tmux`, or another
  supervisor.
- **Remote client:** a mobile app or another desktop app that talks to the host
  over the existing dashboard HTTP and WebSocket surface.
- **Session owner:** the host. Conversation storage, memory, skills, tools,
  cron jobs, and profiles remain on the Hermes machine. The remote client is a
  controller and display surface, not a second agent runtime.
- **No relay:** Hermes should not require a hosted middlebox for mobile access.
  A hosted relay may be a separate product decision later, but it is not part
  of this design.

## Connectivity

Supported connection paths should all terminate at the same dashboard host:

- **Local network or direct IP:** run `hermes dashboard --host 0.0.0.0 --port
  9119 --no-open` with dashboard auth enabled.
- **Tailscale or similar private overlay:** use the MagicDNS name or tailnet IP
  as the saved host URL. The dashboard's Host-header and peer guards still
  apply.
- **Cloudflare Tunnel or equivalent reverse proxy:** terminate TLS at the tunnel
  and forward to the dashboard. The public URL should preserve the dashboard
  prefix and WebSocket upgrade paths.

The mobile app should store a list of host records, for example:

- display name
- base URL
- last seen server version
- auth mode and provider names from `/api/mobile/bootstrap`
- user-selected profile or instance label, once supported by authenticated APIs

## Bootstrap Contract

`GET /api/mobile/bootstrap` returns:

```json
{
  "server_version": "x.y.z",
  "api_version": 1,
  "auth_required": true,
  "auth_providers": ["basic"],
  "features": {
    "dashboard_status": true,
    "desktop_gateway_ws": true,
    "pty_chat": true,
    "ws_ticket_auth": true,
    "device_pairing": false,
    "hosted_relay": false
  }
}
```

The response intentionally excludes host paths, PIDs, session counts, tokens,
user ids, profile names, and config. Public callers should only learn the
coarse capability and auth shape needed to continue safely.

`api_version` is the mobile bootstrap contract version, not the Hermes package
version. Additive fields may be introduced under the same version; incompatible
changes require a version bump.

## Auth and Security Model

The dashboard keeps its existing split:

- loopback binds use the ephemeral dashboard session token injected into the
  local web UI;
- non-loopback binds require the dashboard auth gate and a registered provider.

Remote clients should treat `/api/mobile/bootstrap` as discovery only. All
state-changing APIs, session APIs, and chat WebSockets must continue through the
existing authenticated dashboard paths. In gated mode, clients should sign in
through an advertised provider, then use the existing WebSocket-ticket flow for
live chat sockets.

Security requirements:

- no secrets in bootstrap responses;
- no weakening of dashboard auth or Host-header validation;
- no unauthenticated session lists, profile details, config, memory, or logs;
- no new core model tools;
- no prompt or toolset mutation in existing conversations.

## Pairing Direction

Device pairing should be an authenticated setup flow layered on top of the
dashboard, not a relay service. The expected direction:

1. A logged-in dashboard operator opens a pairing screen.
2. The host creates a short-lived, single-purpose pairing code.
3. The dashboard renders a QR code containing the base URL and pairing code.
4. The mobile app scans it, calls a pairing endpoint, and receives a scoped
   device credential.
5. The host stores device records under `get_hermes_home()` and lets the user
   revoke them from the dashboard.

This first slice deliberately does not add pairing persistence. The bootstrap
endpoint advertises `"device_pairing": false` until the full flow exists and is
tested.

## Multi-Instance Behavior

The mobile app should model each dashboard host as a separate instance. A user
may save several hosts, such as "Laptop", "Workstation", and "Homelab". Each
host owns its own Hermes home, profiles, sessions, credentials, and tools.

Within one host, profile selection should follow the existing dashboard profile
model instead of inventing a mobile-only session namespace. Remote clients
should display the active instance and profile clearly before sending prompts.

## Non-Goals

- Building a Flutter app in this repository slice.
- Adding a hosted relay.
- Adding pairing persistence before the endpoint and revocation model are
  designed and tested.
- Adding new core model tools.
- Exposing secrets, config, paths, sessions, memory, logs, or profile details
  through unauthenticated APIs.
- Replacing the existing dashboard auth, WebSocket-ticket, or Host-header
  protections.

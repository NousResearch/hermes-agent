# Hermes Flutter Client Prototype

This directory is the first scaffold for a mobile and desktop Flutter client
that remotely drives one or more Hermes dashboard instances. It is intentionally
separate from the Electron desktop app and does not modify the existing web,
desktop, or TUI surfaces.

## Current Scope

- Stores connection profiles for direct Hermes access over LAN, Tailscale, or a
  Cloudflare Tunnel.
- Uses the Hermes dashboard/FastAPI backend as the source of truth.
- Probes `GET /api/status` and the planned `GET /api/mobile/bootstrap` endpoint.
- Opens the dashboard JSON-RPC socket at `/api/ws?token=...` and handles the
  initial `gateway.ready` event.
- Keeps the first UI focused on clean boundaries: instance switching, adding a
  connection, testing status/bootstrap, and explicit New Session/Resume entry
  points.

No hosted relay or cloud service is included. Remote access should be provided
by the user's own network path, such as Tailscale, LAN routing, or Cloudflare
Tunnel.

## Setup

Install Flutter, then run from this directory:

```bash
flutter pub get
flutter analyze
flutter test
```

This worktree may be checked out on hosts without Flutter installed. In that
case, review the Dart source and the test plan below; do not treat missing SDK
validation as a successful Flutter build.

## Architecture

The UI includes a small Hermes design layer under `lib/design/` so mobile does
not drift into stock Material styling. Tokens are derived from the Electron
Desktop stylesheet:

- light blue chrome/sidebar surfaces
- `#0053fd` primary/midground accent
- warm `#cf806d` secondary accent
- compact Desktop-like type scale
- soft elevated cards, hairline borders, and rounded 9-10px controls
- monospace URL/status treatments for agent-host details

```text
lib/
  main.dart
  app.dart
  design/
    hermes_theme.dart
    hermes_components.dart
  hermes_api/
    connection_profile.dart
    hermes_rest_client.dart
    hermes_rpc_client.dart
    url_normalizer.dart
  features/
    connections/
    instances/
    sessions/
    chat/
```

- `ConnectionProfile` describes a saved Hermes dashboard instance: display name,
  base URL, auth token, and timestamps.
- `HermesRestClient` wraps dashboard REST calls. Today it supports
  `GET /api/status` and `GET /api/mobile/bootstrap`.
- `HermesRpcClient` wraps `/api/ws` using newline JSON-RPC framing and exposes a
  stream of gateway events. It records the `gateway.ready` payload when the
  socket starts.
- Riverpod providers keep the prototype simple. The current connection store is
  in memory, with `flutter_secure_storage` wired for the token persistence path
  that should replace it before production use.

## Backend Expectations

The dashboard already serves:

- `GET /api/status`
- `WebSocket /api/ws`

The mobile bootstrap endpoint is included as the intended contract:

- `GET /api/mobile/bootstrap`

Until that endpoint lands server-side, the app displays a bootstrap probe error
without treating it as a connection failure.

## Current Limitations

- Connection persistence is scaffolded but not fully implemented.
- The session list and chat transcript are placeholders; they document the next
  backend/UI contract rather than pretending to be complete.
- OAuth/browser login and WebSocket ticket minting are not implemented. Manual
  URL + token entry is the only supported auth path.
- No generated Flutter platform folders are committed from this scaffold.
- Build and test commands require a local Flutter SDK.

## Test Plan Without Flutter

If `flutter`/`dart` is unavailable on the host:

1. Confirm expected files exist under `apps/flutter_client`.
2. Review `test/url_normalizer_test.dart` and `test/api_models_test.dart`.
3. Check that no generated platform directories or desktop app files changed.
4. Install Flutter later and run `flutter pub get && flutter analyze && flutter test`.

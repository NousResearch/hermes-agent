# Mac App Shell Design

## Context

The v2.10 Multi-Entry Adapter architecture defines five entrypoints: feishu, discord, web, cli, and mac_app. The Mac App is the last entrypoint to be implemented, and it should be a lightweight desktop shell over Hermes Core — not a separate application core.

## Design Principles

1. **Hermes Core is authoritative.** The Mac App wraps the existing Web Console (port 9119). It does not fork or duplicate core logic.
2. **Native feel, web content.** Use WKWebView (Swift) or equivalent to embed the Web Console in a native window with native menu bar, title bar, and dock integration.
3. **Process management.** The Mac App can start/stop the Hermes backend process (hermes serve), show process status, and offer graceful shutdown.
4. **Local notifications.** Forward key events (watchdog timeout, policy.proposed, task completion) as macOS notifications.
5. **Adapter health.** Show adapter status in the menu bar or dock icon (green/yellow/red).
6. **Settings bridge.** Expose a native Settings pane that maps to hermes CLI config commands (model selection, adapter enable/disable, etc.).

## Architecture

```
┌─────────────────────────────────────┐
│  Mac App (Swift / SwiftUI)          │
│  ┌───────────────────────────────┐  │
│  │  Menu Bar                     │  │
│  │  - Status: ● Running          │  │
│  │  - Start/Stop Backend         │  │
│  │  - Open Console               │  │
│  │  - Settings                   │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  WKWebView (Web Console)      │  │
│  │  - Loads http://localhost:9119│  │
│  │  - Native scroll / gestures   │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  Background Service           │  │
│  │  - Process lifecycle          │  │
│  │  - Notification bridge        │  │
│  │  - Adapter health poll        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  Hermes Core (Python)               │
│  - Web Server (port 9119)           │
│  - Feishu Adapter                   │
│  - Discord Adapter (dormant)        │
│  - CLI (native)                     │
│  - EntryAdapter / SessionBinding    │
└─────────────────────────────────────┘
```

## EntryAdapter Integration

When the Mac App starts, it registers as a `mac_app` entrypoint via the v2.10 EntryAdapterRegistry:

```python
# In mac app startup
registry.register(MacAppAdapter(webview_port=9119))
```

The MacAppAdapter:
- `entrypoint = "mac_app"`
- Normalizes local user actions into EntryEvent (e.g., clicking "Start Task" in native menu)
- Reports health via `/api/v2.10/adapters/health`
- Does NOT call agents directly

## Notification Bridge

The Mac App polls `/api/v2.10/adapters/health` and watches for key events:
- Watchdog timeout → macOS notification
- Policy proposed → macOS notification with action buttons
- Task completion → macOS notification

This does NOT need a separate push notification system. It uses the existing Hermes API.

## Non-goals

- The Mac App does NOT implement its own agent execution
- The Mac App does NOT replace the Web Console
- The Mac App does NOT add streaming cards
- The Mac App does NOT manage Discord or Feishu adapters
- The Mac App does NOT require cloud deployment

## Recommended Implementation Order

1. **WebView shell** — minimal window that loads localhost:9119
2. **Process management** — start/stop hermes serve from the app
3. **Menu bar status** — adapter health indicator
4. **Notification bridge** — forward key events to macOS notifications
5. **Settings pane** — native UI for common config

## Status

Design only. No code implemented. Pending Phase 7+ decision.

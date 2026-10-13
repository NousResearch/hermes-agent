# Hermes Agent — iOS Interface

A native **SwiftUI** client for a self-hosted [Hermes Agent](https://hermes-agent.nousresearch.com/)
gateway. "Hermes in your pocket": chat with your agent, watch tools execute in
real time, resume CLI sessions, and switch models — all from your iPhone, with
the brain (Hermes Core) staying private on your own server.

This interface was scaffolded from the project PRD using the six "vibe-coding"
documents as the brief. The mapping below shows how each doc became code.

---

## Architecture (TRD → code)

Hybrid client-server, exactly as the PRD specifies. The phone is the **interface**;
your VPS / home server runs the **Hermes Core brain**.

```
┌─────────────────────────┐         HTTPS / SSE          ┌──────────────────────────┐
│   iOS App (SwiftUI)      │  ───────────────────────▶   │   Hermes Gateway          │
│                          │   /api/sessions              │   API_SERVER_ENABLED=true │
│  • Keychain credentials  │   /api/sessions/{id}/chat/   │                           │
│  • SSE streaming client  │       stream  (SSE)          │   • 70+ tools             │
│  • TUI gateway WS client │   /v1/models                 │   • terminal / web        │
│                          │  ◀───────────────────────   │   • SQLite (FTS5) state    │
│                          │   WebSocket JSON-RPC         │                           │
│                          │   command.dispatch           │                           │
└─────────────────────────┘                              └──────────────────────────┘
```

| Layer | Tech | Files |
| --- | --- | --- |
| UI | SwiftUI (iOS 17+) | `HermesAgent/Views/*` |
| State | `@MainActor` `ObservableObject` | `App/AppState.swift`, `ViewModels/*` |
| REST | `URLSession` actor | `Networking/HermesAPIClient.swift` |
| Streaming | `URLSession.bytes` SSE parser | `Networking/SSEClient.swift` |
| Slash commands | WebSocket JSON-RPC 2.0 | `Networking/TUIGatewayClient.swift` |
| Secrets | Keychain | `Networking/KeychainStore.swift` |

No third-party dependencies — pure first-party Apple frameworks.

---

## How the build maps to the 6 documents

**01 · PRD** — MVP feature set is implemented: native chat, tool-progress
visualization, slash commands, session sync, model switching, Keychain setup.

**02 · TRD** — `URLSession` for HTTP/SSE, `NWProtocol`-class WebSocket via
`URLSessionWebSocketTask`, Keychain for the API key. Targets iOS 17.

**03 · App Flow**
```
Launch
  └─ Connection configured?
       ├─ no  → ConnectionSetupView  (Server URL + API Key → Keychain)
       └─ yes → ping /health
                 ├─ ok   → SessionListView
                 │           ├─ tap session  → ChatView (loads /messages, streams turns)
                 │           ├─ + new         → POST /api/sessions → ChatView
                 │           ├─ model chip    → ModelPickerView  (/v1/models)
                 │           └─ gear          → SettingsView (disconnect)
                 └─ fail → ConnectionSetupView (error surfaced)
```
Empty, loading, and error states are handled throughout (`EmptyStateView`,
`TypingIndicator`, inline error banners).

**04 · UI/UX Brief** — Dark-mode-first, minimal, Linear/Raycast-flavored. Hermes
amber accent (`#E8A33D`). All tokens centralized in `App/Theme.swift`.

**05 · Backend Schema** — `Models/Session.swift` and `Models/ChatMessage.swift`
are decoded **directly from the Hermes `hermes_state.py` SQLite schema**
(`sessions` / `messages` columns: `id`, `role`, `content`, `tool_calls`,
`timestamp`, `title`, `parent_session_id`, token counts, …) so future data
synchronization stays compatible.

**06 · Implementation Plan** — Phased and traceable: setup → models → networking
→ chat streaming → tool progress → slash commands → polish. See the checklist
at the bottom.

---

## API contract used

All endpoints are the real Hermes API Server surface (`gateway/platforms/api_server.py`):

| Purpose | Endpoint |
| --- | --- |
| Health probe | `GET /health` |
| List models | `GET /v1/models` |
| List skills | `GET /v1/skills` |
| List sessions | `GET /api/sessions?limit=&offset=&source=` |
| Create session | `POST /api/sessions` |
| Rename / delete | `PATCH` / `DELETE /api/sessions/{id}` |
| Message history | `GET /api/sessions/{id}/messages` |
| **Stream a turn** | `POST /api/sessions/{id}/chat/stream` (SSE) |
| Slash commands | TUI Gateway `command.dispatch` over `wss://…/api/ws` |

**Auth:** `Authorization: Bearer <API_SERVER_KEY>` on every request.

**SSE events handled** (`Models/StreamEvent.swift`): `run.started`,
`message.started`, `assistant.delta`, `tool.started`, `tool.completed`,
`tool.progress` (incl. `_thinking` reasoning), `assistant.completed`,
`run.completed`, `approval.request`, `error`, `done`.

---

## Running it

Requires Xcode 15+ on macOS. The Swift sources are committed; the `.xcodeproj`
is generated with [XcodeGen](https://github.com/yonaskolb/XcodeGen):

```bash
brew install xcodegen
cd apps/ios
xcodegen generate
open HermesAgent.xcodeproj
```

Set your `DEVELOPMENT_TEAM` in `project.yml` (or in Xcode signing), build to a
simulator or device, then on first launch enter:

- **Server URL** — e.g. `https://hermes.your-domain.com` (Tailscale / Cloudflare
  Tunnel / reverse proxy in front of your gateway)
- **API Key** — matches `API_SERVER_KEY` on the gateway

### Backend prerequisite

Run the Hermes gateway with the OpenAI-compatible API server enabled:

```bash
API_SERVER_ENABLED=true API_SERVER_KEY=<your-strong-key> ./hermes
```

---

## Project layout

```
apps/ios/
├── project.yml                 # XcodeGen spec
└── HermesAgent/
    ├── App/                    # entry point, AppState, Theme
    ├── Models/                 # Codable types mirroring hermes_state.py + SSE/slash
    ├── Networking/             # REST, SSE, WebSocket JSON-RPC, Keychain
    ├── ViewModels/             # SessionList + Chat controllers
    ├── Views/                  # Setup, Sessions, Chat, ToolProgress, ModelPicker…
    └── Info.plist
```

---

## Status checklist (Implementation Plan)

- [x] Phase 1 — Project scaffold, theme, app state
- [x] Phase 2 — Schema-aligned models (`sessions` / `messages`)
- [x] Phase 3 — Connection wizard + Keychain auth
- [x] Phase 4 — Session list (sync, create, rename, delete)
- [x] Phase 5 — Chat with SSE streaming
- [x] Phase 6 — Real-time tool-progress bar + reasoning
- [x] Phase 7 — Slash-command engine (TUI gateway dispatch)
- [x] Phase 8 — Model picker (`/v1/models`)
- [ ] Future — Push approvals, Shortcuts/Siri, Widgets, offline SLM (PRD §7)

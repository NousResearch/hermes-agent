# feat(dashboard): add native web chat page for session history browsing

## What does this PR do?

Adds a **Web Chat** page (`/webchat`) to the Hermes dashboard that renders session history as native browser HTML instead of embedding a terminal emulator.

The existing `/chat` page works by spawning `hermes --tui` inside an xterm.js PTY over WebSocket. That is great on a local LAN, but it breaks or feels sluggish when the dashboard is served over a Cloudflare Tunnel, a reverse proxy, or a mobile connection: PTY streams are character-by-character and the VT100 escape sequences make the output hard to read on small screens.

The new `/webchat` page solves this by talking directly to the existing REST API endpoints (`/api/sessions`, `/api/sessions/:id/messages`) and rendering conversations with native React components — role-based message bubbles, inline tool-call badges, relative timestamps, and a searchable session sidebar. No PTY, no xterm, no terminal font required.

## Related Issue

Fixes #20845

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [x] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔒 Security fix
- [ ] 📝 Documentation update
- [ ] ✅ Tests (adding or improving test coverage)
- [ ] ♻️ Refactor (no behavior change)
- [ ] 🎯 New skill (bundled or hub)

## Changes Made

### `web/src/pages/NativeWebChatPage.tsx` *(new file, ~260 lines)*

Self-contained page with three visual regions:

**Session sidebar** (left, 256 px)
- Loads up to 100 sessions via `api.getSessions()` on mount.
- Live search/filter by title or model name.
- Refresh button to re-fetch without a full page reload.
- Each row shows: session title (or preview fallback), relative last-active time, message count, and model short name.
- Active/live sessions are flagged with a green indicator dot.
- Clicking a row pushes `?session=<id>` to the URL so the view is deep-linkable and survives a browser refresh.

**Message pane** (right, flex-1)
- Fetches `api.getSessionMessages(id)` when the active session changes.
- User messages are right-aligned with a `User` icon; assistant messages left-aligned with a `Bot` icon.
- Tool-result rows (`role: "tool"`) are rendered as compact mono lines with a `Wrench` icon rather than full bubbles.
- Assistant messages that have `tool_calls` show inline badges listing the called tool names before the text.
- Auto-scrolls to the bottom on load.
- **"Open in Chat"** button in the pane header navigates to `/chat?resume=<id>` so the user can immediately pick up the conversation in the TUI if they want to send a new message.

**Model badge** (top-right of header)
- Calls `api.getModelInfo()` once on mount and displays the active model's short name.
- Clicking it navigates to `/models` for model switching.

### `web/src/App.tsx` *(2 small edits)*

1. `import NativeWebChatPage from "@/pages/NativeWebChatPage"` — adds the import.
2. `"/webchat": NativeWebChatPage` added to `BUILTIN_ROUTES_CORE`.
3. `{ path: "/webchat", label: "Web Chat", icon: Globe }` prepended to `BUILTIN_NAV_REST` so the page appears at the top of the sidebar nav, above Sessions.

The `Globe` icon was already imported in `App.tsx` (it's in the `ICON_MAP`), so no new Lucide import is needed.

## How to Test

1. Start Hermes with the dashboard enabled (`hermes dashboard` or `hermes gateway --dashboard`).
2. Have at least one completed session with messages.
3. Navigate to `http://localhost:<port>/webchat` (or click **Web Chat** in the sidebar).
4. The session sidebar loads; click any session to view its message history.
5. Verify user / assistant / tool messages render correctly with the expected layout.
6. Type in the search box and confirm the session list filters in real time.
7. Click **Open in Chat** in the message pane header — the browser should navigate to `/chat?resume=<session-id>` and the TUI should resume that session.
8. Click the model badge in the top-right — the browser should navigate to `/models`.
9. Test over a tunnelled connection (e.g. Cloudflare Tunnel or `ngrok`) — the page should load and scroll smoothly without PTY lag.

## Checklist

### Code

- [x] I've read the Contributing Guide
- [x] My commit messages follow Conventional Commits (`feat(dashboard):`)
- [x] I searched for existing PRs to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this feature
- [ ] I've run `pytest tests/ -q` and all tests pass
- [x] I've added the page as a route and nav item consistently with every other built-in page
- [x] Tested on Ubuntu 24.04

### Documentation & Housekeeping

- [x] No new config keys — N/A
- [x] No tool schema changes — N/A
- [x] No cross-platform concerns (pure front-end React, no OS primitives) — N/A

## Screenshots / Logs

**Session sidebar + message pane (assistant response with tool call badges):**

```
┌─ Web Chat ───────────────────────────────────────────────────────────┐
│ [🔍 Search sessions…]                                    [Model: claude-…] │
├──────────────────────┬───────────────────────────────────────────────┤
│ ● Fix deploy script  │  Fix deploy script        [Open in Chat ↗]   │
│   2m ago · 14 msgs   │                                               │
│                      │  ┌─[User]──────────────────────────────┐     │
│ Review PR #42        │  │ Can you fix the deploy.sh script?   │     │
│   1h ago · 8 msgs    │  └─────────────────────────────────────┘     │
│                      │                                               │
│ Refactor auth layer  │  [Bot] 🔧 read_file  🔧 write_file           │
│   3h ago · 31 msgs   │  I've updated deploy.sh to use the           │
│                      │  correct environment variable…               │
│ …                    │                                               │
└──────────────────────┴───────────────────────────────────────────────┘
```

## Notes

- **Read-only by design.** Sending new messages still requires the TUI (`/chat`). A follow-up PR could add a REST-based compose input once `POST /api/sessions/:id/prompt` is exposed publicly, but that is out of scope here and noted in #20845.
- **No PTY spawn.** The page makes only two REST calls (`getSessions`, `getSessionMessages`) — it is safe to use over any HTTP proxy including Cloudflare Tunnel with no extra configuration.
- The `/webchat` route is a standard built-in route and respects the existing profile-scope switcher, management profile `?profile=` injection, and session-token auth exactly like every other dashboard page.

---

Branch: `private/issue-20845-native-webchat`
Base: `main`

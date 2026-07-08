# HT AI Agent Gateway Protocol (MVP subset)

**Status:** Reference for `apps/ht-web`
**Source of truth:** `ui-tui/src/gatewayTypes.ts` (type mirror), `apps/shared/src/json-rpc-gateway.ts` (client), and the `@method(...)` registrations in `tui_gateway/server.py`. This document is the *chat-frontend subset* — the full dispatcher exposes ~117 methods and ~40 event types; a chat UI needs the ~10 methods and ~20 events below.

This protocol is undocumented upstream; this file is the contract `apps/ht-web` is built and tested against. When the two disagree, the TypeScript mirror in `gatewayTypes.ts` wins (it's what the shipping TUI uses).

---

## 1. Transport & framing

- **Wire:** JSON-RPC 2.0, newline-delimited JSON, bidirectional. Identical over stdio (`python -m tui_gateway.entry`) and WebSocket (`/api/ws`). `apps/ht-web` uses the WebSocket transport.
- **Backend:** `ht serve --host 127.0.0.1 --port <p>` runs the gateway headless and mounts `/api/ws`.
- **Client:** reuse `JsonRpcGatewayClient` from `@hermes/shared` — it handles request/response correlation, the connect handshake, timeouts, and event fan-out. Do not reimplement framing.

### Request frame (client → server)
```json
{ "jsonrpc": "2.0", "id": "r1", "method": "prompt.submit", "params": { "session_id": "ab12cd34", "text": "hello" } }
```

### Response frame (server → client)
```json
{ "jsonrpc": "2.0", "id": "r1", "result": { "ok": true } }
```
Errors: `{ "jsonrpc": "2.0", "id": "r1", "error": { "code": 4004, "message": "..." } }`. The shared client rejects the pending promise with the `message`.

### Event frame (server → client, unsolicited)
```json
{ "jsonrpc": "2.0", "method": "event", "params": { "type": "message.delta", "session_id": "ab12cd34", "payload": { "text": "..." } } }
```
`id` is absent on events. The shared client routes `method === "event"` frames to `.on(type, handler)` listeners by `params.type`. **Every event carries an optional `session_id`** — a multi-session frontend MUST filter events against the session it's rendering (the gateway can drive several sessions on one socket).

---

## 2. Auth

`ht serve` on loopback with no dashboard auth configured accepts the socket directly. When the server enforces auth (`_ws_auth_ok` in `hermes_cli/web_server.py`), the URL carries a query credential:
- **token mode:** `?token=<injected>` — the server injects `window.__HT_SESSION_TOKEN__` / `window.__HERMES_SESSION_TOKEN__` into the served `index.html`; forward it as the `token` query param.
- **oauth/ticket mode:** mint a single-use ticket via `/api/auth/ws-ticket` immediately before connecting; pass as `?ticket=<t>`.

Build the URL with `buildHermesWebSocketUrl({ path: '/api/ws', authParam: ['token', t] })` from `@hermes/shared`. For a bare loopback `ht serve` (MVP default) no auth param is needed.

---

## 3. Handshake

1. Client opens the socket. `JsonRpcGatewayClient.connect(url)` resolves on `open`.
2. Server immediately emits **`gateway.ready`** with a `skin` payload (brand name, colors, banner art). The frontend applies the skin so branding is server-driven, not hardcoded.
3. Client calls `session.create` (new chat) or `session.resume` (reopen a prior conversation) to get a `session_id`.
4. All subsequent `prompt.submit` / `*.respond` calls and inbound events reference that `session_id`.

---

## 4. Methods (MVP subset)

All params are objects. `session_id` is required on everything except `session.create` / `session.list` / `session.active_list`.

| Method | Key params | Result | Purpose |
|---|---|---|---|
| `session.create` | `cols?`, `cwd?`, `title?`, `source?`, `messages?` (seed history) | `SessionCreateResponse` `{ session_id, info? }` | Start a fresh conversation |
| `session.resume` | `session_id` | `SessionResumeResponse` `{ session_id, messages[], info?, running?, status?, inflight? }` | Reopen a stored conversation with its transcript |
| `session.list` | — | `SessionListResponse` `{ sessions: SessionListItem[] }` | List stored conversations for the sidebar |
| `session.active_list` | — | `SessionActiveListResponse` `{ sessions: SessionActiveItem[] }` | List live/in-memory sessions with status |
| `session.delete` | `session_id` | `SessionDeleteResponse` `{ deleted }` | Delete a stored conversation |
| `session.title` | `session_id` | `SessionTitleResponse` `{ title?, pending? }` | Fetch/poll the auto-generated title |
| `prompt.submit` | `session_id`, `text`, `truncate_before_user_ordinal?` | `PromptSubmitResponse` `{ ok }` | Send a user turn. If a turn is running it is queued (and by default interrupts) |
| `session.interrupt` | `session_id` | `SessionInterruptResponse` `{ ok }` | Stop the in-flight turn |
| `clarify.respond` | `session_id`, `request_id`, `answer` | `{ ok }` | Answer a `clarify.request` |
| `approval.respond` | `session_id`, `choice` (`allow`/`deny`/…), `all?` | `{ resolved }` | Answer an `approval.request` |
| `secret.respond` | `session_id`, `request_id`, `value` | `{ ok }` | Provide a requested secret/env var |
| `sudo.respond` | `session_id`, `request_id`, `password` | `{ ok }` | Provide a sudo password |

Response shapes are the `*Response` interfaces in `gatewayTypes.ts`; `apps/ht-web/src/gateway/types.ts` re-declares the subset it consumes.

### `SessionListItem`
`{ id, title, preview, message_count, started_at, source? }`

### `SessionActiveItem`
`{ id, status: 'idle'|'starting'|'waiting'|'working', title?, preview?, model?, message_count?, last_active?, started_at? }`

### `GatewayTranscriptMessage` (from `session.resume`)
`{ role: 'user'|'assistant'|'system'|'tool', text?, name?, context? }`

---

## 5. Events (MVP subset)

Grouped by concern. `payload` shapes are from the `GatewayEvent` union in `gatewayTypes.ts`.

### Connection / branding
| Event | Payload | Handling |
|---|---|---|
| `gateway.ready` | `{ skin?: GatewaySkin }` | Apply skin (brand name, colors). Marks the socket live. |
| `skin.changed` | `GatewaySkin` | Re-apply skin live. |
| `session.info` | `SessionInfo` | Update model badge / session metadata header. |
| `error` | `{ message? }` | Surface a global error toast. |
| `gateway.stderr` | `{ line }` | Debug log ring (dev only). |
| `gateway.protocol_error` | `{ preview? }` | Debug: malformed frame. |

### Assistant output stream
| Event | Payload | Handling |
|---|---|---|
| `message.start` | — | Open a new assistant message bubble; clear the streaming buffer. |
| `message.delta` | `{ text?, rendered? }` | Append `text` to the streaming assistant bubble. **Ignore `rendered`** — it's terminal-column-aware Rich markup for the TUI; render `text` as Markdown client-side. |
| `message.complete` | `{ text?, rendered?, reasoning?, usage? }` | Finalize the bubble with `text` (authoritative full text); record `usage`. |
| `thinking.delta` | `{ text? }` | Append to a collapsible "thinking" area (optional in MVP). |
| `reasoning.delta` / `reasoning.available` | `{ text?, verbose? }` | Reasoning trace (optional in MVP). |
| `status.update` | `{ kind?, text? }` | Header status line ("working", "waiting"…). |

### Tool activity
| Event | Payload | Handling |
|---|---|---|
| `tool.start` | `{ tool_id, name?, args_text?, context?, todos? }` | Add a tool-call row keyed by `tool_id`. |
| `tool.progress` | `{ name?, preview? }` | Update the active tool row's preview. |
| `tool.generating` | `{ name? }` | Show "generating arguments" state. |
| `tool.complete` | `{ tool_id, name?, result_text?, summary?, inline_diff?, error?, duration_s?, todos? }` | Close the row keyed by `tool_id`; show summary/error. |

### Interactive requests (require a `*.respond` call)
| Event | Payload | Respond with |
|---|---|---|
| `clarify.request` | `{ request_id, question, choices: string[]\|null }` | `clarify.respond` `{ request_id, answer }` |
| `approval.request` | `{ command, description, allow_permanent? }` | `approval.respond` `{ choice, all? }` |
| `secret.request` | `{ request_id, env_var, prompt }` | `secret.respond` `{ request_id, value }` |
| `sudo.request` | `{ request_id }` | `sudo.respond` `{ request_id, password }` |

### Misc (optional in MVP)
| Event | Payload | Handling |
|---|---|---|
| `background.complete` | `{ task_id, text }` | Notify a background task finished. |
| `notification.show` / `notification.clear` | `{ id?, key?, level?, text?, kind?, ttl_ms? }` | Transient toast. |
| `subagent.*` | `SubagentEventPayload` | Spawn-tree view — **out of MVP scope**. |
| `voice.*`, `moa.*`, `billing.*`, `browser.*` | — | **Out of MVP scope.** |

---

## 6. Minimal chat lifecycle

```
connect(ws)                                  ── socket opens
  ← event gateway.ready { skin }             ── apply branding
→ session.create {}                          ── or session.resume { session_id }
  ← result { session_id }
→ prompt.submit { session_id, text }
  ← result { ok: true }
  ← event message.start
  ← event message.delta { text: "Hel" }      ── append
  ← event message.delta { text: "lo" }        ── append
  ← event tool.start { tool_id, name }        ── (if the turn calls a tool)
  ← event approval.request { command, ... }   ── (if the tool needs approval)
→ approval.respond { session_id, choice: "allow" }
  ← event tool.complete { tool_id, summary }
  ← event message.complete { text: "Hello", usage }
```

Interrupt: `→ session.interrupt { session_id }`.

---

## 7. Rendering rules for a non-terminal client

1. **Never render `rendered` fields.** They are pre-formatted for a fixed terminal column width. Use `text`.
2. Render assistant `text` as Markdown (`react-markdown` + a syntax highlighter).
3. Key tool rows by `tool_id` — `tool.start` and `tool.complete` correlate through it.
4. `message.complete.text` is authoritative; if deltas and the final text disagree (e.g. after a compression or edit), replace the bubble with the final text.
5. Filter every event by `session_id` against the active session before applying it.

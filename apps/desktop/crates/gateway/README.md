# Native Rust gateway

This crate runs the Rust `hermes_engine::Session::turn` over real HTTP and WebSocket transports. It does not spawn the Python gateway. It shares the desktop's `hermes-library` catalog, native transcript files, configured providers, approved workspace tools and optional MCP adapter.

It implements a **scoped subset**, checked against the supplied Hermes upstream checkout's `apps/shared/src/gateway-contract.openrpc.json`, `tui_gateway/ws.py`, `tui_gateway/contracts/canonical_sessions.py`, and Desktop `canonical-protocol.ts`. It is not a replacement for upstream's complete dashboard, control socket, account service or session authority. The native Desktop in this workspace uses the same engine directly; this listener is an optional integration surface.

## Run

From the workspace root:

```sh
cargo run -p hermes-gateway -- --port 8765
```

`--data-dir PATH` selects the catalog; otherwise it uses `Session::default_dir()` (including `HERMES_RUST_DATA_DIR`). `--host` accepts only a loopback IP. Port `0` chooses a free port. The single startup JSON line contains the HTTP origin, WebSocket URL and token file path. Ctrl-C or SIGTERM interrupts active turns, saves their terminal state, closes clients and stops the listener.

Provider configuration is the engine's `HERMES_RUST_BASE_URL`, `HERMES_RUST_MODEL` and `HERMES_RUST_API_KEY` environment configuration. The default Nous endpoint can use the desktop's valid Portal login. A custom endpoint receives only its explicitly configured key. Provider changes made in another running desktop are not automatically applied to an already running gateway. No provider secret appears in RPC snapshots.

The CLI creates a random, private `gateway.token` in the selected data directory. On Unix it requires mode `0600` for an existing token file. `--token-file PATH` selects another file; `HERMES_RUST_GATEWAY_TOKEN` supplies a token without writing it. Tokens must contain 32–512 printable ASCII characters without spaces. The secret is never printed by the server.

## Connect

Native clients first send authenticated `POST /api/auth/ws-ticket`, using either `Authorization: Bearer <gateway token>` or `X-Hermes-Session-Token: <gateway token>`. The reply is `{"ticket":"…","ttl_seconds":30}` with `Cache-Control: no-store`.

Then open `/api/ws?native_dial=1` with both WebSocket subprotocols:

```text
hermes-gateway-v1
hermes-gateway-ticket.<ticket>
```

The server selects only `hermes-gateway-v1`. Tickets expire after 30 seconds and are atomically consumed once. Reconnects require a fresh ticket. A canonical socket must create or resume a session before controlling it. Native-mode sockets never accept the reusable token in place of a ticket.

For legacy JSON-RPC clients, open `/api/ws?ticket=<fresh ticket>` without `native_dial`, or send the token in the upgrade's bearer/session-token header. The legacy `?token=` path is also supported for existing clients; prefer a one-use ticket so the reusable credential does not enter a URL.

The first frame has method `event`, type `gateway.ready`, a replay epoch, heartbeat support and `session_authority: true` on a native dial (`false` on legacy). `runtime.describe` returns the supported creation parameters and the explicit `rust-engine-subset-v1` capability. It does not advertise full upstream `sessions-v1` support. `rpc.discover` returns the exact method list and scope limitations. The first-party `gateway.ping` heartbeat is answered throughout a running turn, including approval waits.

The listener checks the actual peer, Host and Origin. It accepts loopback Hosts with the listener's exact port; absent Origin, that same loopback HTTP origin, `file://`, or `app://hermes`. It does not trust forwarded/proxy headers, expose a public bind option, provide permissive CORS, or accept an Internet site's Origin. No TLS or remote deployment is implemented.

## Supported operations

| Methods | Behavior |
| --- | --- |
| `ping`, `gateway.ping` | Liveness; the first-party heartbeat returns `{"ok":true}`. |
| `rpc.discover`, `runtime.describe`, `gateway.capabilities`, `client.capabilities` | Honest subset discovery. Server-request RPCs and batch clarify are not advertised. |
| `session.create` | New durable native catalog session. Supports title, model, configured-provider label and an existing `cwd`; source/cols are presentation metadata. Explicit unsupported policies are refused. Creation request IDs are saved for retry deduplication across restarts. |
| `session.resume`, `session.activate` | Attach this socket to a native/imported catalog session by exact Rust ID. Return transcript, live state and pending approvals. `omit_messages` skips the transcript while retaining its count. No upstream title lookup or SQLite runtime session is impersonated. |
| `session.list`, `session.active_list` | List stored/live catalog sessions, with title/query/limit and workspace filtering. No cursor pagination. Native `session.list` defaults to live scope; request `scope: "stored"` for saved sessions. |
| `session.history`, `session.info`, `session.status` | Native transcript projection and current engine state. Row IDs identify projected rows and are stable while the append-only transcript is unchanged. Rewind is not implemented. |
| `prompt.submit` | Admit one plain-text Rust turn and stream real model deltas. Native replies carry an admission receipt; legacy replies have `status: "streaming"`. One turn runs per session. Busy or explicit queued submissions are refused. |
| `prompt.receipt`, `prompt.cancel` | Read the retained admission's outcome, or request interruption of its active turn. Cancellation ACKs describe current state; wait for terminal receipt/event to observe completion. |
| `approval.pending`, `approval.received`, `approval.respond` | Real engine tool approval. Only the offered `once` and `deny` choices are accepted. Pending approvals survive socket reconnects; unanswered approvals expire after five minutes. Native responses require prompt ID and execution generation. No standing approval is granted. |
| `session.interrupt` | Cancel streaming/model/tool approval work and persist interruption. Native controls require the current execution generation, so an older control cannot stop a newer turn. |
| `session.events.since` | Bounded ordered event replay with sequence and restart epoch. A gap or different epoch explicitly requests a fresh snapshot. |
| `session.title`, `session.mutate` | Rename; canonical `session.mutate` also changes the model at an idle boundary. Canonical edits require revision fences and request IDs; model changes also require execution generation. Exact mutation retries are deduplicated within retained process state. Other mutation operations are refused. |
| `session.close` | Detach this socket from an idle session; release its engine lock when its last attachment closes it. It never deletes history. |

Requests use JSON-RPC 2.0 IDs and object parameters. Unknown methods return `-32601`; malformed JSON returns `-32700`; invalid arguments return `-32602`; state/policy refusals include `error.data.reason` and, for stale fences, current revision/generation. Only `profile: "default"` is served. Batch JSON-RPC requests are not supported.

A minimal native conversation is:

1. `runtime.describe {}`
2. `session.create {"request_id":"unique-create-id","source":"gui","title":"Example","cwd":"/existing/project"}`
3. `prompt.submit {"session_id":"returned-id","submission_id":"unique-send-id","text":"Explain this project"}`
4. Consume `message.start`, `message.delta`, `message.interim`, `tool.start`, `approval.request`, `tool.complete`, `message.complete`, and `session.info` events as applicable.
5. For a tool card, send `approval.respond {"session_id":"…","prompt_id":"…","execution_generation":1,"choice":"once"}` or `"deny"`.
6. On reconnect, mint a new ticket and `session.resume` before sending or responding. Resume's `prompts` array restores cards even when their original events were missed.

Events use `{"jsonrpc":"2.0","method":"event","params":{"type":"message.delta","session_id":"…","seq":3,"replay_epoch":"…","execution_generation":1,"payload":{"text":"hello","turn_id":"…","segment":0,"offset":0}}}`. Delta offsets count UTF-16 code units, matching the first-party renderer; interim segments have independent offsets. Complete events use upstream statuses `complete`, `error` or `interrupted`. Approval events expose both canonical `prompt_id` and legacy `request_id`.

## Persistence and limits

Conversation content is the engine's atomically saved `conversation.json`. Creation identity/model/source/cwd metadata lives alongside it in `gateway.json`; the catalog remains owned by `hermes-library`. The gateway cannot open a conversation while another desktop/process holds its exclusive engine lock; it reports a `session_busy` refusal. Close the other owner before opening that conversation through the gateway. Separate sessions can run concurrently.

Replay is at most 512 events and 512 KiB per loaded session; admissions and mutation receipts retain the last 256 entries. Those caches reset on process restart. Clients must refetch history after a replay gap/restart and must not blindly resend an old ambiguous tool-producing turn outside the retained admission window. Unfinished saved turns recover as interrupted through the engine. This crate does not claim upstream's durable admission ledger or exactly-once semantics across process death.

At most 64 sessions are loaded and 32 sockets connected. WebSocket frames/messages are capped at 1 MiB and prompts at 256 KiB; a slow client is disconnected when its bounded outgoing queue fills and can reconcile from history/replay. Closing a socket leaves its active turn running, so a short transport failure does not abandon an approval. The approval deadline and explicit interruption still apply. Stop the gateway gracefully to persist shutdown interruption.

Unsupported: queued/steered/redirected inputs, multimodal attachments, edit/regenerate/rewind/branch/compression/reset/delete mutations, cross-profile routing, legacy server-to-client request RPCs, upstream tool/plugin catalogs, process/control-socket discovery, bot/room transports, billing, scheduler, voice, screen/browser control, dashboard settings REST and automatic handoff to a Python gateway. Any optional MCP server is external to Rust; it is connected through the engine, not a fallback to Python.

## Verification

```sh
cargo test -p hermes-gateway
cargo clippy -p hermes-gateway --all-targets -- -D warnings
```

The integration tests bind real loopback HTTP/WebSocket listeners and a deterministic SSE model fixture. They exercise single-use ticket redemption, subprotocol negotiation, auth/Host/Origin rejection, native create deduplication, streamed Rust tool approval, reconnect/resume, real UTC tool output, history after gateway restart, admission receipts, replay epochs, fenced rename retries, legacy UTF-16 deltas, busy admission refusal and interruption of a live provider stream. They do not require credentials or contact a paid model provider. This transport suite establishes the documented shapes; it does not certify every upstream Desktop/TUI screen.

## Original shared client port

`src/replay.rs` ports the replay state from the official **NousResearch/hermes-agent v0.23.3** `apps/shared/src/json-rpc-gateway.ts`. The baseline SHA-256 is `6271ca1b034e50df8ce88a2e3f0149a66247d0a2b989880bf8a16e6df94eda8e`. It was checked against the clean official checkout as well as the unchanged source included in this repository. The port preserves these original distinctions:

- Live event cursors advance monotonically; ordinary live delivery still includes late frames, as the original does. Replayed and parked events use sequence deduplication.
- Reconnect installs every session's hold before publishing the open state. Gap events dispatch before held live frames, then that session's history barrier resolves.
- Each session settles independently. An unsupported replay or its 10-second deadline releases its held frames; losing the owning socket resolves its barrier `false`.
- A process restart clears only legacy process cursors. Canonical session epochs remain independent, and a parked new epoch is adopted after the old replay gap.
- Truncated tails are skipped. `snapshot_required` emits `session.replay_gap` and retires that cursor/epoch. `not_found` retires the deleted session. The cursor limit is 256, with least recently advanced cursors retired through an explicit gap event.
- Old socket responses/cleanup cannot flush a replacement socket's replay hold. The process epoch is attached as the original `replayEpoch` annotation.

`src/client.rs` supplies the standalone Rust WebSocket transport for that state: native ticket minting, ws/wss URL validation, multiplexed request IDs, request deadlines and cancellation, heartbeat liveness, typed refusal reasons, pending server-request redelivery, and history barriers. It exposes a broadcast event stream rather than JavaScript callbacks. A consumer seeing `RecvError::Lagged` must re-snapshot its session. Connection errors omit credential-bearing URLs. There is no JavaScript runtime dependency in the Rust client or server.

```rust,ignore
use hermes_gateway::client::{GatewayClient, ClientMessage};
use serde_json::json;

let client = GatewayClient::default();
let mut messages = client.subscribe();
client.connect_native("http://127.0.0.1:8765", &saved_gateway_token).await?;
let session = client.request("session.create", json!({"title":"Example"})).await?;
let sid = session["session_id"].as_str().unwrap();
client.request("prompt.submit", json!({"session_id":sid,"text":"Hello"})).await?;
// Consume ClientMessage::Event values, and render/answer approval cards.
// After an explicit reconnect, wait before publishing a pending history read:
if let Some(barrier) = client.session_replay_barrier(sid) {
    if !barrier.wait().await { /* discard the old read; read again on next open */ }
}
client.close();
```

For upstream legacy server-to-client requests, set `ClientOptions.server_requests = true`, subscribe before connecting, and handle `ClientMessage::ServerRequest`. Its response handle permits the first `respond` or `fail` only. `open_requests` replay is delivered before the containing RPC resolves. The native server in this crate uses canonical approval events instead of the legacy server-request RPC path.

The machine-readable `tests/fixtures/upstream-replay.json` contains **18 vectors whose expected traces were produced by executing the original TypeScript client**, including its original channel, on synthetic sockets. `tests/replay_vectors.rs` compares the Rust implementation against those traces. The TypeScript verifier uses Node's built-in type stripping and a loader that resolves the source's `.js` imports to neighboring `.ts` files; it does not rewrite source logic, install packages or contact a service.

From this workspace directory, the optional reference check is:

```sh
node --experimental-strip-types \
  --loader ./crates/gateway/tests/ts-source-loader.mjs \
  ./crates/gateway/tests/replay-reference.mjs \
  ../shared/src/json-rpc-gateway.ts \
  ./crates/gateway/tests/fixtures/upstream-replay.json --verify
```

The Rust transport test additionally connects this client to the real native Rust server, disconnects it while a clock tool awaits approval, completes the turn through another viewer, and verifies one replayed terminal event after reconnect, a valid history barrier, the saved transcript, and concurrent RPCs. These checks establish the ported replay behavior and essential native flow; they do not claim a full port of the original Desktop screens, JS callback registry, or all upstream canonical mutation adapters.

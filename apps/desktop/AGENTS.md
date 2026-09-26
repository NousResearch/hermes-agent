# Desktop app

The root guidance applies. Read `apps/desktop/DESIGN.md` for the visual and interaction contract, `apps/desktop/BUILDING.md` for build workflow, and `apps/desktop/src/AGENTS.md` for renderer/backend contracts.

## Authority and state

Desktop is a native chat client, not the dashboard or embedded TUI. Electron owns machine lifecycle and native capability, the renderer owns presentation and window-local interaction, and the backend owns sessions, tools, model calls, and streaming.

- The renderer reaches native capability only through the typed Electron bridge. It does not import Node/Electron directly or reimplement agent behavior.
- Backend-owned data is cached in the renderer, then reconciled. Merge refreshes with live/pinned state, guard stale async responses, roll back failed optimistic writes, and preserve references for no-op updates.
- Persisted keys declare their connection, profile, session, project, or window scope. Shared renderer state stays in the narrowest feature-owned store.
- Only the foreground surface publishes into the shared view. Background activity never steals navigation, focus, or the visible transcript.

## Session identity and switching

Use the stable identity for navigation and persistence, runtime identity for a live stream, and lineage root for state that survives compression. Translate explicitly at boundaries.

`prompt.submit` durably writes the session and user row before agent construction. The turn adopts that row rather than appending a duplicate. Preserve resumability when a first turn is interrupted.

Every new profile-keyed renderer persistence family must join both `migrateTilesForProfile` and `dropTilesForProfile` in `src/store/session-states.ts`. A profile rename must move tabs, owner routes, transcript caches, and remembered routes together.

A missing session may fall back to a fresh draft through `goneSessionVerdict`, but it preserves the unsent draft, does not clobber an existing draft, and offers an inline undoable notice without a toast, focus change, or extra navigation.

Treat connection changes as distinct operations:

- Local, remote, or cloud connection apply is a soft re-home. Keep the shell mounted, clear gateway-bound stores explicitly, and reconnect.
- Changing the runtime `HERMES_HOME` is a hard re-home and may reload the window.
- A live profile swap preserves background profiles, merges lists, and changes the foreground only on explicit selection.

After any switch, the active socket, profile, connection state, REST route, and filesystem route must agree.

## Resolution and authentication

Encode fallback precedence once as data or a pure resolver. Validate each candidate before use. A failed read may fall through. An authoritative write fails or rolls back instead of silently retargeting. Bound retries and distinguish missing capability from transient failure.

OAuth gateway connections mint a fresh one-time WebSocket ticket for every dial. Only a confirmed auth rejection triggers reauthentication. Transport and server failures remain connection errors. Connection tests exercise the actual WebSocket/auth leg.

Electron cookie partition names must avoid characters Electron percent-escapes. Renaming a persistent partition signs users out and requires an explicit migration decision.

## Guest-content boundary

Sandboxed artifact iframes and the preview `<webview>` cannot open OS resources on their own. Keep `setWindowOpenHandler` deny-only and keep `allowpopups` absent.

The sole external-link path is the `persist:hermes-preview` guest preload installed by `electron/preview-guest-preload-entry.ts`: it accepts only a trusted primary-button click on `a[target="_blank"]`, exposes nothing to page script, forwards through `sendToHost`, admits only `http:` and `https:` in `src/lib/preview-external.ts`, and then uses the existing `hermes:openExternal` policy. Do not widen the partition, gesture gate, or schemes. `file:` never reaches `shell.openPath` from guest content.

## Client behavior

Agent-callable renderer tools are session capabilities. Gate them from the session source and named toolsets, not `HERMES_DESKTOP` or another backend environment variable.

Loading, empty, reconnecting, stale/degraded, and exhausted states remain distinct and recoverable. Keyboard ownership follows focus. Hidden terminals and other expensive stateful surfaces remain alive. Hot paths avoid broad subscriptions, layout read-after-write, and mounting expensive content during a gesture. Test realistic transcripts and intermediate interaction states.

## Free tier

`free_tier.status` is backend truth and `free_tier.ack_notice` persists the one-time notice. Do not add a renderer local-storage latch. Billing, the status chip, and onboarding share one sign-in dialog and branch on `billing.free_tier` before generic logged-in state. User-facing copy avoids "guest", "anonymous", "claim", and "Nous Portal". Branch on structured `free_tier` fields, not provider display names.

## Tests

Use the package's declared scripts and `scripts/run_tests.sh` for Python-owned backend paths. Test local and remote routing, profile isolation, resolver failure rungs, stale response order, optimistic rollback, guest navigation, and realistic performance. Apply the checklist in `DESIGN.md` and update all affected locales.
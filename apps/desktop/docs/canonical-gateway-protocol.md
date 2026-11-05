# Canonical local gateway protocol

Native local connections use the ordinary gateway owner. Closing Desktop,
changing profile or evicting a viewer never transfers that owner into Electron's
child-process pool. Local ensure retains the update gate and rechecks profile
retirement after asynchronous waits. Remote SSH, URL and OAuth routing retain
their existing authentication paths; failure never creates a local replacement.

## Authentication

`gateway ensure --json` returns a credential-free endpoint. Native main requests
private `session-ticket` grants with exactly `profile_id`, `instance_id` and
`purpose`. WebSocket grants use `interactive`; JSON and file HTTP requests use
`native-http`. Each HTTP retry mints a fresh grant and sends it only in
`X-Hermes-Gateway-Ticket`, scoped to the endpoint's exact origin. HTTP credentials
are not stored in the renderer or attached to URLs/public connection descriptors.

Streaming `/api/fs/download` and its 404-only `/api/fs/read-data-url` fallback
use the same captured backend descriptor and session/profile selectors. Each
connection retry and fallback request obtains a new grant. Downloads do not
follow redirects or repeat the save dialog/body write after a stream failure.
The native grant remains primary-profile-only; rejected profile selectors do
not fall back to a remote credential.

Local previews and seekable `hermes-media://stream` playback read the native
filesystem, not gateway HTTP. Remote media remains on its existing token/OAuth
transport (`/api/files/stream`); remote data-URL previews use the JSON IPC API.
Native HTTP grants are never attached to arbitrary image URLs, web previews,
provider voice endpoints, or remote media requests.

## Session and input identities

`src/api/canonical-protocol.ts` adapts only native canonical sockets. Creation
uses source `gui`, not `cli`. The foreground draft retains its `request_id` after
an uncertain create ACK; New chat starts a new intent. Unsupported explicit
launch settings are rejected visibly rather than silently dropped. Renderer
routing profile, terminal column count and disabled fast mode are not execution
policy overrides. This does not provide a crash-durable create-intent journal.

The canonical `admission_id` is distinct from the submitted input ID. Receipt
projection retains both and verifies `ref.session_id` against the requested
session before acknowledging the prepared input. The existing native private
atomic-file journal is still written before transport admission and retired only
after a matching accepted receipt. An unknown receipt retains it for explicit
retry. Renderer-only localStorage recovery is not native crash durability.

## Image history and recovery

Canonical native image history carries image-routing hints followed by flattened
`[screenshot]` parts. Desktop projects that complete suffix into attachment
thumbnails without rewriting stored messages or image bytes. Caption prose and
assistant text remain text; legacy `@image:` history keeps its existing rendering.

The prepared-image recovery affordance offers only ordinary unqueued drafts in
their original connection/profile/session. Slash invocations, legacy ambiguous
attempts and other recovery identities stay in the journal unchanged; they are
not offered as ordinary image drafts or automatically resubmitted by this view.

## Controls and metadata

Live and replayed prompt projections retain `prompt_id` and execution generation,
including named event listeners. Stop uses the captured session generation;
approval and clarify responses resolve the exact projected prompt, never the
latest arbitrary prompt. Resume restores pending approval/clarify projections.

Canonical `session.title` and `session.archive` RPC requests become
`session.mutate` with a retained request ID and expected revision. An ambiguous
failure retains that CAS payload; an explicit revision conflict requires a fresh
snapshot before retry. Existing REST metadata writers are not migrated by this
RPC adapter and do not yet provide universal revision fencing.

## Cutover limits

The canonical authority currently exposes a narrower RPC set than the full
Desktop application. Unsupported model/tool/slash/session-management RPCs,
cross-window pending-input text projection,
Windows validated named-pipe bootstrap, and WSL topology require further
integration. Successful authenticated transport is not proof of a usable mounted
chat or native journal retirement. End-to-end acceptance must compose the HTTP
grant server and drive real Electron controls with disposable userData.

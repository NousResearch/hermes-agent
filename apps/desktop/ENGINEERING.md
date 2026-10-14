# Desktop engineering: the long form

The reasoning and worked detail behind the summarized sections of [`AGENTS.md`](./AGENTS.md). The
rules themselves are in `AGENTS.md`, which agents load automatically; this file is read on purpose.

## Identity is not incidental

Sessions have more than one identity, and conflating them is a recurring source
of "session not found" and vanishing history. Reason about which identity a
surface needs: durable navigation and anything the user pins or persists key off
the stable/durable identity; live streaming keys off the runtime identity; state
that must outlive compression keys off the lineage root. Keep the mapping between
them explicit and translate at the boundary rather than passing the wrong id
inward.

Two guarantees follow from this (#111868). The user's message is durable at
send: `prompt.submit` writes the session row AND the user row before the agent
build starts, and the turn adopts that row (`_adopt_submit_user_row` →
`agent._pending_cli_user_message`) instead of appending a second one, so a
freeze or force-quit during a slow first build leaves a resumable transcript
(user message present, no reply). And renderer state keyed by profile name —
persisted tabs (`tilesByProfile`), Bot tile owner routes, cached transcript
tails, remembered session/route, session owner hints — follows a profile
rename via `migrateTilesForProfile(old, new)` (`store/session-states.ts`), the
rename sibling of `dropTilesForProfile`. Add any new profile-keyed localStorage
family to BOTH, or a rename leaves it pointing at a backend that no longer
exists ("Couldn't open this session" on every restore).

When an id is verifiably gone anyway (`goneSessionVerdict` → `'draft'`), the
window drops to a fresh draft without toasting or looping — and the unsent
text stashed under the dead key follows it: the verdict calls
`announceGoneSessionDraft(id)` and the composer's swap onto the fresh scope
consumes it once (`adoptGoneSessionDraft`, `store/composer.ts`), seeding the
composer and publishing the inline, undoable `$restoredDraftNotice`. Offer,
don't hijack: no navigation beyond the drop itself, no focus steal, no toast,
and an already non-empty fresh draft is never clobbered.

## Server truth is cached, not owned

The renderer paints from a cache of backend truth, so it must reconcile, not
assume:

- **Merge, don't clobber.** A refresh is new information layered over what you
  already know, not a replacement that can drop live or pinned rows.
- **Be optimistic, then honest.** Direct manipulation should paint immediately
  from a snapshot; a failed write rolls back visibly and an authoritative
  refresh gets the last word.
- **Guard against the past.** Async results can arrive out of order; a stale
  response must never overwrite newer intent. Generation counters and request
  tokens exist for this.
- **Isolate the foreground.** Only the surface the user is looking at may publish
  into the shared view; background work updates its own cache quietly.
- **Coalesce noise, flush signal.** Batch high-frequency cosmetic updates, but
  let terminal transitions (a turn finishing, needing input, failing) reach the
  user immediately.
- **Preserve reference identity on no-ops.** Handing React a fresh array that
  contains the same data re-renders expensive trees for nothing.

## Switching context is a re-home, not a reboot

Changing profile, connection, or mode is a workspace switch, not a cold start.
The shell and whatever the user was doing stay put; only the gateway-bound view
is cleared and repopulated, and the previous context must not leak into the next
one. Reserve the full-screen boot/connecting experience for a genuinely unusable
backend.

There are three distinct switch shapes, and conflating them is the classic bug:

- A **connection/mode apply** (local ↔ remote ↔ cloud) is the soft re-home:
  shell mounted, gateway-bound stores explicitly wiped, then reconnect. Query
  invalidation alone cannot evict live session stores — wipe them.
- A **runtime home change** (switching the underlying `HERMES_HOME` profile) is
  a hard re-home: the window legitimately reloads and state resets by remount.
- A **live profile swap** in the same window activates another profile's socket
  while background profiles keep streaming; lists merge rather than wipe, and
  only an explicit user selection starts a fresh foreground draft.

Treating a soft switch as hard flickers the app; treating a hard one as soft
strands stale rows. After any swap, the active socket, active profile, and
connection atoms must agree, or REST and filesystem calls route to the wrong
backend.

## Cross everything as an observable ladder

Desktop lives at the seams: versions, profiles, local vs remote vs cloud,
partially installed runtimes, stale caches, older backends. The durable technique
for all of it is the same — an ordered ladder of candidates:

1. Precedence is written down, in one place, as data or a pure function.
2. A candidate is trusted only after it is validated at the right boundary.
   Existence is not proof; probe what you're about to rely on.
3. A failed *read* falls to the next rung; a failed *authoritative write*
   surfaces or rolls back rather than silently retargeting.
4. A missing capability and a transient failure are different: the first may
   enable a compatibility path or a disabled state; the second should retry.
5. Retries are bounded and end in a real recovery affordance — never an infinite
   spinner or a hot loop.
6. One resolver owns each policy so every caller gets the same answer. Scatter is
   how two call sites drift apart.

This is the shape of backend discovery, command/version fallbacks, connection and
auth resolution, workspace-cwd selection, capability detection, and preview
normalization alike. Learn the shape, not a snapshot of the current rungs.

Two auth-flavored corollaries worth naming because they are easy to get wrong:

- **One-time credentials are never reused.** An OAuth gateway connection mints a
  fresh WebSocket ticket on every dial and never falls back to the cached URL.
  Only a confirmed 401/403 (or an explicitly tagged auth rejection) means
  reauthentication; timeout, network, malformed-response, and server failures
  remain connectivity errors. Only long-lived token/local auth may reuse a
  cached URL as a lower rung.
- **A connection test must exercise the leg you'll actually use.** An HTTP
  status probe passing while the WebSocket/auth leg fails is a false positive
  that ships as "it said connected but nothing works."
- **Cookie-jar partition names contain nothing Electron percent-escapes.** A
  `persist:` partition becomes a `Partitions/<escaped name>` folder; a folder
  name with `%3A` (an escaped `:`) gets a cookie store Windows can neither read
  nor write, so the session silently never persists. `electron/oauth-partition.ts`
  pins the invariant; renaming a partition signs its users out once — say so.

## Guest content never opens anything by itself

Untrusted HTML runs in two places: sandboxed `allow-scripts` iframes (artifact
previews) and the preview pane's `<webview>` (`persist:hermes-preview`). Neither
may drive the OS browser without the user's hand on it (GHSA-9f4c-93c8-jc8g):
`setWindowOpenHandler` denies everything and never opens a URL as a side
effect (`electron/window-open-policy.ts`), and the webview has no
`allowpopups` — do not add it.

A guest page's `target="_blank"` links (Streamlit's "Ask Google" traceback
button) reach the OS browser through one explicit bridge instead:

- `main.ts` installs `electron/preview-guest-preload-entry.ts` via
  `will-attach-webview`, keyed on the `persist:hermes-preview` partition only.
  It is the app's only guest preload; a new webview does not inherit it unless
  it opts into that partition.
- The preload runs in the isolated world, exposes nothing to the page, and
  forwards only a **trusted** (`event.isTrusted`) primary-button click on an
  `a[target="_blank"]` to the host via `ipcRenderer.sendToHost`. A synthetic
  `dispatchEvent(click)` from page script is dropped there; page `window.open`
  stays blocked.
- `PreviewPane` admits `http:`/`https:` only (`src/lib/preview-external.ts`)
  and hands the URL to the existing `hermes:openExternal` IPC, which applies
  main's URL policy. `file:` is excluded on purpose: a guest must never reach
  `shell.openPath`.

Widening any of those three (partition key, trusted-click gate, scheme set)
reopens the gesture-less forced-navigation class the advisory closed.

## Nous free tier: state is pulled, never latched in the renderer

The free tier (a Nous identity with no account, `hermes_cli/anon_auth.py`) reaches the renderer
through one JSON-RPC pair: `free_tier.status` (has_guest, enabled, available,
notice_pending, model, label) read from local auth state with zero network, and
`free_tier.ack_notice`, which persists the one-time notice flag on the identity itself. The
first-launch ready screen and the own-key strip are the SAME state rendered for two situations,
keyed on `notice_pending`; there is no localStorage latch, so the CLI and the desktop cannot
disagree about whether the notice was shown. Sign-in goes through the existing
`POST /api/providers/oauth/nous/start` + poll route, which over a free-tier identity registers the
connector transfer and reports `reason`, `account_email` and `model` on completion; every entry
point (Billing, status chip, ready screen) opens the one free-tier sign-in dialog. Never branch on
provider display names: the picker row carries `free_tier_row`, status cards carry `free_tier`.

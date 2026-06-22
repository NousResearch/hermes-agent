# Bug: Desktop composer locks on "Starting Hermes…" after sleep/wake (stale cached backend)

- **Status:** Diagnosed; upstream `main` has a first-pass revalidate recovery,
  and this branch hardens it with atomic `getConnection(..., { revalidate: true })`
  revalidation plus a two-miss remote rebuild threshold.
- **Severity:** High (user is fully locked out of an active session; only an app restart recovers)
- **Component:** `apps/desktop` (Electron main + renderer gateway boot/reconnect)
- **Primarily affects:** Remote / `global-remote` mode backends
- **Workaround:** Quit and reopen the desktop app

> This document preserves the original diagnosis for review. Upstream now includes
> a simpler revalidate IPC; this branch keeps the recovery behavior but reduces
> reconnect races and false-positive remote rebuilds.

---

## 1. User-visible symptom

After the machine sleeps (lid closed / idle) and is woken later, returning to an
open chat in Hermes Desktop leaves the message composer **disabled**, showing the
placeholder **"Starting Hermes…"**. The user cannot type or send anything — they
appear "locked out" of a session that was working fine before sleep.

The session itself is long-lived (the reporter's status bar showed
`Session 2:23:38`), and **quitting and reopening the app restores chat
immediately.**

Reporter's words:

> "after i leave a chat for a while in hermes desktop app… and come back to it…
> i can't seem to continue chatting? i seem to be locked out?"
> "i think i put my laptop to sleep and came back to it"
> "btw it works after i quit and re-open hermes desktop app"

This is a **legitimate bug**, not a usage error. A session is supposed to survive
sleep/wake and reconnect transparently.

## 2. Reproduction

1. Launch Hermes Desktop connected to a **remote** gateway (`global-remote` mode,
   or any `mode: 'remote'` connection).
2. Open/continue a chat and confirm the composer is enabled.
3. Put the Mac to sleep (or let the network drop long enough that the remote WS is
   torn down), then wake it some time later.
4. Observe: the composer stays disabled on **"Starting Hermes…"** and never
   recovers, no matter how long you wait.
5. Quit the app and reopen it → chat works again.

## 3. The exact stuck state (why the placeholder text matters)

The placeholder is a precise signal of the internal gateway state.

`apps/desktop/src/app/chat/composer/index.tsx:207`

```ts
const placeholder = disabled
  ? gatewayState === 'closed' || gatewayState === 'error'
    ? t.composer.placeholderReconnecting   // "Reconnecting to Hermes…"
    : t.composer.placeholderStarting       // "Starting Hermes..."
  : restingPlaceholder
```

`apps/desktop/src/app/chat/index.tsx:184,359`

```ts
const gatewayOpen = gatewayState === 'open'
// …
<ChatBar disabled={!gatewayOpen} … />
```

So:

- `disabled` ⟺ `gatewayState !== 'open'`.
- The placeholder is **"Starting Hermes…"** specifically when `disabled` **and**
  `gatewayState ∉ {closed, error}` — i.e. the gateway is pinned in **`connecting`**
  (or `idle`), **not** cleanly closed.

If the socket had merely closed, we'd see **"Reconnecting to Hermes…"**. We see
"Starting Hermes…", which means the renderer is perpetually mid-connect against a
backend it can never reach.

## 4. Root cause

The renderer's reconnect loop is **sound** — it is *not* where the lockout
originates. The lockout comes from the Electron main process handing the renderer
a **stale, cached connection descriptor** that points at a backend which is no
longer reachable, and never re-validating or rebuilding it for the life of the
process.

### 4a. The renderer reconnect loop is correct (ruled out as the cause)

`apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts`

- On a post-boot `closed`/`error` it schedules a reconnect with exponential
  backoff (1s → 2s → 4s → 8s → 15s cap, `:166-178`, `:212-226`).
- `attemptReconnect` (`:115-164`) is serialized by a single `reconnecting` latch
  that is **always** reset in a `finally` block (`:157-163`) — it cannot leak
  `true` forever.
- Wake signals (`powerMonitor` resume, `online`, `visibilitychange`) nudge an
  immediate reconnect (`:230-243`).

`apps/shared/src/json-rpc-gateway.ts`

- `connect()` carries a **15s connect timeout** (`DEFAULT_CONNECT_TIMEOUT_MS`,
  `:62-65`) that forces a hung `connecting` → `error` (`:161-182`) and drops the
  half-open socket, so a zombie `connecting` cannot persist on its own.

Conclusion: the renderer keeps retrying forever and would recover the instant
`getConnection()` returned a reachable descriptor. It never does.

### 4b. The main process caches the connection and never invalidates it (the bug)

The reconnect path is:

`getConnection(profile)` (preload `:4`) → IPC `hermes:connection` (`main.cjs:4607`)
→ `ensureBackend(profile)` (`main.cjs:4135`) → `startHermes()` (`main.cjs:4312`).

`startHermes()` is a **pure cache hit with no liveness check**:

`apps/desktop/electron/main.cjs:4322`

```js
if (connectionPromise) return connectionPromise
```

`connectionPromise` is only ever cleared by:

- the local backend child's `'error'`/`'exit'` handlers
  (`main.cjs:4400-4415`, `4416-4438` → null `connectionPromise`), or
- `startHermes()`'s own boot `catch` (`:4462-4475`), or
- `resetHermesConnection()` (`:4084-4093`), which today is only called on a
  **profile switch / connection-config change**, never during a normal reconnect.

### 4c. Remote mode has no child process, so the cache is never cleared

For a **remote** primary, `startHermes()` resolves via `resolveRemoteBackend()`
and returns **without spawning any child** — `hermesProcess` stays `null`:

`apps/desktop/electron/main.cjs:4328-4348`

```js
const remote = await resolveRemoteBackend(primaryProfileKey())
if (remote) {
  await waitForHermes(remote.baseUrl, remote.token)
  // … returns { mode: 'remote', baseUrl, token, wsUrl, … }
  return { /* remote descriptor */ }
}
```

Because there is no child process, the `'error'`/`'exit'` handlers that would
normally null `connectionPromise` **never exist**. The resolved remote descriptor
(`{ baseUrl, token, wsUrl }`, captured at boot) is therefore cached for the
**entire lifetime of the main process**.

### 4d. Putting it together

1. At boot the remote is reachable; `startHermes()` resolves and caches a good
   `{ mode: 'remote', baseUrl, token, wsUrl }`.
2. Sleep tears down the live WebSocket (and, over a long sleep, the remote
   endpoint the descriptor points at may move / restart / drop the session).
3. On wake the renderer's loop fires `getConnection()` → `startHermes()` →
   returns the **same cached descriptor** (`:4322`).
4. The renderer re-mints a WS ticket and calls `gateway.connect()` against that
   descriptor. It can't connect → `connecting` → (15s) `error` → backoff →
   `connecting` → … **forever**. The composer stays disabled on "Starting Hermes…".
5. Nothing in the running app ever re-resolves or rebuilds the remote connection,
   because only a child-process exit (which never happens for remote) clears the
   cache.

## 5. Why quit + reopen fixes it

`connectionPromise` is a **module-level variable** (`main.cjs:476-477`). A full
quit tears down the entire Node/Electron process, so on relaunch
`connectionPromise === null` and `startHermes()` is forced to rebuild from
scratch — re-running `resolveRemoteBackend()` + `waitForHermes()` and producing a
fresh, reachable descriptor. This is the single state reset that an in-app
reconnect cannot trigger, and it is exactly why reopening works while waiting does
not.

## 6. Scope of impact

- **Remote / `global-remote` mode (primary):** broken as described — no self-heal
  without an app restart.
- **Local mode:** mostly self-heals, because the local child's `'exit'` handler
  nulls `connectionPromise` on a real crash, letting the next `getConnection()`
  respawn. **One residual gap:** a local child that becomes *unresponsive but
  never exits* (hung) would also hand back a cached-but-dead descriptor. This is
  rarer and is treated as a follow-up (see §8).

## 7. Verification

Diagnosis was cross-checked by three independent investigations plus a fix
red-team:

- **Remote-cache lens — confirms (0.98):** in remote mode `connectionPromise` is
  never invalidated for the life of the process; `startHermes()` returns the
  cached value unconditionally at `:4322`.
- **Renderer-loop lens — confirms (0.98):** the renderer reconnect loop is
  provably sound (no latch leak, 15s timeout prevents stuck `connecting`, retries
  indefinitely) → the lockout must be backend-side.
- **Relaunch-diff lens — confirms (0.95):** the only state a full relaunch resets
  that an in-app reconnect cannot is the module-level `connectionPromise` cache.

## 8. Implemented hardening

**Revalidate-on-reconnect.** Keep an opt-in liveness check so a reconnect cannot
re-dial a dead cached backend, but make the check part of `getConnection()` rather
than a separate preflight IPC:

- The renderer's **backoff-paced** reconnect (`use-gateway-boot.ts` only) calls
  `getConnection(profile, { revalidate: true })`.
- On a cache hit with `revalidate`, `startHermes()` fast-probes the **public**
  `/api/status` (token-free `fetchPublicJson`, ~2.5s). If two consecutive
  backoff-scoped probes fail for a `mode === 'remote'` primary backend, it drops
  the cache via `resetHermesConnection()` and rebuilds, so the renderer's existing
  loop gets a fresh, reachable descriptor — no app restart required.

Red-team-driven guard rails (to avoid regressions):

- **Do not** add `revalidate` to `use-gateway-request.ts` (it fires on any
  transient request blip and could needlessly tear down a healthy backend).
- **Only tear down `mode === 'remote'`** connections; local backends self-heal via
  the child `exit` handler, so a probe miss there is treated as "WS not reattached
  yet", not "backend dead".
- **Require 2 consecutive probe failures** before rebuild, so a single
  captive-portal / VPN-re-establishing blip on wake doesn't trigger a respawn the
  backoff loop would have ridden out.
- Steady-state and cold boot stay `revalidate`-off → **zero added latency** on the
  happy paths.

### Out of scope / follow-ups

- Pool (non-primary, multi-profile) backends are not revalidated in this change.
- The local "alive-but-wedged, never exits" case (§6).
- An optional renderer watchdog for a zombie `connecting` state (the loop is
  already proven sound, so this is belt-and-suspenders).

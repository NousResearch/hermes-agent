# Bridge Contract — `window.hermesDesktop` (browser shim, v3 remote-only)

Extracted from the pinned **hermes-agent v2026.7.7.2** checkout in
`desktop-port/vendor/`. Every claim below carries `file:line` evidence from that
checkout. This document is the reference basis for `hermes-web-shim.js` and is
authoritative wherever other notes disagree with it.

**Path base (corrected):** every `file:line`
citation is **relative to `desktop-port/vendor/`**. Renderer sources therefore
read `apps/desktop/src/…`, the Electron main process reads
`apps/desktop/electron/…`, the shared WS helper reads `apps/shared/src/…`, and
the browser dashboard reference reads `web/src/…`. (The prior revision stated the
same base but then cited bare `electron/…`, `store/…`, `global.d.ts` — which
resolve one directory too high; all citations are now fully qualified.) The
vendor `electron/*.ts` sources ship compiled as `electron/*.cjs`; line numbers
are for the `.cjs` files actually present.

**Mode:** the shim targets **token mode** — a self-hosted gateway whose
`GET /api/status` returns `auth_required: false`. The shim implements only the
token / `?token=` path, not OAuth `?ticket=`.

**v3 pivot:** the renderer is bundled inside an iOS app and reaches the gateway
**remotely only**. There is no same-origin host and no dashboard-token scrape.
The connection source is a stored config in `localStorage` key
**`hermes.remoteGateway`** (`{ url, token }` JSON). This replaces the v2 source
(`window.location.origin` + `window.__HERMES_SESSION_TOKEN__` scrape).

---

## §1 — `api(request)`: request shape + handler semantics

**Type** (`apps/desktop/src/global.d.ts:65`):
`api: <T>(request: HermesApiRequest) => Promise<T>`

**`HermesApiRequest`** (`apps/desktop/src/global.d.ts:521-530`):

| field       | type              | notes |
|-------------|-------------------|-------|
| `path`      | `string`          | Full path **including the query string** — callers bake the query in (`apps/desktop/src/hermes.ts:204-208`, e.g. `/api/sessions?limit=40&offset=0&…`). There is **no** separate `query` field. |
| `method?`   | `string`          | Defaults to `GET`. |
| `body?`     | `unknown`         | A plain JS object; JSON-serialized by the handler (`apps/desktop/src/hermes.ts:264-268` passes `{ archived }`). |
| `timeoutMs?`| `number`          | Per-request timeout; default 15000 ms (below). |
| `profile?`  | `string \| null`  | Routes to a profile's backend in Electron. **The shim ignores it** — one remote backend. |

**Handler `hermes:api`** (`apps/desktop/electron/main.cjs:6555-6596`):
- Builds `url = \`${connection.baseUrl}${requestPath}\`` (`main.cjs:6579`).
- Token mode calls `fetchJson(url, connection.token, { method, body, timeoutMs })`
  (`main.cjs:6591-6595`) and **returns its result directly**.

**`fetchJson`** (`apps/desktop/electron/main.cjs:3285-3345`):
- Headers **always** include `'Content-Type': 'application/json'` and
  `'X-Hermes-Session-Token': token` (`main.cjs:3302-3303`).
- Body serialized only when `body !== undefined`.
- **Return value is parsed JSON — NOT an `{ ok, status, body }` envelope**:
  - `statusCode >= 400` → **reject** `new Error(\`${statusCode}: ${text || res.statusMessage}\`)` (`main.cjs:3313-3315`). **Confirms the `"<status>: <body>"` error format.**
  - empty body → `resolve(null)` (`main.cjs:3317-3319`).
  - HTML body — `/^\s*<(?:!doctype|html)/i` **OR** `content-type` contains `text/html` — → reject an "Expected JSON … endpoint is likely missing" diagnostic (`main.cjs:3325-3335`).
  - otherwise `resolve(JSON.parse(text))`; parse failure → reject `Invalid JSON from …` (`main.cjs:3336-3339`).

**Header name** independently confirmed: `SESSION_HEADER = "X-Hermes-Session-Token"`
(`web/src/lib/api.ts:38`); WS uses `?token=` (`apps/desktop/electron/connection-config.cjs:13-14,70`).

**Default timeout**: `DEFAULT_FETCH_TIMEOUT_MS = 15_000`; `resolveTimeoutMs`
returns the value when finite `> 0`, else the fallback
(`apps/desktop/electron/hardening.cjs:6,13-23`).

**Shim `api()`** (`hermes-web-shim.js`): reads the **stored** `{ url, token }`
via `requireConnection()`; fetches `config.url + path`; **always** sends both
headers; `JSON.stringify(body)` when defined; `AbortSignal.timeout(timeoutMs)`
(default 15000); throws `` `${status}: ${text}` `` on `>= 400`; `null` for empty;
throws the HTML diagnostic (body sniff **and** `content-type: text/html`); else
`JSON.parse`.

**v3 changes vs v2's `api()`:**
- Connection source is the stored config, **not** `window.location.origin`.
- `credentials: 'include'` **removed** — token auth
  is header-based; no cookies are involved for the remote gateway.
- **No 401 auto-recovery.** v2 re-scraped the dashboard token and retried on 401;
  there is no token to re-scrape now. A 401 surfaces verbatim as `` `${status}: ${body}` `` so the renderer's recovery UI (§6) takes over.
- HTML fall-through diagnostic now also checks `content-type: text/html`,
  matching `fetchJson` (`main.cjs:3327`).
- No stored config → `api()` throws the NO_CONFIG error (§6). Callers that run
  independently of boot (e.g. `refreshActiveProfile → api('/api/profiles/active')`,
  `apps/desktop/src/store/profile.ts:113`) get a clean error, not a hang.

---

## §2 — `HermesConnection` fields the renderer reads (from stored config)

**Type** (`apps/desktop/src/global.d.ts:359-373`) — required unless marked `?`:

| field                 | type                                    | read by |
|-----------------------|-----------------------------------------|---------|
| `baseUrl`             | `string`                                | media download URL (`apps/desktop/src/lib/media.ts:72,75`), cache key (`apps/desktop/src/lib/desktop-fs.ts:24`) |
| `isFullscreen`        | `boolean`                               | window/title-bar chrome; merged on `onWindowStateChanged` (`apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts:313-317`) |
| `mode?`               | `'local' \| 'remote'`                   | `isRemoteGateway()` (`apps/desktop/src/lib/media.ts:116`), `isRemoteMode()` (`apps/desktop/src/store/updates.ts:244`) |
| `authMode?`           | `'oauth' \| 'token'`                    | WS path selection in `resolveGatewayWsUrl` (`apps/shared/src/websocket-url.ts:38`) |
| `nativeOverlayWidth`  | `number`                                | title-bar overlay width |
| `source?`             | `'env' \| 'local' \| 'settings'`        | connection provenance display |
| `token`               | `string`                                | authed media download URL (`apps/desktop/src/lib/media.ts:72,75`) |
| `wsUrl`               | `string`                                | WS fallback in `resolveGatewayWsUrl` (`apps/shared/src/websocket-url.ts:63`) |
| `logs`                | `string[]`                              | boot diagnostics panel |
| `profile?`            | `string`                                | mint scope (`apps/shared/src/websocket-url.ts:36`), cache key (`apps/desktop/src/lib/desktop-fs.ts:24`) |
| `windowButtonPosition`| `{ x: number; y: number } \| null`      | traffic-light inset |

**Shim `getConnection()`**: `requireConnection()` then
`baseUrl=config.url, isFullscreen=false, mode='remote', authMode='token',
nativeOverlayWidth=0, source='settings', token=config.token,
wsUrl=buildWsUrl(config.url, config.token), logs=[], windowButtonPosition=null`.
`profile` left `undefined` (single backend; read as `connection.profile || ''`,
`desktop-fs.ts:24`). No stored config → **throws** (§6).

---

## §3 — WebSocket URL construction

**`buildGatewayWsUrl(baseUrl, token)`** (`apps/desktop/electron/connection-config.cjs:65-71`):

```js
const parsed  = new URL(baseUrl)
const wsScheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
const prefix   = parsed.pathname.replace(/\/+$/, '')
return `${wsScheme}://${parsed.host}${prefix}/api/ws?token=${encodeURIComponent(token)}`
```

**Shim `buildWsUrl(baseUrl, token)`** reproduces this **byte-for-byte** from the
stored URL (scheme from the URL's own protocol, `http→ws` / `https→wss`, path
prefix preserved). Used by `getConnection().wsUrl` and `getGatewayWsUrl()`.
`resolveGatewayWsUrl` (token mode) mints via `getGatewayWsUrl` and falls back to
`conn.wsUrl` (`apps/shared/src/websocket-url.ts:34-64`); both resolve identically.

**URL normalization** — `normalizeRemoteBaseUrl`
(`apps/desktop/electron/connection-config.cjs:40-63`): trim; require `http:`/`https:`;
strip hash/search; strip trailing slashes. The shim's `normalizeBaseUrl` mirrors
it and is applied on **save/apply/test/probe** so the stored URL is always clean.

---

## §4 — `profile.get()` return shape

**Type** (`apps/desktop/src/global.d.ts:58-64`): `profile.get: () => Promise<DesktopActiveProfile>`.
**`DesktopActiveProfile`** (`apps/desktop/src/global.d.ts:386-390`): `{ profile: string | null }`.

**Boot usage** (`apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts:364-365`):
`const pref = await desktop.profile?.get?.(); const profileKey = (pref?.profile ?? '').trim() || 'default'`.
`null` preference ⇒ `profileKey = 'default'`.

**Shim**: `profile.get → { profile: null }` (the primary profile). `profile.set`
accepts only the primary profile (`''` / `null` / `'default'`) and **throws** for
any named profile, so `switchProfile()` (`store/profile.ts:140`, called unguarded)
fails loudly rather than leaving the UI believing it switched while requests still
hit the one unscoped backend. `api`, `getConnection`, `getGatewayWsUrl` and
`touchBackend` apply the same `assertPrimaryProfile` guard: this client binds to a
single remote gateway, so profile switching is disabled by design.

---

## §5 — UNGUARDED bridge calls in `store/zoom.ts` and `store/profile.ts`

`apps/desktop/src/store/zoom.ts` (guard `window.hermesDesktop?.zoom`, `:19`):
- `zoom.get().then(({ percent }) => …)` (`:20`) — must resolve `{ level, percent }` (`global.d.ts:93`).
- `zoom.onChanged(({ percent }) => …)` (`:21`) — must return an unsubscribe.
- `zoom.setPercent(percent)` (`:16`, guarded).

Shim `zoom`: `get → { level: 0, percent: 100 }` (100 % = `$zoomPercent` default,
`store/zoom.ts:13`), `setPercent → () => {}`, `onChanged → () => () => {}`.

`apps/desktop/src/store/profile.ts`:
- `api<ActiveProfileResponse>({ path: '/api/profiles/active', timeoutMs })` (`:113`, **unguarded**, covered by §1).
- `profile.set(name)` (`:140`, **unguarded**, covered by §4).

---

## §6 — Boot path + the no-config failure

`const desktop = window.hermesDesktop` (`apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts:85`).

**Synchronous listener registrars** (`use-gateway-boot.ts`): `onBootProgress(cb)`
(`:224`), `getBootProgress()` (`:225`), `onBackendExit(cb)` (`:321`) are evaluated
**before / outside** `boot()`'s try/catch. A missing one throws an **uncaught
`TypeError` → the overlay hangs at 2 %**, *not* a clean fail. The shim provides
all three as real functions; `getBootProgress()` returns `running: true` so
applying the snapshot never prematurely hides the overlay (`store/boot.ts:41`) —
dismissal is owned by `completeDesktopBoot()` (`:398`).

**`getConnection()` is the boot's connection gate** (`use-gateway-boot.ts:336`,
reconnect `:147`): it runs **inside** `boot()`'s try/catch, so a throw →
`failDesktopBoot(err.message)` (`:400-407`).

**No-config path (v3-defined):** with no usable stored `{ url, token }`,
`getConnection()` throws
`"No Hermes gateway is configured. Open Settings → Gateway to enter your gateway
URL and session token."` → `failDesktopBoot(message)` → the **`BootFailureOverlay`**
renders (visible when `boot.error && !boot.running`,
`apps/desktop/src/components/boot-failure-overlay.tsx:39`). This is a **defined
error, not a hang, and no token fetch occurs**. The `GatewayConnectingOverlay`
gets out of the way on any boot error (returns `null`,
`apps/desktop/src/components/gateway-connecting-overlay.tsx:148`).

**Clarification.** `boot-failure-overlay.tsx` (first-run recovery) does **not**
embed the gateway panel. `BootFailureOverlay` renders recovery *buttons* — Retry
(`:113-117`), Repair (`:119-123`), **Use Local Gateway** (`:125-130`, calls
`applyConnectionConfig({ mode: 'local' })`), Open Logs (`:166`) — and, only for a
remote **OAuth**-reauth failure (`isRemoteReauthFailure`, `boot-failure-reauth.ts:35-46`:
`mode==='remote' && remoteAuthMode==='oauth' && !remoteOauthConnected`), a "Sign in"
button (`:137-164`). It never renders `<GatewaySettings>`. The token-entry panel
lives at **Settings → Gateway** (`apps/desktop/src/app/settings/index.tsx:247-248`),
reached via the titlebar / command palette. Consequences for token-mode
remote-only first-run (known limitations):
- The overlay offers **no token-entry affordance** for a token gateway
  (`isRemoteReauthFailure` is false for token mode → no "Sign in" button).
- The overlay is `fixed inset-0 z-[1400]` (`boot-failure-overlay.tsx:176`) while
  `SettingsView`/`OverlayView` is `z-50` (`apps/desktop/src/app/overlays/overlay-view.tsx:53`),
  and both are siblings in `desktop-controller.tsx` (overlay `:1066`, settings
  `:1073`). So the boot overlay visually **covers** the gateway panel; whether
  the panel can be reached while the overlay is up has not been verified
  interactively.
- "Use Local Gateway" calls `applyConnectionConfig({ mode: 'local' })`; the shim
  **rejects local mode** (remote-only). The overlay swallows the rejection
  (`.catch(() => undefined)`, `:128`), so the button is an inert no-op — correct
  for this build, but it means the overlay's only enabled actions are
  Retry/Repair (both just reload → boot fails again with the same message).

The shim provides the full `*ConnectionConfig` family (§7) so that **once the
panel is reached**, first-run configuration works end-to-end. Whether the panel
should be surfaced directly from the overlay is a separate UI decision, outside
the scope of this connection-layer documentation.

---

## §7 — Gateway-settings connection-config family

Backs `apps/desktop/src/app/settings/gateway-settings.tsx` and
`apps/desktop/src/components/boot-failure-overlay.tsx`. **Types**
(`apps/desktop/src/global.d.ts:51-57`):

```ts
getConnectionConfig:   (profile?: null | string)         => Promise<DesktopConnectionConfig>
saveConnectionConfig:  (payload: DesktopConnectionConfigInput) => Promise<DesktopConnectionConfig>
applyConnectionConfig: (payload: DesktopConnectionConfigInput) => Promise<DesktopConnectionConfig>
testConnectionConfig:  (payload: DesktopConnectionConfigInput) => Promise<DesktopConnectionTestResult>
probeConnectionConfig: (remoteUrl: string)               => Promise<DesktopConnectionProbeResult>
```

### `DesktopConnectionConfig` (`apps/desktop/src/global.d.ts:392-403`)
`{ envOverride, mode, profile, remoteAuthMode, remoteOauthConnected,
remoteTokenPreview: string|null, remoteTokenSet: boolean, remoteUrl }`. Built by
`sanitizeDesktopConnectionConfig` (`apps/desktop/electron/main.cjs:4822-4860`):
`remoteTokenPreview = tokenPreview(token)`, `remoteTokenSet = Boolean(token)`.
**`tokenPreview`** (`connection-config.cjs:202-210`): `null` when empty,
`'set'` when `len <= 8`, else `'...' + last6`. **Never the raw token.**

**Shim `getConnectionConfig(profile)` → `sanitizeConfig`:** remote-only, so
`mode:'remote', remoteAuthMode:'token', remoteOauthConnected:false,
envOverride:false`; `profile` echoes the trimmed scope (or `null`);
`remoteTokenPreview`/`remoteTokenSet` derived from the stored token;
`remoteUrl` = stored URL, or the **default gateway URL as a UI prefill** when
nothing is stored. The prefill is display-only and is never used by
`getConnection()`/`api()` as a connection fallback.

### `DesktopConnectionConfigInput` (`apps/desktop/src/global.d.ts:405-413`)
`{ mode, profile?, remoteAuthMode?, remoteToken?, remoteUrl? }`.
Panel `payload()` (`gateway-settings.tsx:283-289`):
`{ mode: state.mode, profile: scope ?? undefined, remoteAuthMode: authMode,
remoteToken: authMode==='token' ? remoteToken.trim() || undefined : undefined,
remoteUrl: trimmedUrl }`.

**Shim `coerceRemote(payload)`** — mirrors `coerceDesktopConnectionConfig`
(`main.cjs:4872-4908`) token-inheritance (omitted token keeps the stored one) and
`buildRemoteBlock`'s token requirement (`main.cjs:4865-4869`):
- rejects `mode !== 'remote'` (local unsupported) and `remoteAuthMode === 'oauth'`
  (oauth unsupported);
- normalizes the URL; resolves token = `payload.remoteToken.trim()` or the stored
  token; **throws `"Remote gateway session token is required."`** when neither.

- **`saveConnectionConfig`**: `coerceRemote` → write `localStorage` → return
  `sanitizeConfig`. Handler parity: `main.cjs:6372-6377` (write, then sanitize).
  The panel `setState(next)`, clears the token box, toasts "saved"
  (`gateway-settings.tsx:305-315`).
- **`applyConnectionConfig`**: same, **plus reconnect**. **There is no
  `onConnectionApplied` event anywhere in the vendor** (grep of `src` + `electron`
  is empty). "Apply = reconnect" is implemented in Electron as *tear down the
  backend and reload the window*: `main.cjs:6378-6397`
  (`teardownPrimaryBackendAndWait()` + `mainWindow?.reload()`), confirmed by the
  renderer comment "applyConnectionConfig reloads the window from the main
  process" (`boot-failure-overlay.tsx:127`). The shim reproduces this with
  `window.location.reload()` (deferred ~150 ms so the promise resolves and the
  panel's "reconnecting" toast paints first, `gateway-settings.tsx:311-315`);
  boot then re-runs `getConnection()` against the freshly-stored config.

### `DesktopConnectionTestResult` (`apps/desktop/src/global.d.ts:415-419`)
`{ baseUrl, ok, version: string|null }`. Panel reads `result.baseUrl`,
`result.version` (`gateway-settings.tsx:405`).

**Shim `testConnectionConfig`** mirrors `testDesktopConnectionConfig`
(`main.cjs:5118-5173`): (1) `GET {baseUrl}/api/status` (public) for reachability
+ version; (2) a **live WS leg** on `{ws}://…/api/ws?token=…`, because
`/api/status` is public and the token is only truly validated on the WS upgrade.
The WS probe mirrors `probeGatewayWebSocket`
(`apps/desktop/electron/gateway-ws-probe.cjs:45-152`): open → 750 ms grace → ok;
a frame → ok; error / close-before-open / early-close / 10 s timeout → not ok,
with a reason. On WS failure it throws the vendor's "Reached the gateway over
HTTP, but the live WebSocket (/api/ws) connection failed: …" message
(`main.cjs:5160-5165`). A **wrong token** therefore yields a clean failure (the WS
upgrade is refused), not a false-positive "reachable".

### `DesktopConnectionProbeResult` (`apps/desktop/src/global.d.ts:431-438`)
`{ baseUrl, reachable, authMode: 'oauth'|'token'|'unknown', providers, version, error }`.

**Token-vs-OAuth detection.** The panel probes the URL as
the user types (`gateway-settings.tsx:166-205`) and shows the **token box** when
the effective `authMode` is `'token'` (`:209-215,551-569`); until the scheme is
known (`authResolved`, `:230-236`) neither control renders. `probeRemoteAuthMode`
(`main.cjs:5053-5116`) reads the **public** `GET /api/status`:
`authModeFromStatus` (`connection-config.cjs:217-219`) →
`auth_required ? 'oauth' : 'token'`. `/api/status` needs **no token** (verified:
returns 200 with `auth_required:false` unauthenticated). **Shim
`probeConnectionConfig`** reproduces this: normalize → `GET /api/status`
(no token header) → `auth_required:false ⇒ 'token'` (so an `http://…` token
gateway shows the token field); network/parse failure ⇒ `reachable:false`
(never throws, matching `:5071-5080`). `providers` is always `[]` — provider
listing is OAuth-only metadata (`main.cjs:5085-5106`) and oauth is unsupported.

### Deliberately NOT provided
`oauthLoginConnectionConfig` / `oauthLogoutConnectionConfig`
(`global.d.ts:56-57`) — only invoked on OAuth-detected paths
(`gateway-settings.tsx:347,371`; `boot-failure-overlay.tsx:145`), which cannot
occur for a token gateway (`authMode` is `'token'`; `isRemoteReauthFailure` is
false). `revealLogs` is provided (§8).

---

## §8 — Boot-failure recovery support methods (new reachable surface in v3)

v3 is the first shim that can **reach** `BootFailureOverlay` (the no-config path
fails boot on purpose, §6). Its effect body calls, when it becomes visible,
`window.hermesDesktop?.getRecentLogs()` (`boot-failure-overlay.tsx:50-52`) — the
`?.` guards only `hermesDesktop`, so a **missing `getRecentLogs` throws
synchronously and breaks the overlay render**. The click handlers `revealLogs`
(`:166`), `resetBootstrap` (`:115`), `repairBootstrap` (`:121`) have the same
`?.`-on-`hermesDesktop`-only shape. The shim therefore provides all four as safe
stubs (types: `global.d.ts:97-98,178-179`):

| method | shim return |
|--------|-------------|
| `getRecentLogs()` | `{ path: '', lines: [] }` |
| `revealLogs()` | `{ ok: false, path: '', error: 'Logs are unavailable in the iOS web client.' }` |
| `resetBootstrap()` | `{ ok: true }` (Retry then reloads → re-runs boot) |
| `repairBootstrap()` | `{ ok: true }` (Repair then reloads) |

The first-launch **install** overlay stays dormant: it early-returns unless
`typeof desktop.onBootstrapEvent === 'function'`
(`apps/desktop/src/components/desktop-install-overlay.tsx:270-272`), and the shim
does not provide `onBootstrapEvent` / `getBootstrapState`.

---

## §9 — Other bridge methods the shim provides

| method | type source | shim return |
|--------|-------------|-------------|
| `getGatewayWsUrl(profile?)` | `apps/desktop/src/global.d.ts:27` | token WS URL from stored config (§3); throws if no config |
| `revalidateConnection()` | `apps/desktop/src/global.d.ts:23` | `{ ok: <stored config usable>, rebuilt: false }` (nothing to rebuild for a stateless remote; reconnect loop ignores the result, `use-gateway-boot.ts:145`) |
| `touchBackend(profile?)` | `apps/desktop/src/global.d.ts:26` | `{ ok: true }` |
| `getVersion()` | `apps/desktop/src/global.d.ts:182`, `DesktopVersionInfo` `:240-246` | `{ appVersion:'ios-web-shim', electronVersion:'', nodeVersion:'', platform, hermesRoot:'' }` object (consumed at `store/updates.ts:231-234`, guarded) |
| `signalDeepLinkReady()` | `apps/desktop/src/global.d.ts:169` | `{ ok: true }` |
| `on{BootProgress,BackendExit,PowerResume,WindowStateChanged,FocusSession,NotificationAction,DeepLink,ClosePreviewRequested,OpenUpdatesRequested,PreviewFileChanged}` | `apps/desktop/src/global.d.ts` | no-op registrars returning an unsubscribe |

All other feature surfaces (`git`, `terminal`, `themes`, file ops, clipboard/
image, preview helpers, `openExternal`, `openNewSessionWindow`, …) are accessed
either guarded (`?.`) or only behind user actions (settings pages, link clicks,
previews) — never during boot or first paint — so omitting them is safe. Verified
by enumerating every `window.hermesDesktop.` reference in `apps/desktop/src`.

---

## §10 — Storage layer + security

- Key **`hermes.remoteGateway`**, value `{"url": "...", "token": "..."}` JSON in
  `localStorage`.
- `readStoredRaw()` never throws (returns `null` on absent/malformed).
- The token is written **only** to `localStorage` and sent **only** as the
  gateway's own `X-Hermes-Session-Token` REST header and `?token=` WS query param.
  It is **never** logged and **never** embedded in an error message
  (`api()`/`fetchStatus` errors carry only status + server body; the HTML
  diagnostic carries only the URL; the WS-probe reason carries only a close code).
- There is **no dashboard-token scrape** (the v2 `window.__HERMES_SESSION_TOKEN__`
  extraction, `dashboard-token.cjs`, is gone) and **no `credentials: 'include'`**.

---

## §11 — First-Run Connect Screen + Reconnect Recovery (shim-owned, additive)

This confirms §6's finding and builds the fix: `BootFailureOverlay`
(`apps/desktop/src/components/boot-failure-overlay.tsx:176`) is
`fixed inset-0 z-[1400]` and offers **no token-entry control** for a token
gateway — only Retry/Repair/"Use Local Gateway" (rejected here, remote-only)/
Open Logs, confirmed by re-reading the component for this task (no `<input>`,
no `<GatewaySettings>` import). `use-gateway-boot.ts:336,403` is unchanged from
§6: `getConnection()` throws inside `boot()`'s try/catch → `failDesktopBoot`.
The token-entry panel (`gateway-settings.tsx`) is reachable only via
Settings → Gateway, i.e. only **after** a boot has already succeeded once — a
true first run (empty `localStorage`) or a token rotation that breaks the
stored config can never reach it. The shim closes that gap itself, entirely in
`hermes-web-shim.js`, with no vendor patch:

**1. Blocking first-run overlay.** On load, if `readStoredRaw()` yields no
usable `{ url, token }` (the identical condition `requireConnection()` uses),
the shim builds its own full-screen `<div>` — own `<style>` tag, no framework
— and appends it to `document.body` with `z-index: 2147483000`. That is
chosen to sit far above `BootFailureOverlay`'s `z-[1400]` (and above anything
else the vendor could plausibly use) with a lot of headroom, not just barely
above it. Fields: Gateway-URL (prefilled with the module's
`DEFAULT_GATEWAY_URL` UI-only constant), Token (`type=password`, empty,
placeholder "Session-Token"), a "Verbinden" submit button, and an inline
status line. The shim script is a classic `<script>` injected before the
module bundle (`inject-shim.mjs`), so it runs while `<head>` is still being
parsed and `document.body` does not exist yet; mounting is deferred to
`DOMContentLoaded` via a small `whenBodyReady()` helper.

**2. "Verbinden" wiring.** The submit handler calls the existing
`window.hermesDesktop.testConnectionConfig({ mode: 'remote', remoteAuthMode:
'token', remoteUrl, remoteToken })` — the same bridge method the
gateway-settings panel uses (§7), so both surfaces are validated identically
(HTTP `/api/status` reachability + a live `/api/ws` WebSocket probe). Only on
success: the module-local `writeStored()` persists `{ url: result.baseUrl,
token }` (the normalized URL `testConnectionConfig` returned, and the same
EFFECTIVE token value that was just tested) to
`localStorage["hermes.remoteGateway"]`. "Effective" matters when the token
field is left blank: `coerceRemote()` (§7) inherits the stored token for the
live test, and the submit handler mirrors that same inheritance
(`trimmed field value || stored token`) before persisting — so a URL-only
change on the recovery (⚙) overlay re-persists the working stored token
instead of overwriting it with an empty string. If the effective token is
still empty (only reachable if `testConnectionConfig` ever resolved without a
token, which `coerceRemote`'s own "Remote gateway session token is required."
throw prevents today), the handler treats it as a failed attempt and writes
nothing — the same as any other failure. The status line shows a fixed
"Verbunden. Wird neu geladen…", and
`window.location.reload()` runs after a 150 ms delay (mirroring
`applyConnectionConfig`'s own deferred reload, §7) so boot re-runs
`getConnection()` against the freshly-stored config. On failure: **nothing is
written**, and the status line shows a single fixed string,
`"Connection failed. Please check the gateway URL and token."` — never
`err.message`/`String(err)` from the rejected promise, so a future change to
`testConnectionConfig`'s internal error text can't leak anything onto this
screen. (Its current error text is already token-free per §10 and the
hardening below; this screen simply doesn't depend on that continuing to
hold.) The overlay stays open and the fields stay editable for another
attempt. Because nothing is ever written on failure, the dead end "bad token
persisted → boot fails → BootFailureOverlay with no way back in" cannot occur.

**3. Reconnect-recovery affordance.** Independent of whether config exists,
the shim also injects a small fixed button (`⚙`, ~2.25rem circle, ~45%
opacity, bottom-left, `left`/`bottom` computed via
`max(0.75rem, env(safe-area-inset-*))`) at `z-index: 2147483000 - 1` — one
below the connect overlay, so it's naturally covered while the overlay is
open, but otherwise sits above ordinary app content (including a *later*
`BootFailureOverlay`, whose `z-[1400]` this still clears by a wide margin).
Clicking it reopens the same overlay, prefilled with the **stored** URL (not
the default) and an empty token field, and — unlike the first-run case — with
a "Abbrechen" button and Escape-to-close, since a working config already
exists at that point. This lets a user re-enter a rotated token, including
while stuck back in `BootFailureOverlay` after a rotation breaks the stored
config on a later boot, without reinstalling. (Once connected, the vendor's
own Settings → Gateway panel remains reachable too, exactly as in the Mac
app — this affordance only covers the gap where that panel can't be reached
yet.) The button is a single, clearly delimited block
(`ensureRecoveryButton()` + its CSS rule) and is trivially removable by
deleting that function and its call site — nothing else depends on it.

**4. No regression.** Everything above is additive: new top-level `const`s,
functions, and one `whenBodyReady(...)` call appended after the existing
`window.hermesDesktop = { … }` assignment. It reads `readStoredRaw`/
`writeStored`/`DEFAULT_GATEWAY_URL` from the same closure and calls
`window.hermesDesktop.testConnectionConfig` like any external caller would;
it does not modify `api()`, the WS URL builders, `getConnection()`, or any
`*ConnectionConfig` method.

**Token-leak hardening.** While in this area,
`probeWebSocket`'s `new WebSocket(wsUrl)` constructor `catch` previously
surfaced `err.message`/`String(err)` as the failure `reason`. `wsUrl` embeds
`?token=<value>`, and a constructor-thrown `DOMException`/`Error` can
(engine-dependently) include the offending URL in its message — the only
place in the file where that was theoretically possible. That branch now
returns a fixed, generic reason, `'WebSocket-Verbindung fehlgeschlagen.'`,
never derived from the caught error. The `error`/`close` **event** handlers
right below it are unchanged: their reasons are fixed phrases plus, at most, a
numeric close `event.code` — never the URL or token — so those stay as-is.

**Known limitations (not yet verified interactively):** the visual/UX pass
(actual rendering, safe-area behavior on a notched device, dark/light
appearance), and the real-token happy path (successful "Verbinden" →
`localStorage` write → reload → normal boot). Both require exercising the
shim in a real browser/device rather than static code review.

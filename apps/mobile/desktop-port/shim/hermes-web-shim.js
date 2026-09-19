/*
 * Browser shim for `window.hermesDesktop` — remote gateway, token mode.
 *
 * The Hermes desktop renderer (apps/desktop) normally talks to an Electron
 * preload bridge (electron/preload.cjs). In a plain browser (this app runs the
 * renderer bundled inside an iOS app, from capacitor://localhost or a localhost
 * dev server) that bridge is absent, so use-gateway-boot.ts fails immediately
 * with "Desktop IPC bridge is unavailable." This shim reconstructs the exact
 * bridge contract the renderer's boot, gateway-settings panel, and zoom/profile
 * stores rely on, backed by a REMOTE gateway whose URL + session token are read
 * from localStorage (there is no same-origin host and no token scraping).
 *
 * Every shape here is extracted from the hermes-agent v2026.7.7.2 checkout in
 * desktop-port/vendor/; see desktop-port/shim/CONTRACT.md for file:line
 * evidence. Where this disagrees with the task brief's skeleton, CONTRACT.md
 * governs.
 *
 * Security: the session token is stored ONLY in localStorage under
 * `hermes.remoteGateway` (JSON `{ url, token }`). It is NEVER logged, NEVER
 * embedded in an error message, and never sent anywhere except as the gateway's
 * own `X-Hermes-Session-Token` REST header / `?token=` WS query param.
 */
(() => {
  'use strict'

  // localStorage key + JSON shape the connection config is persisted under.
  const STORAGE_KEY = 'hermes.remoteGateway'

  // UI prefill ONLY (optional). If non-empty, it is shown as the default value
  // in the "connect" screen's URL field before anything is saved. Leave empty
  // for a generic build; a self-hoster may set their own gateway URL here for
  // convenience. It is NEVER an implicit fallback for a real connection:
  // getConnection()/api() read the STORED config exclusively.
  const DEFAULT_GATEWAY_URL = ''

  // Shown when boot / a REST call runs with no usable stored config. Actionable,
  // and free of any token material.
  const NO_CONFIG_MESSAGE =
    'No Hermes gateway is configured. Open Settings → Gateway to enter your gateway URL and session token.'

  // ── Stored config layer ─────────────────────────────────────────────────
  // Raw read: returns { url, token } with possibly-empty strings, or null when
  // nothing is stored / the JSON is malformed. Never throws.
  function readStoredRaw() {
    let raw
    try {
      raw = window.localStorage.getItem(STORAGE_KEY)
    } catch {
      return null
    }
    if (!raw) {
      return null
    }
    try {
      const parsed = JSON.parse(raw)
      if (!parsed || typeof parsed !== 'object') {
        return null
      }
      return {
        url: typeof parsed.url === 'string' ? parsed.url.trim() : '',
        token: typeof parsed.token === 'string' ? parsed.token : ''
      }
    } catch {
      return null
    }
  }

  function writeStored(config) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ url: config.url, token: config.token }))
    } catch {
      throw new Error('Could not save the gateway configuration to local storage.')
    }
  }

  // A usable connection needs BOTH a URL and a token; otherwise boot must fail
  // with NO_CONFIG_MESSAGE rather than dial a half-configured / default target.
  function requireConnection() {
    const stored = readStoredRaw()
    if (!stored || !stored.url || !stored.token) {
      throw new Error(NO_CONFIG_MESSAGE)
    }
    return stored
  }

  // ── URL helpers ─────────────────────────────────────────────────────────
  // Mirrors connection-config.cjs:40-63 normalizeRemoteBaseUrl: trim, require
  // http(s), drop hash/search, strip trailing slashes.
  function normalizeBaseUrl(rawUrl) {
    const value = String(rawUrl || '').trim()
    if (!value) {
      throw new Error('Remote gateway URL is required.')
    }
    let parsed
    try {
      parsed = new URL(value)
    } catch (err) {
      throw new Error('Remote gateway URL is not valid: ' + (err && err.message ? err.message : String(err)))
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Remote gateway URL must be http:// or https://, got ' + parsed.protocol)
    }
    parsed.hash = ''
    parsed.search = ''
    parsed.pathname = parsed.pathname.replace(/\/+$/, '')
    return parsed.toString().replace(/\/+$/, '')
  }

  // Mirrors connection-config.cjs:65-71 buildGatewayWsUrl(baseUrl, token):
  //   `${ws|wss}://${host}${prefix}/api/ws?token=${encodeURIComponent(token)}`
  function buildWsUrl(baseUrl, token) {
    const parsed = new URL(baseUrl)
    const wsScheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
    const prefix = parsed.pathname.replace(/\/+$/, '')
    return wsScheme + '://' + parsed.host + prefix + '/api/ws?token=' + encodeURIComponent(token)
  }

  // Mirrors connection-config.cjs:202-210 tokenPreview: masked, never the token.
  function tokenPreview(value) {
    const raw = String(value || '')
    if (!raw) {
      return null
    }
    return raw.length <= 8 ? 'set' : '...' + raw.slice(-6)
  }

  // Mirrors hardening.cjs:13-23 resolveTimeoutMs (default 15_000ms).
  const DEFAULT_FETCH_TIMEOUT_MS = 15000
  function resolveTimeoutMs(value) {
    const parsed = Number(value)
    return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : DEFAULT_FETCH_TIMEOUT_MS
  }

  // ── Single-profile constraint ────────────────────────────────────────────
  // This client binds to ONE remote gateway, which is the primary ("default")
  // profile. The desktop's multi-profile model spawns a separate local backend
  // per HERMES_HOME from the Electron pool and routes profile-scoped requests
  // to it (store/profile.ts:206-263; requests carry `profile` via desktop-fs.ts
  // / desktop-git.ts). None of that maps onto a single remote gateway, so profile
  // switching is DISABLED here: rather than silently issuing a named profile's
  // request against the one unscoped backend (the mismatch #64962's review
  // flagged), any non-primary profile throws. Mirrors normalizeProfileKey
  // (store/profile.ts:21): trimmed, empty/null → 'default'.
  function normalizeProfileKey(name) {
    return String(name == null ? '' : name).trim() || 'default'
  }
  function isPrimaryProfile(profile) {
    return normalizeProfileKey(profile) === 'default'
  }
  function assertPrimaryProfile(profile) {
    if (!isPrimaryProfile(profile)) {
      throw new Error(
        'This iOS client is bound to a single remote gateway (the default profile). ' +
          'Profile switching is not available in this build; change the gateway in Settings instead.',
      )
    }
  }

  // ── api(request) ───────────────────────────────────────────────────────
  // Mirrors the Electron `hermes:api` handler (main.cjs:6555-6596) + fetchJson
  // (main.cjs:3285-3352). Returns PARSED JSON directly (the generic <T>), NOT an
  // { ok, status, body } envelope. Contract:
  //   - request: { path (incl. query string), method?, body?, timeoutMs?, profile? }
  //     No separate `query` field (hermes.ts:204-208). A non-primary `profile`
  //     throws (single-profile client — see assertPrimaryProfile above); the
  //     primary/absent profile resolves to the one remote backend.
  //   - headers: Content-Type + X-Hermes-Session-Token ALWAYS sent.
  //   - success: empty body → null; HTML body → throw diagnostic; else JSON.parse.
  //   - failure (status >= 400): throw Error(`${status}: ${body}`).
  // v3 change: the connection source is the STORED remote config (was:
  // window.location.origin + scraped token). A 401 is NO LONGER auto-recovered —
  // it surfaces as `${status}: ${body}` so the renderer's recovery UI takes over.
  async function api(request) {
    assertPrimaryProfile(request && request.profile)
    const conn = requireConnection()
    const path = String((request && request.path) || '')
    const method = (request && request.method) || 'GET'
    const hasBody = request && request.body !== undefined
    const timeoutMs = resolveTimeoutMs(request && request.timeoutMs)

    let res
    try {
      res = await fetch(conn.url + path, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'X-Hermes-Session-Token': conn.token
        },
        body: hasBody ? JSON.stringify(request.body) : undefined,
        signal: AbortSignal.timeout(timeoutMs)
      })
    } catch (err) {
      if (err && err.name === 'TimeoutError') {
        throw new Error('Timed out connecting to Hermes backend after ' + timeoutMs + 'ms')
      }
      throw err
    }

    const text = await res.text()

    if (!res.ok) {
      // fetchJson error format (main.cjs:3314): `${status}: ${body}`.
      throw new Error(res.status + ': ' + (text || res.statusText))
    }
    if (!text) {
      return null
    }
    // A 2xx HTML body means the request fell through to index.html (an
    // unregistered /api path); surface a clear diagnostic. Mirrors fetchJson
    // (main.cjs:3321-3335), which checks BOTH the body sniff and the
    // `content-type: text/html` header.
    const contentType = res.headers.get('content-type') || ''
    if (/^\s*<(?:!doctype|html)/i.test(text) || /text\/html/i.test(contentType)) {
      throw new Error(
        'Expected JSON from ' + conn.url + path + ' but got HTML (status ' + res.status + '). ' +
          'The endpoint is likely missing on the Hermes backend.'
      )
    }
    try {
      return JSON.parse(text)
    } catch {
      throw new Error('Invalid JSON from ' + conn.url + path + ' (status ' + res.status + '): ' + text.slice(0, 200))
    }
  }

  // ── getConnection(profile?) ──────────────────────────────────────────────
  // Full HermesConnection (global.d.ts:359-373) from the stored config. Every
  // required field is populated so title-bar / window-state consumers never read
  // undefined. `mode:'remote'` drives isRemoteGateway()/isRemoteMode();
  // `authMode:'token'` selects the token WS path in resolveGatewayWsUrl
  // (apps/shared/src/websocket-url.ts:38). NO stored config → throw
  // NO_CONFIG_MESSAGE, which boot() catches → failDesktopBoot() → BootFailureOverlay
  // (use-gateway-boot.ts:336,400-407; §6).
  async function getConnection(profile) {
    assertPrimaryProfile(profile)
    const conn = requireConnection()
    return {
      baseUrl: conn.url,
      isFullscreen: false,
      mode: 'remote',
      authMode: 'token',
      nativeOverlayWidth: 0,
      source: 'settings',
      token: conn.token,
      wsUrl: buildWsUrl(conn.url, conn.token),
      logs: [],
      windowButtonPosition: null,
      // Primary profile: this client is bound to one remote gateway, so the
      // connection descriptor is never scoped to a named profile. Keeps
      // desktopFsProfile()/request routing on 'default'.
      profile: null
    }
  }

  // ── Gateway-settings connection-config family ────────────────────────────
  // Backs gateway-settings.tsx + boot-failure-overlay.tsx. Remote token-only:
  // local/oauth modes are rejected (constraints: "Local/Cloud … must not be
  // selectable/functional"). Returns/echoes the DesktopConnectionConfig /
  // *Result shapes the panel destructures.

  // sanitize → DesktopConnectionConfig (global.d.ts:392-403). Mirrors
  // sanitizeDesktopConnectionConfig (main.cjs:4822-4860): remote-only, masked
  // token preview, envOverride=false. `remoteUrl` prefills with the default URL
  // when nothing is stored (UI convenience only — never a connection fallback).
  function sanitizeConfig(profileScope) {
    const stored = readStoredRaw()
    const scopeKey = String(profileScope == null ? '' : profileScope).trim() || null
    const token = stored ? stored.token : ''
    return {
      envOverride: false,
      mode: 'remote',
      profile: scopeKey,
      remoteAuthMode: 'token',
      remoteOauthConnected: false,
      remoteTokenPreview: tokenPreview(token),
      remoteTokenSet: Boolean(token),
      remoteUrl: stored && stored.url ? stored.url : DEFAULT_GATEWAY_URL
    }
  }

  // Validate + resolve a DesktopConnectionConfigInput (global.d.ts:405-413) into
  // a { url, token } stored block. Enforces remote token-mode. Mirrors
  // coerceDesktopConnectionConfig token inheritance (main.cjs:4872-4908): an
  // omitted token keeps the stored one.
  function coerceRemote(payload) {
    if (!payload || payload.mode !== 'remote') {
      throw new Error('This build only connects to a remote Hermes gateway; local mode is unavailable.')
    }
    if (payload.remoteAuthMode === 'oauth') {
      throw new Error('This gateway uses OAuth, which this build does not support. Use a session-token gateway.')
    }
    const stored = readStoredRaw()
    const rawUrl =
      payload.remoteUrl != null && String(payload.remoteUrl).trim() ? payload.remoteUrl : (stored && stored.url) || ''
    const url = normalizeBaseUrl(rawUrl)
    const incoming = typeof payload.remoteToken === 'string' ? payload.remoteToken.trim() : ''
    const token = incoming || (stored ? stored.token : '')
    if (!token) {
      throw new Error('Remote gateway session token is required.')
    }
    return { url, token }
  }

  // GET the PUBLIC /api/status (no token needed). Used by probe + the HTTP leg
  // of the connection test. Throws a clean `${status}: ${body}` / timeout error.
  async function fetchStatus(baseUrl, token) {
    const headers = token ? { 'X-Hermes-Session-Token': token } : {}
    let res
    try {
      res = await fetch(baseUrl + '/api/status', { method: 'GET', headers, signal: AbortSignal.timeout(8000) })
    } catch (err) {
      if (err && err.name === 'TimeoutError') {
        throw new Error('Timed out reaching the gateway at ' + baseUrl + ' after 8000ms')
      }
      throw err
    }
    const text = await res.text()
    if (!res.ok) {
      throw new Error(res.status + ': ' + (text || res.statusText))
    }
    try {
      return text ? JSON.parse(text) : null
    } catch {
      return null
    }
  }

  // Live WebSocket validation, mirroring gateway-ws-probe.cjs:45-152. /api/status
  // is public, so the TOKEN is only truly validated on the /api/ws upgrade:
  //   open (+ grace) or a frame → ok; error / close-before-open / early-close /
  //   timeout → not ok, with a reason (never containing the token).
  function probeWebSocket(wsUrl) {
    const CONNECT_TIMEOUT_MS = 10000
    const READY_GRACE_MS = 750
    return new Promise(resolve => {
      let settled = false
      let opened = false
      let connectTimer = null
      let graceTimer = null
      let socket

      const finish = result => {
        if (settled) {
          return
        }
        settled = true
        if (connectTimer !== null) clearTimeout(connectTimer)
        if (graceTimer !== null) clearTimeout(graceTimer)
        try {
          if (socket && socket.close) socket.close()
        } catch {
          // best-effort teardown
        }
        resolve(result)
      }

      try {
        socket = new WebSocket(wsUrl)
      } catch {
        // A WebSocket-constructor throw (malformed URL, disallowed scheme, …)
        // can embed the full wsUrl — which contains `?token=<value>` — in the
        // DOMException/Error message on some engines. Unlike the close/error
        // event reasons below (which only ever carry a close code or a fixed
        // phrase), this text is NOT known to be token-free, so never surface
        // err.message/String(err) here: a fixed, generic reason only.
        finish({ ok: false, reason: 'WebSocket connection failed.' })
        return
      }

      socket.addEventListener('open', () => {
        if (settled) return
        opened = true
        graceTimer = setTimeout(() => finish({ ok: true }), READY_GRACE_MS)
      })
      socket.addEventListener('message', () => finish({ ok: true }))
      socket.addEventListener('error', () => {
        finish({
          ok: false,
          reason: opened
            ? 'The gateway accepted the connection then closed it (token rejected?).'
            : 'The gateway closed the WebSocket before it opened.'
        })
      })
      socket.addEventListener('close', event => {
        if (settled) return
        const code = event && typeof event.code === 'number' ? event.code : null
        const base = opened
          ? 'The gateway accepted the connection then closed it (token rejected?).'
          : 'The gateway closed the WebSocket before it opened.'
        finish({ ok: false, reason: code ? base + ' (code ' + code + ')' : base })
      })

      connectTimer = setTimeout(
        () => finish({ ok: false, reason: 'Timed out after ' + CONNECT_TIMEOUT_MS + 'ms waiting for the WebSocket to open.' }),
        CONNECT_TIMEOUT_MS
      )
    })
  }

  // No-op event registrar: returns an unsubscribe function, matching the real
  // preload.cjs contract (every on*() returns () => void).
  const noopUnsub = () => () => {}

  window.hermesDesktop = {
    // ── Core REST / connection ──────────────────────────────────────────
    api,
    getConnection,
    // resolveGatewayWsUrl (websocket-url.ts:34-64) mints via this in token mode
    // and falls back to conn.wsUrl; both resolve to the same token WS URL here.
    getGatewayWsUrl: async profile => {
      assertPrimaryProfile(profile)
      const conn = requireConnection()
      return buildWsUrl(conn.url, conn.token)
    },
    // { ok, rebuilt } per global.d.ts:23. Nothing to "rebuild" for a stateless
    // remote; ok reflects whether a usable stored config is present.
    revalidateConnection: async () => {
      const raw = readStoredRaw()
      return { ok: Boolean(raw && raw.url && raw.token), rebuilt: false }
    },
    // Idle-reaper keepalive (global.d.ts:26). No pool here → always ok for the
    // primary profile; a named profile throws (single-profile client).
    touchBackend: async profile => {
      assertPrimaryProfile(profile)
      return { ok: true }
    },

    // ── Gateway settings (gateway-settings.tsx + boot-failure-overlay.tsx) ──
    getConnectionConfig: async profile => sanitizeConfig(profile),
    saveConnectionConfig: async payload => {
      const block = coerceRemote(payload)
      writeStored(block)
      return sanitizeConfig(payload && payload.profile)
    },
    applyConnectionConfig: async payload => {
      const block = coerceRemote(payload)
      writeStored(block)
      const next = sanitizeConfig(payload && payload.profile)
      // Reconnect == a full renderer reload: the vendor's applyConnectionConfig
      // tears down the backend and reloads the window from the main process
      // (main.cjs:6392-6393; boot-failure-overlay.tsx:127). Boot then re-runs
      // getConnection() against the freshly-stored config. Deferred so this
      // promise resolves and the panel's success toast paints first.
      setTimeout(() => {
        try {
          window.location.reload()
        } catch {
          // ignore — nothing else to do in a headless context
        }
      }, 150)
      return next
    },
    testConnectionConfig: async payload => {
      const { url, token } = coerceRemote(payload)
      // 1. HTTP reachability + version via the public /api/status.
      const status = await fetchStatus(url, token)
      // 2. Live WS leg — the actual token/transport check (status is public).
      const ws = await probeWebSocket(buildWsUrl(url, token))
      if (!ws.ok) {
        throw new Error(
          'Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ' +
            ws.reason +
            ' The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or the gateway auth/origin guard.'
        )
      }
      return { ok: true, baseUrl: url, version: (status && status.version) || null }
    },
    // Probe how a gateway authenticates, WITHOUT sending credentials, so the
    // panel shows the token box for an http(s) token gateway. Mirrors
    // probeRemoteAuthMode (main.cjs:5053-5116): /api/status.auth_required=false
    // ⇒ 'token'. Network/parse failure ⇒ reachable:false (never throws). OAuth
    // provider listing is intentionally omitted (oauth is unsupported here).
    probeConnectionConfig: async remoteUrl => {
      let baseUrl
      try {
        baseUrl = normalizeBaseUrl(remoteUrl)
      } catch (err) {
        return {
          baseUrl: String(remoteUrl || ''),
          reachable: false,
          authMode: 'unknown',
          providers: [],
          version: null,
          error: err && err.message ? err.message : String(err)
        }
      }
      let status
      try {
        status = await fetchStatus(baseUrl, null)
      } catch (err) {
        return {
          baseUrl,
          reachable: false,
          authMode: 'unknown',
          providers: [],
          version: null,
          error: err && err.message ? err.message : String(err)
        }
      }
      const authRequired = Boolean(status && status.auth_required)
      return {
        baseUrl,
        reachable: true,
        authMode: authRequired ? 'oauth' : 'token',
        providers: [],
        version: (status && status.version) || null,
        error: null
      }
    },

    // ── Boot-failure recovery support (boot-failure-overlay.tsx) ──────────
    // v3 is the first shim that can REACH the boot-failure overlay (the no-config
    // path fails boot on purpose). getRecentLogs is called as the overlay becomes
    // visible (:51) via `window.hermesDesktop?.getRecentLogs()` — the `?.` guards
    // only hermesDesktop, so a MISSING method throws synchronously and breaks the
    // overlay. revealLogs (:166) + resetBootstrap (:115) + repairBootstrap (:121)
    // are click handlers with the same shape. All are safe stubs; Retry/Repair
    // simply reload (which re-runs boot).
    getRecentLogs: async () => ({ path: '', lines: [] }),
    revealLogs: async () => ({ ok: false, path: '', error: 'Logs are unavailable in the iOS web client.' }),
    resetBootstrap: async () => ({ ok: true }),
    repairBootstrap: async () => ({ ok: true }),

    // ── Profile ─────────────────────────────────────────────────────────
    // profile.get → DesktopActiveProfile { profile: string | null }
    // (global.d.ts:386-390). null → the primary profile; boot defaults to
    // 'default' (use-gateway-boot.ts:365). profile.set formerly echoed the name
    // as a silent no-op (store/profile.ts:140 calls it unguarded), which let the
    // UI believe it had switched while requests still hit the one backend. It now
    // accepts only the primary profile and throws otherwise, so switchProfile()
    // fails loudly instead of desyncing UI state from the backend.
    profile: {
      get: async () => ({ profile: null }),
      set: async name => {
        assertPrimaryProfile(name)
        return { profile: null }
      }
    },

    // ── Version ─────────────────────────────────────────────────────────
    // DesktopVersionInfo object (global.d.ts:240-246), not a string —
    // store/updates.ts:231-234 stores it into $desktopVersion.
    getVersion: async () => ({
      appVersion: 'ios-web-shim',
      electronVersion: '',
      nodeVersion: '',
      platform: (navigator && navigator.platform) || 'web',
      hermesRoot: ''
    }),

    // ── Zoom (store/zoom.ts:16-21) ──────────────────────────────────────
    // Guarded by `?.zoom`, but once present get()/onChanged() run unguarded and
    // get() is destructured as `{ percent }`. 100% = the $zoomPercent default.
    zoom: {
      get: async () => ({ level: 0, percent: 100 }),
      setPercent: () => {},
      onChanged: noopUnsub
    },

    // ── Boot-progress (use-gateway-boot.ts:224-227) ─────────────────────
    // onBootProgress + getBootProgress are called SYNCHRONOUSLY in the boot
    // effect, before boot()'s try/catch — a missing method throws an uncaught
    // TypeError and the overlay hangs at 2%. getBootProgress returns a
    // DesktopBootProgress (global.d.ts:451-459) with running:true so applying it
    // never prematurely hides the overlay (store/boot.ts:41); the renderer's own
    // steps + completeDesktopBoot() drive dismissal.
    onBootProgress: noopUnsub,
    getBootProgress: async () => ({
      error: null,
      fakeMode: false,
      message: 'Connecting to Hermes…',
      phase: 'remote.connect',
      progress: 5,
      running: true,
      timestamp: Date.now()
    }),

    // ── Lifecycle / event registrars (all → unsubscribe) ────────────────
    // onBackendExit is called unguarded in the boot effect (line 321); the rest
    // are guarded but provided for a complete, no-throw surface.
    onBackendExit: noopUnsub,
    onPowerResume: noopUnsub,
    onWindowStateChanged: noopUnsub,
    onFocusSession: noopUnsub,
    onNotificationAction: noopUnsub,
    onDeepLink: noopUnsub,
    onClosePreviewRequested: noopUnsub,
    onOpenUpdatesRequested: noopUnsub,
    onPreviewFileChanged: noopUnsub,
    signalDeepLinkReady: async () => ({ ok: true })
  }

  // ── Connect screen (first-run + reconnect recovery; CONTRACT.md §11) ────
  // Self-contained and purely additive: its own DOM/CSS, wired only to
  // testConnectionConfig() above (plus the module-local writeStored/
  // readStoredRaw/DEFAULT_GATEWAY_URL). Does not alter api()/WS/panel
  // behavior — nothing below this point is part of the hermesDesktop bridge
  // contract. Two entry points:
  //   1. True first run (no usable stored config): a blocking full-screen
  //      overlay, z-index 2147483000 — far above the vendor
  //      BootFailureOverlay's z-[1400] (boot-failure-overlay.tsx:176) —
  //      since that overlay has no token-entry affordance at all
  //      (CONTRACT.md §6) and Settings → Gateway is unreachable before boot
  //      ever completes once.
  //   2. A small, always-mounted recovery affordance (bottom-left, safe-area
  //      aware) that reopens the same overlay, so a rotated token can be
  //      re-entered without reinstalling — including while the renderer is
  //      stuck back in BootFailureOverlay after the rotation broke the
  //      stored config on a later boot.
  // Both paths funnel into the SAME testConnectionConfig() the gateway-
  // settings panel uses. localStorage is written ONLY after that call
  // resolves successfully, using the normalized baseUrl it returns — a bad
  // token is therefore never persisted.

  const CONNECT_OVERLAY_ID = 'hermes-shim-connect-overlay'
  const RECOVERY_BUTTON_ID = 'hermes-shim-recovery-button'
  const CONNECT_STYLE_ID = 'hermes-shim-connect-style'
  const OVERLAY_Z_INDEX = 2147483000

  // Fixed, generic failure copy. Deliberately NEVER err.message/String(err):
  // testConnectionConfig()'s own throw text is already token-free by
  // construction (CONTRACT.md §10, and the WS-constructor hardening above),
  // but this screen takes no dependency on that staying true under a future
  // change elsewhere in the file — it never echoes caught error text.
  const CONNECT_FAILURE_MESSAGE = 'Connection failed. Please check the gateway URL and token.'

  function whenBodyReady(cb) {
    if (document.body) {
      cb()
    } else {
      // The shim is a classic <script> injected right before the module
      // bundle in <head> (inject-shim.mjs), so it runs synchronously while
      // <head> is still being parsed — document.body is null at this point.
      document.addEventListener('DOMContentLoaded', cb, { once: true })
    }
  }

  function ensureConnectStyles() {
    if (document.getElementById(CONNECT_STYLE_ID)) {
      return
    }
    const style = document.createElement('style')
    style.id = CONNECT_STYLE_ID
    // Nous look: accent #0053FD, light bg #F8FAFF / text #17171A, dark bg
    // #0D2F86 / text #FFE6CB, system font stack. Safe-area insets on both
    // the overlay's outer padding and the recovery button's position.
    style.textContent =
      // iOS keyboard safety: top:0/left:0/right:0 + explicit height (NOT
      // inset:0/bottom:0 — those fight an explicit height) so the second
      // `height` declaration (100dvh) can cleanly override the `100vh`
      // fallback in engines that support dvh, while older engines simply
      // ignore the invalid dvh value and keep 100vh. flex-direction:column +
      // justify-content:flex-start top-aligns the card instead of centering
      // it, so when the iOS software keyboard shrinks the visual viewport the
      // card stays reachable instead of being shoved off-screen; overflow-y
      // auto + -webkit-overflow-scrolling let the user scroll the focused
      // field into view above the keyboard.
      '#' + CONNECT_OVERLAY_ID + '{position:fixed;top:0;left:0;right:0;height:100vh;height:100dvh;' +
      'z-index:' + OVERLAY_Z_INDEX + ';display:flex;flex-direction:column;' +
      'align-items:center;justify-content:flex-start;box-sizing:border-box;' +
      'overflow-y:auto;-webkit-overflow-scrolling:touch;' +
      'padding:max(3rem,calc(env(safe-area-inset-top) + 1.5rem)) max(1.25rem,env(safe-area-inset-right))' +
      ' max(1.25rem,env(safe-area-inset-bottom)) max(1.25rem,env(safe-area-inset-left));' +
      'background:#F8FAFF;color:#17171A;' +
      'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}' +
      '@media (prefers-color-scheme:dark){#' + CONNECT_OVERLAY_ID + '{background:#0D2F86;color:#FFE6CB;}}' +
      '#' + CONNECT_OVERLAY_ID + ' *{box-sizing:border-box;}' +
      '.hermes-shim-card{width:100%;max-width:24rem;display:flex;flex-direction:column;gap:0.7rem;' +
      'padding:1.75rem;border-radius:1rem;background:rgba(255,255,255,0.92);' +
      'border:1px solid rgba(0,83,253,0.18);box-shadow:0 20px 60px rgba(13,47,134,0.18);}' +
      '@media (prefers-color-scheme:dark){.hermes-shim-card{background:rgba(13,47,134,0.6);' +
      'border-color:rgba(255,230,203,0.22);box-shadow:0 20px 60px rgba(0,0,0,0.5);}}' +
      '.hermes-shim-title{margin:0;font-size:1.125rem;font-weight:600;letter-spacing:-0.01em;}' +
      '.hermes-shim-subtitle{margin:0 0 0.2rem;font-size:0.8125rem;opacity:0.72;}' +
      '.hermes-shim-label{font-size:0.75rem;font-weight:500;opacity:0.8;}' +
      '.hermes-shim-input{width:100%;font:inherit;font-size:0.9375rem;padding:0.5rem 0.65rem;' +
      'border-radius:0.5rem;border:1px solid rgba(23,23,26,0.22);background:#ffffff;color:#17171A;}' +
      '@media (prefers-color-scheme:dark){.hermes-shim-input{border-color:rgba(255,230,203,0.3);' +
      'background:rgba(255,255,255,0.08);color:#FFE6CB;}}' +
      '.hermes-shim-input:focus{outline:2px solid #0053FD;outline-offset:1px;}' +
      '.hermes-shim-status{min-height:1.1rem;font-size:0.8125rem;}' +
      '.hermes-shim-status--pending{opacity:0.7;}' +
      '.hermes-shim-status--error{color:#C81E3A;}' +
      '@media (prefers-color-scheme:dark){.hermes-shim-status--error{color:#FFB4C2;}}' +
      '.hermes-shim-status--ok{color:#0053FD;}' +
      '@media (prefers-color-scheme:dark){.hermes-shim-status--ok{color:#FFE6CB;}}' +
      '.hermes-shim-actions{display:flex;justify-content:flex-end;gap:0.5rem;margin-top:0.2rem;}' +
      '.hermes-shim-btn{font:inherit;font-size:0.875rem;font-weight:600;padding:0.5rem 1rem;' +
      'border-radius:0.5rem;border:none;cursor:pointer;}' +
      '.hermes-shim-btn--primary{background:#0053FD;color:#ffffff;}' +
      '.hermes-shim-btn--primary:disabled{opacity:0.6;cursor:default;}' +
      '.hermes-shim-btn--ghost{background:transparent;color:inherit;opacity:0.7;}' +
      '.hermes-shim-btn--ghost:hover{opacity:1;}'
    document.head.appendChild(style)
  }

  // True first run only: no stored config at all, or a partial one (URL with
  // no token, or vice versa) — the same condition requireConnection() uses to
  // decide boot must fail. Mirrors that check rather than re-deriving it.
  function isFirstRunNoConfig() {
    const stored = readStoredRaw()
    return !(stored && stored.url && stored.token)
  }

  function setConnectStatus(el, text, kind) {
    el.textContent = text
    el.className = 'hermes-shim-status' + (kind ? ' hermes-shim-status--' + kind : '')
  }

  function hideRecoveryButton() {
    const btn = document.getElementById(RECOVERY_BUTTON_ID)
    if (btn) {
      btn.style.display = 'none'
    }
  }

  function showRecoveryButton() {
    const btn = document.getElementById(RECOVERY_BUTTON_ID)
    if (btn) {
      btn.style.display = ''
    }
  }

  function closeConnectOverlay() {
    const el = document.getElementById(CONNECT_OVERLAY_ID)
    if (el) {
      el.remove()
    }
    showRecoveryButton()
  }

  // Builds + mounts the overlay. Idempotent (a second call while one is
  // already open is a no-op) and makes NO network call itself — only the
  // "Connect" click handler it wires up does that.
  function openConnectOverlay(opts) {
    const dismissible = Boolean(opts && opts.dismissible)
    if (document.getElementById(CONNECT_OVERLAY_ID)) {
      return
    }
    ensureConnectStyles()
    hideRecoveryButton()

    // URL prefill: the stored URL when reopening (recovery case), else the
    // default gateway URL as a UI convenience on a true first run — never an
    // implicit connection fallback (only getConnection()/api() read storage
    // for that, and only after a successful "Connect"). Token is always
    // blank, even when reopening with a stored config.
    const stored = readStoredRaw()
    const prefillUrl = stored && stored.url ? stored.url : DEFAULT_GATEWAY_URL

    const overlay = document.createElement('div')
    overlay.id = CONNECT_OVERLAY_ID
    overlay.setAttribute('role', 'dialog')
    overlay.setAttribute('aria-modal', 'true')
    overlay.setAttribute('aria-label', 'Connect to Hermes')
    // Static template — no untrusted value is interpolated into markup;
    // url/token are assigned via .value below, never concatenated into HTML.
    overlay.innerHTML =
      '<form class="hermes-shim-card" novalidate>' +
      '<h1 class="hermes-shim-title">Connect to Hermes</h1>' +
      '<p class="hermes-shim-subtitle">Enter your gateway URL and session token.</p>' +
      '<label class="hermes-shim-label" for="hermes-shim-url">Gateway URL</label>' +
      '<input class="hermes-shim-input" id="hermes-shim-url" type="text" inputmode="url" ' +
      'autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false" ' +
      'placeholder="https://your-gateway…" />' +
      '<label class="hermes-shim-label" for="hermes-shim-token">Token</label>' +
      '<input class="hermes-shim-input" id="hermes-shim-token" type="password" autocomplete="off" ' +
      'autocapitalize="off" autocorrect="off" spellcheck="false" placeholder="Session token" />' +
      '<div class="hermes-shim-status" id="hermes-shim-status" aria-live="polite"></div>' +
      '<div class="hermes-shim-actions">' +
      (dismissible
        ? '<button type="button" class="hermes-shim-btn hermes-shim-btn--ghost" id="hermes-shim-cancel">Cancel</button>'
        : '') +
      '<button type="submit" class="hermes-shim-btn hermes-shim-btn--primary" id="hermes-shim-connect">Connect</button>' +
      '</div>' +
      '</form>'

    document.body.appendChild(overlay)

    const form = overlay.querySelector('form')
    const urlInput = overlay.querySelector('#hermes-shim-url')
    const tokenInput = overlay.querySelector('#hermes-shim-token')
    const statusEl = overlay.querySelector('#hermes-shim-status')
    const connectBtn = overlay.querySelector('#hermes-shim-connect')
    const cancelBtn = overlay.querySelector('#hermes-shim-cancel')

    urlInput.value = prefillUrl
    tokenInput.value = ''

    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => closeConnectOverlay())
    }
    if (dismissible) {
      overlay.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
          closeConnectOverlay()
        }
      })
    }

    form.addEventListener('submit', e => {
      e.preventDefault()
      const urlValue = urlInput.value
      const tokenValue = tokenInput.value

      setConnectStatus(statusEl, 'Verbinde…', 'pending')
      connectBtn.disabled = true
      urlInput.disabled = true
      tokenInput.disabled = true
      if (cancelBtn) cancelBtn.disabled = true

      window.hermesDesktop
        .testConnectionConfig({
          mode: 'remote',
          remoteAuthMode: 'token',
          remoteUrl: urlValue,
          remoteToken: tokenValue
        })
        .then(result => {
          // Persist exactly what was just validated: the normalized baseUrl
          // testConnectionConfig() returns, and the EFFECTIVE token it
          // tested — not necessarily the raw field value. coerceRemote()
          // (~line 262-263) inherits the stored token when the field is
          // left empty, and testConnectionConfig validates that inherited
          // token. Reproduce the same inheritance here so an empty-field
          // submit re-persists the working stored token instead of
          // overwriting it with '' (which would pass the live test using
          // the old token, then strand the user on the next boot with
          // NO_CONFIG_MESSAGE — silent data loss after a "Verbunden" toast).
          const trimmedToken = String(tokenValue || '').trim()
          const stored = readStoredRaw()
          const effectiveToken = trimmedToken || (stored ? stored.token : '')
          if (!effectiveToken) {
            // Defense in depth: coerceRemote() already throws "Remote
            // gateway session token is required." when neither the field
            // nor storage has a token, so testConnectionConfig() should
            // never resolve here with an empty effective token. If it
            // somehow does, never persist an empty token — treat it as a
            // failed attempt instead.
            setConnectStatus(statusEl, CONNECT_FAILURE_MESSAGE, 'error')
            connectBtn.disabled = false
            urlInput.disabled = false
            tokenInput.disabled = false
            if (cancelBtn) cancelBtn.disabled = false
            return
          }
          writeStored({ url: result.baseUrl, token: effectiveToken })
          setConnectStatus(statusEl, 'Connected. Reloading…', 'ok')
          // Reload = the established apply path (no onConnectionApplied
          // event exists; mirrors applyConnectionConfig's own deferred
          // reload above) so boot re-runs getConnection() against the
          // freshly-stored config.
          setTimeout(() => {
            try {
              window.location.reload()
            } catch {
              // ignore — nothing else to do in a headless context
            }
          }, 150)
        })
        .catch(() => {
          // Generic only — see CONNECT_FAILURE_MESSAGE above. Nothing is
          // written to storage on this path; the overlay stays open.
          setConnectStatus(statusEl, CONNECT_FAILURE_MESSAGE, 'error')
          connectBtn.disabled = false
          urlInput.disabled = false
          tokenInput.disabled = false
          if (cancelBtn) cancelBtn.disabled = false
        })
    })

    // Intentionally NOT auto-focused: focusing on mount summons the iOS
    // software keyboard immediately on first load, shrinking the visual
    // viewport before the user has even seen the card. The user taps the
    // field when ready; the overlay's own layout (see ensureConnectStyles)
    // keeps the tapped field reachable once the keyboard is up.
  }

  // Auto-recovery watcher (replaces the old always-on ⚙ recovery button, which
  // read as a "log out" trap). When a config IS stored but boot fails — the
  // classic stale/rotated-token case after a server restart — the renderer
  // lands in the vendor BootFailureOverlay (boot-failure-overlay.tsx:176), which
  // offers Retry/Repair/local/logs but NO token-entry affordance. Detect that
  // overlay via MutationObserver and reopen the SAME connect screen (dismissible,
  // URL prefilled, token blank) so a rotated token can be re-entered without
  // reinstalling. Never fires on a true first run (handled by the blocking
  // overlay below) and disconnects after firing once, so there is no loop.
  //
  // The overlay's root carries `fixed inset-0 z-[1400]` and, uniquely, NOT
  // `backdrop-blur-md` (the only other z-[1400] layer, desktop-install-overlay,
  // always has it) — so the selector matches the boot-failure overlay alone,
  // independent of locale/copy. The connect overlay's z-index (2147483000) is
  // far above 1400, so it paints on top; cancelling it leaves the user on the
  // BootFailureOverlay's own Retry, and a reload re-arms this watcher.
  const BOOT_FAILURE_SELECTOR = 'div.fixed.inset-0.z-\\[1400\\]:not(.backdrop-blur-md)'

  function watchForBootFailure() {
    let triggered = false
    let observer = null

    const tryRecover = () => {
      if (triggered) {
        return
      }
      // A true first run is owned by the blocking overlay; never race it.
      if (isFirstRunNoConfig()) {
        return
      }
      // Connect screen already open (e.g. user tapped Retry mid-flow) → nothing to do.
      if (document.getElementById(CONNECT_OVERLAY_ID)) {
        return
      }
      if (!document.querySelector(BOOT_FAILURE_SELECTOR)) {
        return
      }
      triggered = true
      if (observer) {
        observer.disconnect()
      }
      openConnectOverlay({ dismissible: true })
    }

    observer = new MutationObserver(tryRecover)
    observer.observe(document.body, { childList: true, subtree: true })
    // Catch an overlay that is already mounted when the watcher starts.
    tryRecover()
  }

  // Mount once the DOM exists. The blocking connect overlay auto-opens ONLY on a
  // true first run (no usable stored config); otherwise the boot-failure watcher
  // arms itself. Neither path makes a network call — that only happens from the
  // "Connect" submit handler above, on explicit user action.
  whenBodyReady(() => {
    if (isFirstRunNoConfig()) {
      openConnectOverlay({ dismissible: false })
    } else {
      watchForBootFailure()
    }
  })

  // ── iPhone layout adaptation (v3-design) ────────────────────────────────
  // Purely additive presentation layer: a viewport-meta rewrite + one injected
  // stylesheet. Nothing above this section (bridge contract, connect overlay)
  // is modified.
  //
  // Three problems, three blocks:
  //   1. "Not static": the page pans/bounces/zooms like a website.
  //      → viewport meta maximum-scale=1 + user-scalable=no, a gesturestart
  //        guard (WKWebView pinch), and a fixed-position lock on html/body.
  //        Inner scrollers are untouched: the vendor already scopes scrolling
  //        to overflow-y-auto containers (chat transcript
  //        components/assistant-ui/thread, session list
  //        app/chat/sidebar/virtual-session-list, settings panels) — the lock
  //        only pins html/body/#root, which vendor styles.css:445-453 already
  //        declares overflow:hidden.
  //   2. Safe areas: viewport-fit=cover makes env(safe-area-inset-*) nonzero;
  //      the vendor derives ALL top-chrome offsets from --titlebar-height /
  //      --titlebar-controls-* custom properties set INLINE on
  //      [data-slot="sidebar-wrapper"] (app-shell.tsx:166-186), so overriding
  //      those vars with !important (author !important beats a non-!important
  //      inline style) safe-area-shifts the titlebar band, all three fixed
  //      button clusters, the session-list top padding, notifications and the
  //      right-rail in one place. The bottom home-indicator inset is applied as
  //      padding-bottom on <main>, which pushes BOTH the pane area and the
  //      statusbar up off the indicator in one move — so the statusbar stays a
  //      compact strip directly above the indicator (no tall empty band) and
  //      the composer, anchored to the pane bottom, sits right above it.
  //   3. Desktop chrome: under (pointer: coarse) only — hide the two
  //      touch-useless titlebar buttons (keyboard-shortcut panel, haptics
  //      mute: web-haptics rides navigator.vibrate, which iOS WKWebView does
  //      not implement), grow the titlebar controls + statusbar row to touch
  //      size, widen the composer to the full usable width, and make full-screen
  //      overlays (settings / command center) near-full-bleed.
  const IOS_STYLE_ID = 'hermes-shim-ios-style'

  // Rewrite the viewport meta BEFORE the bundle mounts (the shim executes
  // synchronously in <head>, after the meta tag, before the module bundle).
  // viewport-fit=cover is the precondition for env(safe-area-inset-*) to be
  // nonzero; maximum-scale=1 + user-scalable=no kill pinch/double-tap zoom
  // (including the iOS auto-zoom on <16px inputs). WebKit re-evaluates the
  // meta on mutation, so setAttribute is sufficient.
  function applyMobileViewportMeta() {
    const content = 'width=device-width, initial-scale=1.0, viewport-fit=cover, maximum-scale=1, user-scalable=no'
    let meta = document.querySelector('meta[name="viewport"]')
    if (!meta) {
      meta = document.createElement('meta')
      meta.setAttribute('name', 'viewport')
      ;(document.head || document.documentElement).appendChild(meta)
    }
    meta.setAttribute('content', content)
  }

  function iosLayoutCss() {
    return (
      // ── 1. Static app lock ─────────────────────────────────────────────
      // position:fixed on html/body is the canonical iOS anti-rubber-band:
      // the document itself can never pan, so WKWebView's scroll view has
      // nothing to bounce. overscroll-behavior:none stops chained overscroll
      // out of the inner scrollers (iOS 16+). touch-action:manipulation
      // removes double-tap-zoom (and the 350ms tap delay) at the shell level
      // WITHOUT restricting pan gestures, so every overflow-y-auto container
      // below #root keeps scrolling (touch-action resolution stops at the
      // nearest scroll container, never reaching body).
      'html,body{position:fixed !important;inset:0 !important;width:100% !important;height:100% !important;' +
      'overflow:hidden !important;overscroll-behavior:none !important;}' +
      'html{-webkit-text-size-adjust:100%;}' +
      'html,body,#root{touch-action:manipulation;}' +
      '#root{height:100% !important;overflow:hidden !important;}' +
      // ── 2. Safe-area core (env() is 0 on desktop → byte-identical there) ─
      // The shell provider defines the chrome geometry inline
      // (app-shell.tsx:166-186): --titlebar-height:34px,
      // --titlebar-controls-top:6px, --titlebar-controls-left:14px (the shim
      // reports windowButtonPosition:null → TITLEBAR_EDGE_INSET),
      // --titlebar-tools-right:0.75rem (nativeOverlayWidth:0). Growing
      // --titlebar-height by the top inset cascades into every consumer:
      // chat header band (titlebar.ts:22), session-list top padding
      // (chat/sidebar/index.tsx:1047), right-sidebar/terminal pt
      // (right-sidebar/index.tsx:78, desktop-controller.tsx:1277),
      // notifications top offset, floating HUD, thread top spacer.
      "[data-slot='sidebar-wrapper']{" +
      '--titlebar-height:calc(34px + env(safe-area-inset-top)) !important;' +
      '--titlebar-controls-top:calc(6px + env(safe-area-inset-top)) !important;' +
      '--titlebar-controls-left:max(14px, env(safe-area-inset-left)) !important;' +
      '--titlebar-tools-right:max(0.75rem, env(safe-area-inset-right)) !important;}' +
      // Elements that use the var as their HEIGHT with items-center content
      // (chat <header>, right-rail tab strip) must push that content below
      // the island: pad by the same inset so the visual 34px band sits at the
      // bottom of the grown band. (Also matches the aria-hidden drag strips,
      // where padding is inert.) Class name is the Tailwind-compiled form of
      // h-(--titlebar-height).
      '.h-\\(--titlebar-height\\){padding-top:env(safe-area-inset-top);}' +
      // Side insets on <main> for the rounded corners in landscape (portrait: 0).
      // Top stays 0 (the titlebar owns the island); bottom is NOT padded here on
      // purpose (round 3) — see the statusbar rule below.
      'main.relative.z-3{' +
      'padding-left:env(safe-area-inset-left);padding-right:env(safe-area-inset-right);}' +
      // Bottom chrome (round 3): the statusbar BACKGROUND must reach the true
      // bottom edge (into the rounded corners), while its TEXT stays above the
      // home indicator. So the home-indicator inset lives HERE as the statusbar's
      // own padding-bottom (its bg paints through the padding box), and its
      // height grows by the same inset — content sits in the top 1.25rem row,
      // the inset zone below is filled by the bar's bg. Because the statusbar is
      // the last flex child of <main> and <main> no longer reserves the inset,
      // the pane area (and the composer anchored to its bottom, absolute bottom-0)
      // ends exactly at this full-height statusbar's TOP — no gap, no rise beyond
      // the bar. Side padding clears the corners in landscape.
      // The bar fills all the way to the true bottom edge (bg reaches into the
      // rounded corners/home-indicator zone), but the TEXT stays above the
      // corner safe zone: padding-bottom = env(safe-area-inset-bottom) keeps it
      // clear of the rounded corners (on square-cornered devices the inset is
      // 0 -> no empty margin, automatic device adaptation). So the zone doesn't
      // look empty, the bar gets a subtle background of its own + a separator
      // line on top (theme-adaptive via currentColor). Side padding protects
      // the bottom corner ends in portrait too (where inset-left/right is 0).
      "footer[data-slot='statusbar']{" +
      'height:calc(1.25rem + env(safe-area-inset-bottom)) !important;' +
      'min-height:calc(1.25rem + env(safe-area-inset-bottom)) !important;' +
      'padding-bottom:env(safe-area-inset-bottom) !important;' +
      'padding-left:max(0.75rem, env(safe-area-inset-left)) !important;' +
      'padding-right:max(0.75rem, env(safe-area-inset-right)) !important;' +
      'background:color-mix(in srgb, currentColor 11%, transparent) !important;' +
      'border-top:1px solid color-mix(in srgb, currentColor 22%, transparent) !important;}' +
      // ── 3. Touch chrome (gated: never affects desktop-browser debugging) ─
      '@media (pointer: coarse){' +
      // Touch-useless titlebar buttons, scoped to the fixed z-70 clusters
      // (titlebar-controls.tsx:158,188) so codicons elsewhere are untouched:
      // the keybind-panel button (hardware-keyboard feature) and the haptics
      // mute toggle (navigator.vibrate is unavailable in iOS WKWebView).
      'div.fixed.z-70 button:has(> .codicon-keyboard),' +
      'div.fixed.z-70 button:has(> .codicon-mute),' +
      'div.fixed.z-70 button:has(> .codicon-unmute){display:none !important;}' +
      // Touch-sized titlebar controls: 28px squares (up from 20×22). The cluster
      // width math adapts automatically — AppShell computes it from
      // var(--titlebar-control-size) (app-shell.tsx:139).
      //
      // Round 3 — reclaim the top line: shrink the header BAND to exactly the
      // control height (1.75rem = 28px) and pin the fixed control clusters at
      // EXACTLY env(safe-area-inset-top) (was 3px + inset). Effect: the header
      // content region is 28px tall sitting flush below the island (no extra
      // desktop 34px band, no +3px centering offset — the "Aufschlag" from
      // round 1). --titlebar-height overrides the ungated 34px+inset here, so the
      // chat header band and its consumers all tighten by 6px and the title/
      // buttons align right under the island. (The .h-(--titlebar-height)
      // padding-top:env(top) rule still clears the island above this 28px band.)
      // --titlebar-content-inset (chat header's left padding reserving space for
      // the two left buttons): controls-left + 2×(size+gap) + 0.5rem breathing.
      "[data-slot='sidebar-wrapper']{" +
      '--titlebar-control-size:1.75rem;' +
      '--titlebar-control-height:1.75rem;' +
      '--titlebar-height:calc(1.75rem + env(safe-area-inset-top)) !important;' +
      '--titlebar-controls-top:env(safe-area-inset-top) !important;' +
      '--titlebar-content-inset:calc(max(14px, env(safe-area-inset-left)) + 2 * (1.75rem + 0.25rem) + 0.5rem) !important;}' +
      // Round 3 — the transcript over-reserved top space: its content top padding
      // is pt-[calc(var(--titlebar-height)-0.5rem)] (thread/list.tsx:151), which
      // on a phone grew to ~inset+22px because --titlebar-height carries the
      // island inset. But the chat <header> is in-flow ABOVE the scroller, so the
      // transcript does NOT need to re-clear the island — only a small breathing
      // gap below the header. Pin that padding to a flat 1rem. (data-slot is on
      // both the empty + populated content; the empty state also has py-8 which
      // this trims at the top — harmless.) Scrollability is untouched: this is
      // only the first child's top padding inside the scroller.
      "[data-slot='aui_thread-content']{padding-top:1rem !important;}" +
      // Statusbar content row grows to a tappable 1.75rem; the home-indicator
      // inset is ADDED on top (bg fills it, see the ungated rule above) so the
      // text still clears the indicator.
      "footer[data-slot='statusbar']{" +
      'height:calc(1.75rem + env(safe-area-inset-bottom)) !important;' +
      'min-height:calc(1.75rem + env(safe-area-inset-bottom)) !important;}' +
      // Composer: on the desktop it is a centered card capped at --composer-width
      // (48.75rem) with a 2rem side gutter and its visible surface = root − 10px
      // of transparent drag padding (styles.css:1098-1104,
      // composer/index.tsx:814). On a phone the 2rem cap leaves it looking
      // narrow and off. Widen the docked (not popped-out) composer to the full
      // usable width — only the side safe-area insets (min 0.5rem) subtracted —
      // keeping the +10px drag-padding compensation so the visible surface lands
      // exactly inside those gutters; it stays centered via its own
      // left-1/2 / -translate-x-1/2. Anchoring (absolute bottom-0 of the pane)
      // is untouched, so the transcript scroller above is unaffected.
      // The popout (undocked floating composer) is disabled on touch via a
      // localStorage flag (see disableComposerPopout()), so the composer is
      // always docked. Only adjust the width: the vendor caps the docked
      // composer at calc(100%-2rem) (composer/index.tsx:814), which looks
      // narrow on the phone. Expand it to the full usable width (minus the
      // side insets, +10px drag-padding compensation); the bottom anchoring
      // (absolute bottom-0 of the pane) and the vendor's own centering remain
      // untouched.
      "[data-slot='composer-root']:not([data-popped-out]){" +
      'width:calc(100% - max(0.5rem, env(safe-area-inset-left)) - max(0.5rem, env(safe-area-inset-right)) + 10px) !important;' +
      'max-width:none !important;}' +
      // Round 3 — close the transcript→composer gap at the bottom. The vendor
      // reserves a clearance spacer below the last message (thread/list.tsx:328)
      // of height --thread-last-message-clearance = composer-measured-height +
      // status + 2rem (styles.css:336). But the scroller already extends UNDER
      // the composer surface (--thread-viewport-height = 100% - measured +
      // surface, styles.css:377-380), so the composer only actually OVERLAPS the
      // scroller by --composer-surface-measured-height. The extra
      // (measured - surface, i.e. the composer's non-surface padding/controls) +
      // the 2rem desktop breathing over-reserve ~70-80px on a phone → the big
      // empty gap when scrolled to the bottom. Re-anchor the spacer to the ACTUAL
      // overlap: surface height + status + a 0.75rem margin. Never smaller than
      // the surface, so the last message still clears the composer; scrollability
      // untouched (only the trailing spacer's height changes).
      "[data-slot='aui_composer-clearance']{" +
      'height:calc(var(--composer-surface-measured-height) + var(--status-stack-measured-height) + 0.25rem) !important;}' +
      // Full-screen overlays (settings, command center): the desktop floats
      // a card inset by titlebar-height+10px on ALL sides
      // (overlay-view.tsx:58) — with the grown var that would waste ~100px
      // per side on a phone. Pin the card to the safe area instead, and
      // reset --titlebar-height INSIDE the card (a plain descendant
      // declaration beats the inherited value) so the card-internal offsets
      // (close button, split-layout top padding) keep desktop proportions —
      // the card boundary already clears island + home indicator.
      "div[role='presentation'].fixed.inset-0.z-50{" +
      'padding:max(0.625rem, env(safe-area-inset-top)) max(0.625rem, env(safe-area-inset-right)) ' +
      'max(0.625rem, env(safe-area-inset-bottom)) max(0.625rem, env(safe-area-inset-left)) !important;' +
      '--titlebar-height:34px;}' +
      '}'
    )
  }

  // Inject the stylesheet twice on purpose:
  //   - synchronously now, so the static lock + safe-area vars apply from the
  //     first paint (the parser is still inside <head>, so the vendor
  //     stylesheet <link> will land AFTER this node — every rule that must
  //     beat vendor CSS therefore carries !important);
  //   - again on DOMContentLoaded, where appendChild MOVES the existing node
  //     to the end of <head>, behind the vendor stylesheet, so the
  //     non-!important rules also win the equal-specificity cascade.
  function ensureIosLayoutStyles() {
    let style = document.getElementById(IOS_STYLE_ID)
    if (!style) {
      style = document.createElement('style')
      style.id = IOS_STYLE_ID
      style.textContent = iosLayoutCss()
    }
    document.head.appendChild(style)
  }

  // Disable the composer popout (undocked floating composer) on the phone.
  // The vendor initializes $composerPoppedOut from the localStorage flag
  // 'hermes.desktop.composerPopout.enabled' (store/composer-popout.ts:113,
  // storedBoolean(key,false)). If the flag is true, the composer starts out
  // floating freely (position:fixed, inline bottom/right) and ends up in the
  // middle of the text. This shim runs as a classic <script> BEFORE the
  // deferred app module, so we remove the flag (+ the persisted position)
  // before the store reads it -> the composer always boots docked. On a pure
  // iPhone client the popout is pointless anyway.
  function disableComposerPopout() {
    try {
      window.localStorage.removeItem('hermes.desktop.composerPopout.enabled')
      window.localStorage.removeItem('hermes.desktop.composerPopout.position')
    } catch (e) { /* localStorage may be unavailable in edge cases */ }
  }

  disableComposerPopout()
  applyMobileViewportMeta()
  ensureIosLayoutStyles()
  // Registered AFTER the connect-screen mount above, so this runs second and
  // the iOS sheet ends up last in <head> (also after the connect overlay's
  // own style element — their selectors are disjoint except the recovery
  // button, where this sheet must win).
  whenBodyReady(() => ensureIosLayoutStyles())

  // WKWebView can honor pinch gestures regardless of the viewport meta;
  // gesturestart is the iOS-proprietary hook that fires for ANY pinch, and
  // preventDefault() there suppresses the zoom without touching one-finger
  // pans (scrolling in the inner containers is unaffected).
  document.addEventListener('gesturestart', e => e.preventDefault(), { passive: false })
})()

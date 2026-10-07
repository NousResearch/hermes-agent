/**
 * `<webview>` attach-time hardening.
 *
 * Chat windows run with `webviewTag: true` (session-windows.ts) because the
 * preview pane embeds a real `<webview>` for local/remote page previews. That
 * flag is a renderer capability: ANY script running in the renderer can create
 * a `<webview>` element, and the element's own attributes — `nodeintegration`,
 * `preload`, `webpreferences`, `allowpopups`, `disablewebsecurity` — are what
 * Electron uses to build the guest's webPreferences. The preview pane sets safe
 * ones, but nothing forces every other creator to.
 *
 * `will-attach-webview` is the only place the main process gets to overrule
 * that, which is why Electron's security checklist ("Verify WebView options
 * before creation") puts the check here rather than trusting the markup. This
 * module holds the pure decision so it can be unit-tested without Electron,
 * matching how the rest of electron/*.ts splits testable logic out of main.ts.
 *
 * This is defence in depth: reaching it requires script execution in the
 * renderer already. The point is that such a foothold must not be promotable to
 * Node — a `<webview nodeintegration>` would hand the guest `require()` in-process.
 */

/** webPreferences keys that must never survive attach, whatever the markup asked for. */
const FORBIDDEN_PREFERENCE_KEYS = [
  'allowRunningInsecureContent',
  'contextIsolation',
  'experimentalFeatures',
  'nodeIntegration',
  'nodeIntegrationInSubFrames',
  'nodeIntegrationInWorker',
  'preload',
  'sandbox',
  'webSecurity',
  'webviewTag'
] as const

/**
 * Rewrite a guest's webPreferences to the only shape Desktop supports.
 *
 * Deletes rather than overwrites the Node-bearing keys the guest asked for, then
 * re-asserts the safe values, so an unknown future key spelled like `preload`
 * cannot survive by being set twice. `nodeIntegration` is written explicitly
 * (not merely deleted) so the result is self-describing at the call site.
 */
export function hardenedWebviewPreferences(requested: object = {}): Record<string, unknown> {
  const hardened: Record<string, unknown> = { ...requested }

  for (const key of FORBIDDEN_PREFERENCE_KEYS) {
    delete hardened[key]
  }

  hardened.contextIsolation = true
  hardened.nodeIntegration = false
  hardened.nodeIntegrationInSubFrames = false
  hardened.nodeIntegrationInWorker = false
  hardened.sandbox = true
  hardened.webSecurity = true
  hardened.webviewTag = false

  return hardened
}

/**
 * Mutate Electron's live `webPreferences` object into the hardened shape.
 *
 * `will-attach-webview` hands the handler the real object and honours whatever
 * is left on it after the listener returns — so a key must be *deleted*, not
 * merely overwritten. `Object.assign(target, hardened())` would silently keep a
 * guest-supplied `preload`, which is the one key with no safe value to write.
 */
export function applyWebviewHardening<T extends object>(target: T): T {
  const hardened = hardenedWebviewPreferences(target)
  // Electron's WebPreferences has no index signature; the delete pass needs one.
  const view = target as Record<string, unknown>

  for (const key of Object.keys(view)) {
    if (!(key in hardened)) {
      delete view[key]
    }
  }

  return Object.assign(target, hardened)
}

/**
 * Should this `<webview>` be allowed to attach at all?
 *
 * A guest carrying its own `preload` is the one case worth refusing outright
 * rather than sanitising: a preload runs before page script with the guest's
 * privileges, and no legitimate Desktop surface sets one. Everything else is
 * repairable by {@link hardenedWebviewPreferences}, and destroying those guests
 * would break the preview pane for no security gain.
 */
export function shouldBlockWebviewAttach(params: { preload?: unknown } = {}): boolean {
  return typeof params.preload === 'string' && params.preload.length > 0
}

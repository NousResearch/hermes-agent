/**
 * Guest → host event bus for the browser pane's interactive tooling
 * (element picker, component tree, design mode).
 *
 * The webview `preload` is stripped for security (browser-webview-security.cjs),
 * so injected guest scripts have NO ipc/contextBridge channel back to the host.
 * The one channel the host already listens on is the `<webview>` `console-message`
 * event. So discrete guest events ride a NONCE-bound sentinel console line:
 * the guest runs `console.log(SENTINEL + nonce + ':' + json)`, the pane's console
 * handler detects it, validates, dispatches here, and suppresses it from the
 * visible console panel. Mirrors the macrotask-deferred CustomEvent pattern in
 * `composer/focus.ts`.
 *
 * SECURITY — guest pages are UNTRUSTED. `<webview>.executeJavaScript` runs in the
 * page main world, so a determined page can read the nonce from the injected
 * overlay source and forge a sentinel. The nonce only stops accidental collisions
 * and casual spoofing. The real containment lives at the call sites:
 *   - the sentinel carries only a REF/coords, never authoritative element data —
 *     the host re-pulls the real description via executeJavaScript(inspect…);
 *   - a guest event NEVER auto-acts on the agent. Only a host-side user gesture
 *     ("Send to agent" / "Apply") may submit. Treat everything here as a hint.
 */

const SENTINEL = ' __hermesGuest:'
const MAX_SENTINEL_BYTES = 16 * 1024
const GUEST_EVENT = 'hermes:browser-guest-event'

export type BrowserGuestEventKind = 'picked' | 'style-probe' | 'tree-select'

export interface BrowserGuestEvent {
  kind: BrowserGuestEventKind
  /** Element reference (`@e<n>` or `@s<hash>`) the guest reported. Untrusted. */
  ref: string
  tabId: string
  /** Optional probe result for design-mode style apply/revert acks. */
  ok?: boolean
}

/** Per-tab nonce the host bakes into the injected overlay source. */
const nonces = new Map<string, string>()

export const GUEST_SENTINEL_PREFIX = SENTINEL

function makeNonce(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    return Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('')
  }

  // Last-resort fallback (no Web Crypto). The nonce isn't a security boundary —
  // it only suppresses console noise; real data comes from the host re-pull.
  return `n${Date.now().toString(36)}${Math.floor(Math.random() * 1e9).toString(36)}`
}

export function issueGuestNonce(tabId: string): string {
  const nonce = makeNonce()

  nonces.set(tabId, nonce)

  return nonce
}

export function clearGuestNonce(tabId: string): void {
  nonces.delete(tabId)
}

/**
 * Parse one `console-message` string. Returns a validated guest event only when
 * the line is a well-formed, current-nonce sentinel for this tab; otherwise null
 * so the caller keeps treating the line as a normal console message (a real page
 * log that merely contains the prefix is never silently dropped — the nonce makes
 * accidental collision negligible).
 */
export function parseGuestSentinel(tabId: string, message: unknown): BrowserGuestEvent | null {
  if (typeof message !== 'string' || !message.startsWith(SENTINEL) || message.length > MAX_SENTINEL_BYTES) {
    return null
  }

  const body = message.slice(SENTINEL.length)
  const separator = body.indexOf(':')

  if (separator <= 0) {
    return null
  }

  const nonce = body.slice(0, separator)

  if (!nonce || nonces.get(tabId) !== nonce) {
    return null
  }

  let payload: unknown

  try {
    payload = JSON.parse(body.slice(separator + 1))
  } catch {
    return null
  }

  if (!payload || typeof payload !== 'object') {
    return null
  }

  const record = payload as Record<string, unknown>
  const kind = record.kind
  const ref = record.ref

  if (typeof ref !== 'string' || !ref) {
    return null
  }

  if (kind !== 'picked' && kind !== 'tree-select' && kind !== 'style-probe') {
    return null
  }

  return {
    kind,
    ref,
    tabId,
    ...(typeof record.ok === 'boolean' ? { ok: record.ok } : {})
  }
}

export function dispatchBrowserGuestEvent(event: BrowserGuestEvent): void {
  if (typeof window === 'undefined') {
    return
  }

  window.setTimeout(() => window.dispatchEvent(new CustomEvent<BrowserGuestEvent>(GUEST_EVENT, { detail: event })), 0)
}

export function onBrowserGuestEvent(
  kind: BrowserGuestEventKind,
  handler: (event: BrowserGuestEvent) => void
): () => void {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const listener = (event: Event) => {
    const detail = (event as CustomEvent<BrowserGuestEvent>).detail

    if (detail && detail.kind === kind) {
      handler(detail)
    }
  }

  window.addEventListener(GUEST_EVENT, listener)

  return () => window.removeEventListener(GUEST_EVENT, listener)
}

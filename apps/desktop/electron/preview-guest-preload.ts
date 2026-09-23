// Logic half of the preview pane's guest preload. The wiring half lives in
// `preview-guest-preload-entry.ts` (the bundled preload itself); this split
// keeps the handoff rules unit-testable in a plain Node environment.
//
// Mirrored by `PREVIEW_EXTERNAL_CHANNEL` in src/lib/preview-external.ts — the
// renderer tsconfig cannot import from electron/.
export const GUEST_EXTERNAL_CHANNEL = 'preview-open-external'
export const GUEST_WIDGET_INTENT_CHANNEL = 'preview-widget-intent'

interface GuestEventTarget {
  closest(selector: string): { href: string } | null
}

export interface GuestClickEvent {
  button?: number
  isTrusted?: boolean
  target?: unknown
}

export interface GuestHandoffHost {
  addEventListener(type: string, listener: (event: GuestClickEvent | GuestWidgetIntentEvent) => void, capture?: boolean): void
  sendToHost(channel: string, ...args: unknown[]): void
}

export interface GuestWidgetIntentEvent {
  detail?: unknown
}

/**
 * Wire the DOM-capture listener that forwards a clicked `_blank` anchor's
 * absolute URL to the host. `host` is injected so the entry can pass the
 * guest's `document` and `ipcRenderer`, and tests can drive the same rules
 * without Electron or a DOM.
 *
 * Only a trusted primary-button click qualifies. `isTrusted` is what keeps
 * this bridge out of the gesture-less forced-navigation class
 * (GHSA-9f4c-93c8-jc8g): page script can `dispatchEvent` a synthetic click
 * on an anchor, but it cannot forge `isTrusted`.
 */
export function installGuestExternalHandoff(host: GuestHandoffHost): void {
  host.addEventListener(
    'click',
    event => {
      const click = event as GuestClickEvent

      if (click.isTrusted !== true || (click.button ?? 0) !== 0) {
        return
      }

      const target = click.target as GuestEventTarget | null

      if (!target || typeof target.closest !== 'function') {
        return
      }

      // `closest` climbs through the click's own DOM, which this isolated
      // preload world shares with the page: an inner `<span>` inside the
      // anchor resolves to the enclosing `<a target="_blank">` the same way
      // it would for page script.
      const anchor = target.closest('a[target="_blank"]')

      if (!anchor || typeof anchor.href !== 'string' || anchor.href === '') {
        return
      }

      // `anchor.href` is the browser-resolved absolute URL, not the raw
      // attribute, so relative and protocol-relative hrefs arrive fully
      // qualified for the host's scheme admission check.
      host.sendToHost(GUEST_EXTERNAL_CHANNEL, anchor.href)
    },
    true
  )
}

/** Forward only the token-scoped intent event installed by the host renderer.
 * The preload exposes no Electron API to the page; it merely carries an
 * already-validated primitive payload back to its owning webview. */
export function installGuestWidgetIntentHandoff(host: GuestHandoffHost): void {
  host.addEventListener('hermes-widget-intent', event => {
    const detail = (event as GuestWidgetIntentEvent).detail

    if (!detail || typeof detail !== 'object') {
      return
    }

    const { prompt, token } = detail as { prompt?: unknown; token?: unknown }

    if (typeof token !== 'string' || typeof prompt !== 'string' || !prompt.trim()) {
      return
    }

    host.sendToHost(GUEST_WIDGET_INTENT_CHANNEL, token, prompt)
  })
}

// Logic half of the preview pane's guest preload. The wiring half lives in
// `preview-guest-preload-entry.ts` (the bundled preload itself); this split
// keeps the handoff rules unit-testable in a plain Node environment.

export const GUEST_EXTERNAL_CHANNEL = 'preview-open-external'

interface GuestEventTarget {
  closest(selector: string): { href: string } | null
}

export interface GuestHandoffHost {
  addEventListener(type: 'click', listener: (event: unknown) => void, capture?: boolean): void
  sendToHost(channel: string, ...args: unknown[]): void
}

/**
 * Wire the DOM-capture listener that forwards a clicked `_blank` anchor's
 * absolute URL to the host. `host` is injected so the entry can pass the
 * guest's `document` and `ipcRenderer`, and tests can drive the same rules
 * without Electron or a DOM.
 */
export function installGuestExternalHandoff(host: GuestHandoffHost): void {
  host.addEventListener(
    'click',
    event => {
      const target = (event as { target?: unknown }).target as GuestEventTarget | null

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

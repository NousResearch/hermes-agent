// Logic half of the preview pane's guest preload. The wiring half lives in
// `preview-guest-preload-entry.ts` (the bundled preload itself); this split
// keeps the handoff rules unit-testable in a plain Node environment.
//
// Mirrored by `PREVIEW_EXTERNAL_CHANNEL` in src/lib/preview-external.ts — the
// renderer tsconfig cannot import from electron/.
import {
  designInPage,
  type DesignInPage,
  parsePreviewDesignCommand,
  PREVIEW_DESIGN_COMMAND_CHANNEL,
  PREVIEW_DESIGN_EVENT_CHANNEL,
  type PreviewDesignBridgeEvent
} from '@hermes/shared/preview-design'

export const GUEST_EXTERNAL_CHANNEL = 'preview-open-external'

interface GuestEventTarget {
  closest(selector: string): { href: string } | null
}

export interface GuestClickEvent {
  button?: number
  isTrusted?: boolean
  target?: unknown
}

export interface GuestHandoffHost {
  addEventListener(type: 'click', listener: (event: GuestClickEvent) => void, capture?: boolean): void
  sendToHost(channel: string, ...args: unknown[]): void
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
      if (event.isTrusted !== true || (event.button ?? 0) !== 0) {
        return
      }

      const target = event.target as GuestEventTarget | null

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

export interface GuestDesignBridgeHost {
  on(channel: string, listener: (_event: unknown, payload: unknown) => void): void
  sendToHost(channel: string, payload: PreviewDesignBridgeEvent): void
}

/** Run Design Mode in the preload's isolated world: the inspected page shares
 * the DOM, but cannot replace the controller or forge host messages. */
export function installGuestDesignBridge(doc: Document, host: GuestDesignBridgeHost): void {
  let active: { api: DesignInPage; session: number } | null = null
  let generation = 0
  let latestSession = 0
  let pendingSession: number | null = null

  const send = (event: PreviewDesignBridgeEvent) => host.sendToHost(PREVIEW_DESIGN_EVENT_CHANNEL, event)

  const stop = (session: number) => {
    if (active?.session !== session && pendingSession !== session) {
      return
    }

    generation += 1
    pendingSession = null

    if (active?.session === session) {
      const current = active
      active = null
      current.api.teardown()
    }

    send({ session, type: 'stopped' })
  }

  const pump = async (session: number, api: DesignInPage) => {
    while (active?.session === session && active.api === api) {
      const event = await api.wait()

      if (active?.session !== session || active.api !== api) {
        return
      }

      if (event.type === 'end') {
        active = null
        send({ session, type: 'end' })

        return
      }

      send({ session, type: 'select', selection: event.selection })
    }
  }

  const start = async (session: number) => {
    if (session < latestSession) {
      return
    }

    if (active?.session === session) {
      send({ session, type: 'ready' })

      return
    }

    latestSession = session
    pendingSession = session
    const ownGeneration = ++generation

    if (active) {
      const previous = active
      active = null
      previous.api.teardown()
      send({ session: previous.session, type: 'stopped' })
    }

    if (!doc.documentElement) {
      await new Promise<void>(resolve => doc.addEventListener('DOMContentLoaded', () => resolve(), { once: true }))
    }

    if (ownGeneration !== generation) {
      return
    }

    try {
      const api = designInPage(doc)
      pendingSession = null
      active = { api, session }
      api.install()
      send({ session, type: 'ready' })
      void pump(session, api)
    } catch (error) {
      pendingSession = null
      active = null
      send({
        message: error instanceof Error ? error.message.slice(0, 240) : 'Design Mode could not start.',
        session,
        type: 'error'
      })
    }
  }

  host.on(PREVIEW_DESIGN_COMMAND_CHANNEL, (_event, payload) => {
    const command = parsePreviewDesignCommand(payload)

    if (!command) {
      return
    }

    if (command.type === 'start') {
      void start(command.session)
    } else if (command.type === 'stop') {
      stop(command.session)
    } else if (active?.session === command.session) {
      active.api.selectLayer(command.index)
    }
  })
}

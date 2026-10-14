import { type PreviewInputEvent, type PreviewInputHandle, toWebviewInputSpace } from './preview-input'

/** The existing webview capability only; no new renderer privileges. */
export interface PreviewGuest {
  isConnected: boolean
  getWebContentsId: () => number
  getZoomFactor?: () => number
  executeJavaScript: (code: string) => Promise<unknown>
  sendInputEvent: (event: PreviewInputEvent) => void | Promise<void>
  focus: () => void
}

/** Capture once, before any await. Cleanup can only address this generation. */
export function capturePreviewGuest<T extends PreviewGuest>(
  current: () => T | null,
  mayFocus: (guest: T) => boolean = () => true
): PreviewInputHandle | null {
  const guest = current()

  if (!guest || !guest.isConnected) {
    return null
  }

  const generation = guest.getWebContentsId()
  const zoom = guest.getZoomFactor?.()

  const assertSurviving = () => {
    if (!guest.isConnected || guest.getWebContentsId() !== generation) {
      throw new Error('The original preview guest is no longer available.')
    }
  }

  const assertCurrent = () => {
    assertSurviving()

    if (current() !== guest) {
      throw new Error('The preview guest was replaced.')
    }
  }

  const send = (event: PreviewInputEvent) => guest.sendInputEvent(toWebviewInputSpace(event, zoom))

  return {
    identity: guest,
    assertCurrent,
    focus: () => {
      assertCurrent()

      if (mayFocus(guest)) {
        guest.focus()
      }
    },
    run: async code => {
      assertCurrent()
      const result = await guest.executeJavaScript(code)
      assertCurrent()

      return result
    },
    send: event => {
      assertCurrent()

      return send(event)
    },
    release: event => {
      assertSurviving()

      return send(event)
    }
  }
}

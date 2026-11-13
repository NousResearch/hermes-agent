import { describe, expect, it, vi } from 'vitest'

import { ANNOTATE_CROP_PAD, annotateInPage, type AnnotateInPage } from '@/lib/preview-annotate'
import { deferred } from '@/test/deferred'

import {
  bindPreviewExecuteJavaScript,
  captureAnnotateCrop,
  overlayInstallScript,
  type PreviewAnnotateGuest,
  waitAnnotateEvent
} from './preview-annotate-host'
import { withPreviewGuestActivity } from './preview-drive-channels'

function withoutScheduling<T>(
  this: PreviewAnnotateGuest,
  operation: (execute: PreviewAnnotateGuest['executeJavaScript']) => Promise<T>
) {
  return operation(this.executeJavaScript)
}

describe('preview annotate host', () => {
  it('does not evaluate guest template interpolations while wrapping the overlay source', () => {
    const snippet = '${rect.height}'
    const source = '(function(){return `' + snippet + '`})(document)'
    const script = overlayInstallScript(source)

    expect(script).toContain(snippet)
    expect(() => overlayInstallScript(source)).not.toThrow()
  })

  it('calls executeJavaScript as a method so Electron can read this.getWebContentsId', async () => {
    const webview = {
      executeJavaScript(this: { getWebContentsId: () => number }, code: string) {
        expect(this.getWebContentsId()).toBe(7)
        expect(code).toBe('1+1')

        return Promise.resolve(2)
      },
      getWebContentsId: () => 7
    }

    await expect(bindPreviewExecuteJavaScript(webview)('1+1')).resolves.toBe(2)
  })

  it('pads the element crop before capture', async () => {
    const capture = vi.fn(async (_rect: unknown, _signal?: AbortSignal) => 'data:image/png;base64,AA==')

    await captureAnnotateCrop(
      { capture, executeJavaScript: vi.fn(), withActivity: withoutScheduling },
      { height: 16, width: 40, x: 10, y: 20 }
    )

    expect(capture.mock.calls[0][0]).toEqual({
      height: 16 + ANNOTATE_CROP_PAD * 2,
      width: 40 + ANNOTATE_CROP_PAD * 2,
      x: 0,
      y: 20 - ANNOTATE_CROP_PAD
    })
  })

  it('brackets the shot so the crop is taken with only this comment marked', async () => {
    const order: string[] = []

    const executeJavaScript = vi.fn(async (code: string) => {
      order.push(code.includes('beginCapture') ? 'begin' : 'end')

      return true
    })

    const capture = vi.fn(async () => {
      order.push('capture')

      return 'data:image/png;base64,AA=='
    })

    await captureAnnotateCrop(
      { capture, executeJavaScript, withActivity: withoutScheduling },
      { height: 16, width: 40, x: 10, y: 20 }
    )

    expect(order).toEqual(['begin', 'capture', 'end'])
  })

  it('restores saved pins even when the capture fails', async () => {
    const executeJavaScript = vi.fn(async (code: string) => {
      void code

      return true
    })

    const capture = vi.fn(async () => {
      throw new Error('capture exploded')
    })

    await expect(
      captureAnnotateCrop(
        { capture, executeJavaScript, withActivity: withoutScheduling },
        { height: 16, width: 40, x: 10, y: 20 }
      )
    ).rejects.toThrow('capture exploded')

    expect(executeJavaScript.mock.calls.some(([code]) => String(code).includes('endCapture'))).toBe(true)
  })

  it('still captures when the overlay cannot be reached', async () => {
    const capture = vi.fn(async () => 'data:image/png;base64,AA==')

    const executeJavaScript = vi.fn(async (code: string) => {
      void code

      throw new Error('guest is gone')
    })

    await expect(
      captureAnnotateCrop(
        { capture, executeJavaScript, withActivity: withoutScheduling },
        { height: 16, width: 40, x: 10, y: 20 }
      )
    ).resolves.toContain('data:image/png')
  })

  it.each([false, true])(
    'keeps a hidden crop awake through both real paint waits and cleanup (failure=%s)',
    async fail => {
      const previousDesktop = window.hermesDesktop
      const guestWindow = window as Window & { __hermesAnnotate?: AnnotateInPage }
      const previousOverlay = guestWindow.__hermesAnnotate
      let awake = false

      const begin = vi.fn(async () => {
        awake = true

        return 71
      })

      const end = vi.fn(async () => {
        awake = false
      })

      window.hermesDesktop = {
        beginPreviewGuestActivity: begin,
        endPreviewGuestActivity: end
      } as unknown as Window['hermesDesktop']
      let callbacks: FrameRequestCallback[] = []

      const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
        callbacks.push(callback)

        return callbacks.length
      })

      const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
      const api = annotateInPage(document)
      guestWindow.__hermesAnnotate = api
      const captured = deferred<string>()
      const cleaned = deferred<void>()

      const webview = Object.assign(document.createElement('div'), {
        getWebContentsId: () => 42,
        executeJavaScript: vi.fn(async (code: string) => {
          const result = new Function('return (' + code + ')')()

          if (code.includes('endCapture')) {
            await cleaned.promise
          }

          return result
        })
      })

      const capture = vi.fn(() => {
        expect(awake).toBe(true)

        return captured.promise
      })

      const guest: PreviewAnnotateGuest = {
        capture,
        executeJavaScript: bindPreviewExecuteJavaScript(webview),
        withActivity: (operation, signal) =>
          withPreviewGuestActivity(webview, () => operation(bindPreviewExecuteJavaScript(webview)), signal)
      }

      const frame = () => {
        // Model Chromium withholding hidden frames unless a lease is active.
        if (document.visibilityState === 'hidden' && !awake) {
          return
        }

        const batch = callbacks
        callbacks = []

        for (const callback of batch) {
          callback(0)
        }
      }

      try {
        api.install()
        api.showPins([{ kind: 'area', number: 1, rect: { height: 20, width: 40, x: 10, y: 10 } }])

        const pins = document
          .querySelector('hermes-annotate')!
          .shadowRoot!.querySelector('[data-annotate-pin="1"]')!.parentElement!

        const waiting = waitAnnotateEvent(guest)
        expect(begin).not.toHaveBeenCalled()
        const crop = captureAnnotateCrop(guest, { height: 20, width: 40, x: 10, y: 10 })
        const outcome = fail ? expect(crop).rejects.toThrow('capture failed') : expect(crop).resolves.toBe('image')
        await vi.waitFor(() => expect(pins.style.display).toBe('none'))
        visibility.mockReturnValue('hidden')
        frame()
        expect(capture).not.toHaveBeenCalled()
        frame()
        await vi.waitFor(() => expect(capture).toHaveBeenCalledOnce())
        expect(end).not.toHaveBeenCalled()

        if (fail) {
          captured.reject(new Error('capture failed'))
        } else {
          captured.resolve('image')
        }

        await vi.waitFor(() => expect(pins.style.display).toBe('block'))
        expect(awake).toBe(true)
        expect(end).not.toHaveBeenCalled()
        cleaned.resolve()
        await outcome
        expect(begin).toHaveBeenCalledOnce()
        expect(end).toHaveBeenCalledOnce()
        expect(awake).toBe(false)
        // Waiting for another annotation is still pending, without a wake lease.
        api.teardown()
        await expect(waiting).resolves.toEqual({ type: 'end' })
        expect(begin).toHaveBeenCalledOnce()
      } finally {
        api.teardown()
        raf.mockRestore()
        visibility.mockRestore()
        window.hermesDesktop = previousDesktop
        guestWindow.__hermesAnnotate = previousOverlay
      }
    }
  )
})

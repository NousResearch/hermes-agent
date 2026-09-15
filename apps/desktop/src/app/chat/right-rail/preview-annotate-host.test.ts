import { describe, expect, it, vi } from 'vitest'

import { ANNOTATE_CROP_PAD } from '@/lib/preview-annotate'

import {
  bindPreviewExecuteJavaScript,
  captureAnnotateCrop,
  installAnnotateOverlay,
  overlayInstallScript
} from './preview-annotate-host'

describe('preview annotate host', () => {
  it('does not evaluate guest template interpolations while wrapping the overlay source', () => {
    const snippet = '${rect.height}'
    const source = '(function(){return `' + snippet + '`})(document)'
    const script = overlayInstallScript(source)

    expect(script).toContain(snippet)
    expect(() => overlayInstallScript(source)).not.toThrow()
  })

  it('injects the guest overlay source', async () => {
    const executeJavaScript = vi.fn(async (code: string) => {
      expect(code).toContain('hermes-annotate')
      expect(code).toContain('#2F80ED')
      expect(code).toContain('api.install()')
      expect(code).toContain('window.__hermesAnnotate')
    })

    await installAnnotateOverlay({ executeJavaScript })
    expect(executeJavaScript).toHaveBeenCalledOnce()
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
    const capture = vi.fn(async () => 'data:image/png;base64,AA==')

    await captureAnnotateCrop({ capture, executeJavaScript: vi.fn() }, { height: 16, width: 40, x: 10, y: 20 })

    expect(capture).toHaveBeenCalledWith({
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

    await captureAnnotateCrop({ capture, executeJavaScript }, { height: 16, width: 40, x: 10, y: 20 })

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
      captureAnnotateCrop({ capture, executeJavaScript }, { height: 16, width: 40, x: 10, y: 20 })
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
      captureAnnotateCrop({ capture, executeJavaScript }, { height: 16, width: 40, x: 10, y: 20 })
    ).resolves.toContain('data:image/png')
  })
})

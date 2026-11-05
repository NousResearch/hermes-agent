import { afterEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '@/test/deferred'

import { captureAnnotateCrop, createPreviewAnnotateGuest } from './preview-annotate-host'
import { runPreviewGuestScript, withPreviewDeadline } from './preview-drive-channels'

const rect = { height: 20, width: 40, x: 10, y: 10 }
const originalDesktop = window.hermesDesktop
const guestWindow = window as Window & {
  __hermesAnnotate?: { beginCapture: () => Promise<void>; endCapture: () => void }
}

afterEach(() => {
  window.hermesDesktop = originalDesktop
  delete guestWindow.__hermesAnnotate
  vi.useRealTimers()
})

function setup(stalled: 'begin' | 'capture' | 'cleanup' | 'read' | 'viewport') {
  const pending = deferred<never>()
  const leases = new Set<number>()
  let nextToken = 0
  let pinsVisible = true
  const capture = vi.fn(async () => (stalled === 'capture' ? pending.promise : 'image'))
  guestWindow.__hermesAnnotate = {
    beginCapture: async () => {
      pinsVisible = false

      if (stalled === 'begin') {
        await pending.promise
      }
    },
    endCapture: () => {
      pinsVisible = true
    }
  }
  const execute = async (code: string) => {
    if ((stalled === 'viewport' && code.includes('innerWidth')) || (stalled === 'read' && code.includes('innerText'))) {
      return pending.promise
    }

    const result: unknown = new Function('return (' + code + ')')()

    if (stalled === 'cleanup' && code.includes('endCapture')) {
      return pending.promise
    }

    return result
  }
  window.hermesDesktop = {
    beginPreviewGuestActivity: async () => {
      leases.add(++nextToken)

      return nextToken
    },
    endPreviewGuestActivity: async (token: number) => {
      leases.delete(token)
    },
    executePreviewGuestScript: async (_token: number, code: string) => execute(code),
    capturePreview: capture
  } as unknown as Window['hermesDesktop']
  const webview = Object.assign(document.createElement('div'), {
    executeJavaScript: execute,
    getWebContentsId: () => 42
  })

  return { capture, guest: createPreviewAnnotateGuest(webview)!, leases, pinsVisible: () => pinsVisible, webview }
}

describe('bounded first-party preview work', () => {
  it.each(['begin', 'viewport', 'capture', 'cleanup'] as const)(
    'releases every lease when %s never completes',
    async stage => {
      vi.useFakeTimers()
      const fixture = setup(stage)
      const result = captureAnnotateCrop(fixture.guest, rect)
      const rejected = expect(result).rejects.toThrow('Preview operation timed out')
      await vi.advanceTimersByTimeAsync(0)
      expect(fixture.leases.size).toBeGreaterThan(0)
      await vi.advanceTimersByTimeAsync(8_000)
      await rejected
      expect(fixture.leases.size).toBe(0)
      expect(fixture.pinsVisible()).toBe(true)

      if (stage === 'begin' || stage === 'viewport') {
        expect(fixture.capture).not.toHaveBeenCalled()
      }
    }
  )

  it('does not advance capture or dress a replacement overlay after cancellation', async () => {
    vi.useFakeTimers()
    const fixture = setup('begin')
    const lifetime = new AbortController()
    const result = captureAnnotateCrop(fixture.guest, rect, lifetime.signal)
    const rejected = expect(result).rejects.toThrow()
    await vi.advanceTimersByTimeAsync(0)
    const replacementCleanup = vi.fn()
    guestWindow.__hermesAnnotate = { beginCapture: async () => undefined, endCapture: replacementCleanup }
    lifetime.abort()
    await rejected
    expect(fixture.leases.size).toBe(0)
    expect(fixture.capture).not.toHaveBeenCalled()
    expect(replacementCleanup).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('bounds page reads and releases their lease without a polling timer', async () => {
    vi.useFakeTimers()
    const fixture = setup('read')
    const result = withPreviewDeadline(signal =>
      runPreviewGuestScript(fixture.webview, 'document.body.innerText', signal)
    )
    const rejected = expect(result).rejects.toThrow('Preview operation timed out')
    await vi.advanceTimersByTimeAsync(0)
    expect(fixture.leases.size).toBe(1)
    await vi.advanceTimersByTimeAsync(8_000)
    await rejected
    expect(fixture.leases.size).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
  })
})

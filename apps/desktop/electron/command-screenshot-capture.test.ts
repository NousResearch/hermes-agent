import { describe, expect, it, vi } from 'vitest'

import { createScreenshotCapture } from './command-screenshot-capture'

const window = { windowId: 42, width: 1200, height: 800 }
const png = new Uint8Array([137, 80, 78, 71])

function setup() {
  const image = { isEmpty: () => false, toPNG: () => png }
  const sources = vi.fn(async () => [{ id: 'window:42:0', thumbnail: image }])
  const permission = vi.fn(() => true)
  const capture = createScreenshotCapture({ getSources: sources, hasScreenPermission: permission })

  return { capture, sources, permission }
}

describe('Command screenshot capture', () => {
  it('binds a single-use gesture to its original renderer and exact window, never another screen', async () => {
    const { capture, sources } = setup()
    const request = capture.request(7, window)!

    expect(await capture.take(8, request)).toEqual({ ok: false, reason: 'expired' })
    expect(sources).not.toHaveBeenCalled()
    expect(await capture.take(7, request)).toEqual({ ok: true, png })
    expect(sources).toHaveBeenCalledWith({
      types: ['window'],
      thumbnailSize: { width: 2400, height: 1600 },
      fetchWindowIcons: false
    })
    expect(await capture.take(7, request)).toEqual({ ok: false, reason: 'expired' })

    sources.mockResolvedValue([{ id: 'window:99:0', thumbnail: { isEmpty: () => false, toPNG: () => png } }])
    expect(await capture.take(7, capture.request(7, window)!)).toEqual({ ok: false, reason: 'unavailable' })
  })

  it('drops denied, expired, concurrent and disabled captures without leaking a late image', async () => {
    const { capture, sources, permission } = setup()
    permission.mockReturnValue(false)
    expect(await capture.take(7, capture.request(7, window)!)).toEqual({ ok: false, reason: 'screen-permission' })
    expect(sources).not.toHaveBeenCalled()
    permission.mockReturnValue(true)

    vi.useFakeTimers()

    try {
      const expired = capture.request(7, window)!
      vi.advanceTimersByTime(6000)
      expect(await capture.take(7, expired)).toEqual({ ok: false, reason: 'expired' })
    } finally {
      vi.useRealTimers()
    }

    let finish!: (value: Awaited<ReturnType<typeof sources>>) => void
    sources.mockImplementation(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    const pending = capture.take(7, capture.request(7, window)!)
    expect(capture.request(7, window)).toBeNull()
    capture.clear()
    finish([{ id: 'window:42:0', thumbnail: { isEmpty: () => false, toPNG: () => png } }])
    expect(await pending).toEqual({ ok: false, reason: 'expired' })
  })
})

import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('zoom store initialization', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('does not let a stale initial read overwrite a newer change event', async () => {
    type Payload = { level: number; percent: number }

    let emitChanged: ((payload: Payload) => void) | undefined
    let resolveGet: ((payload: Payload) => void) | undefined
    const get = vi.fn(
      () => new Promise<Payload>(resolve => {
        resolveGet = resolve
      })
    )

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      zoom: {
        get,
        setPercent: vi.fn(),
        onChanged: vi.fn((callback: (payload: Payload) => void) => {
          emitChanged = callback
          return vi.fn()
        })
      }
    }

    const { $zoomPercent } = await import('./zoom')
    emitChanged?.({ level: 3, percent: 125 })
    resolveGet?.({ level: 0, percent: 100 })
    await Promise.resolve()

    expect($zoomPercent.get()).toBe(125)
  })
})

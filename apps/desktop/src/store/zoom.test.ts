import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

type ZoomBridge = NonNullable<Window['hermesDesktop']['zoom']>
type ZoomPayload = Awaited<ReturnType<ZoomBridge['get']>>

function installZoomBridge(zoom: ZoomBridge): void {
  desktopWindow.hermesDesktop = { zoom } as unknown as Window['hermesDesktop']
}

beforeEach(() => {
  vi.resetModules()
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('zoom state sync', () => {
  it('uses the initial read when no newer change has arrived', async () => {
    const zoom: ZoomBridge = {
      get: vi.fn().mockResolvedValue({ level: 1, percent: 125 }),
      onChanged: vi.fn(() => vi.fn()),
      setPercent: vi.fn()
    }

    installZoomBridge(zoom)

    const { $zoomPercent } = await import('./zoom')

    expect($zoomPercent.get()).toBe(125)
  })

  it('does not let a stale initial read overwrite a newer zoom event', async () => {
    const callOrder: string[] = []
    let emitChanged: ((payload: ZoomPayload) => void) | undefined
    let resolveInitialRead: ((payload: ZoomPayload) => void) | undefined

    const initialRead = new Promise<ZoomPayload>(resolve => {
      resolveInitialRead = resolve
    })

    const zoom: ZoomBridge = {
      get: vi.fn(() => {
        callOrder.push('get')

        return initialRead
      }),
      onChanged: vi.fn(callback => {
        callOrder.push('subscribe')
        emitChanged = callback

        return vi.fn()
      }),
      setPercent: vi.fn()
    }

    installZoomBridge(zoom)

    const { $zoomPercent } = await import('./zoom')

    expect(callOrder).toEqual(['subscribe', 'get'])
    emitChanged?.({ level: 1, percent: 125 })
    resolveInitialRead?.({ level: 0, percent: 100 })
    await initialRead
    await Promise.resolve()

    expect($zoomPercent.get()).toBe(125)
  })
})

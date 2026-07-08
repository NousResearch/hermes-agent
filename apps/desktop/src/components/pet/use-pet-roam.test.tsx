import { act, type ReactNode, type RefObject, useRef } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/pet', () => ({
  $petMotion: { set: () => undefined },
  $petRoamDir: { set: () => undefined }
}))

import { usePetRoam } from './use-pet-roam'

let root: Root | null = null
let container: HTMLDivElement | null = null
let windowStateCallback: ((payload: { isMinimized?: boolean; isVisible?: boolean }) => void) | null = null

function render(ui: ReactNode) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)

  act(() => {
    root!.render(ui)
  })
}

function cleanup() {
  if (root) {
    act(() => {
      root!.unmount()
    })
  }

  container?.remove()
  root = null
  container = null
}

function setVisibility(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: hidden ? 'hidden' : 'visible' })
}

function installWindowStateBridge() {
  windowStateCallback = null
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      onWindowStateChanged: vi.fn((callback: typeof windowStateCallback) => {
        windowStateCallback = callback

        return () => {
          if (windowStateCallback === callback) {
            windowStateCallback = null
          }
        }
      })
    }
  })
}

function installRaf() {
  const request = vi.fn((_callback: FrameRequestCallback) => 1)
  const cancel = vi.fn()

  Object.defineProperty(window, 'requestAnimationFrame', { configurable: true, value: request })
  Object.defineProperty(window, 'cancelAnimationFrame', { configurable: true, value: cancel })

  return { cancel, request }
}

function RoamHarness({ isInteracting = () => false }: { isInteracting?: () => boolean }) {
  const ref = useRef<HTMLDivElement | null>(null)

  usePetRoam({
    commit: () => undefined,
    containerRef: ref as RefObject<HTMLDivElement | null>,
    enabled: true,
    isInteracting,
    loopMs: 1200,
    overlayOpen: false,
    petH: 64,
    petW: 64
  })

  return <div ref={ref} />
}

describe('usePetRoam RAF scheduling', () => {
  beforeEach(() => {
    ;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
    vi.useFakeTimers()
    setVisibility(false)
    installWindowStateBridge()
    vi.spyOn(Math, 'random').mockReturnValue(0)
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      bottom: 164,
      height: 64,
      left: 100,
      right: 164,
      top: 100,
      width: 64,
      x: 100,
      y: 100,
      toJSON: () => ({})
    } as DOMRect)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    setVisibility(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('uses a pause timer, not RAF, while dwelling at idle', () => {
    const raf = installRaf()

    render(<RoamHarness />)

    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(1)
  })

  it('clears the pause wakeup while the Electron window is paused and restarts it when visible', () => {
    const raf = installRaf()

    render(<RoamHarness />)
    expect(vi.getTimerCount()).toBe(1)

    windowStateCallback?.({ isMinimized: true, isVisible: false })

    expect(raf.cancel).not.toHaveBeenCalled()
    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)

    windowStateCallback?.({ isMinimized: false, isVisible: true })

    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(1)
  })
})

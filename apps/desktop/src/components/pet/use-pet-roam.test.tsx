import { act, type RefObject, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/pet', () => ({
  $petMotion: { set: () => undefined },
  $petRoamDir: { set: () => undefined }
}))

import { reactRoot } from '@/test/react-root'

import { installWindowStateBridge, setDocumentHidden, type WindowStateBridge } from '../../test/window-state'

import { usePetRoam } from './use-pet-roam'

const mount = reactRoot()
let windowState: WindowStateBridge

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
    setDocumentHidden(false)
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    windowState = installWindowStateBridge()
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
    mount.unmount()
    vi.useRealTimers()
    vi.restoreAllMocks()
    setDocumentHidden(false)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('uses a pause timer, not RAF, while dwelling at idle', () => {
    const raf = installRaf()

    mount.render(<RoamHarness />)

    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(1)
  })

  it('clears the pause wakeup while the Electron window is paused and restarts it when visible', () => {
    const raf = installRaf()

    mount.render(<RoamHarness />)
    expect(vi.getTimerCount()).toBe(1)

    windowState.emit({ isMinimized: true, isVisible: false })

    expect(raf.cancel).not.toHaveBeenCalled()
    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)

    windowState.emit({ isMinimized: false, isVisible: true })

    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(1)
  })

  it('keeps roaming while unfocused and cleans up its wake timer on unmount', () => {
    const raf = installRaf()

    mount.render(<RoamHarness />)
    expect(vi.getTimerCount()).toBe(1)

    // Blur alone leaves the window visible, so the roam loop keeps running.
    // Minimize/hide still parks it -- covered by the windowState test above.
    act(() => window.dispatchEvent(new Event('blur')))
    expect(vi.getTimerCount()).toBe(1)

    act(() => window.dispatchEvent(new Event('focus')))
    expect(vi.getTimerCount()).toBe(1)

    mount.unmount()
    expect(vi.getTimerCount()).toBe(0)

    act(() => {
      vi.advanceTimersByTime(2000)
      window.dispatchEvent(new Event('focus'))
    })
    expect(raf.request).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })
})

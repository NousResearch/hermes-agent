import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection } from '@/store/session'
import { stubResizeObserver } from '@/test/jsdom'

import { usePanelTitlebar } from './panel-titlebar'

// The panel tab strip (SESSIONS | BOTS) reserves the width the fixed titlebar
// clusters cover by MEASURING them. On a macOS fullscreen transition the left
// cluster moves to the window edge without changing size, so neither the
// ResizeObserver nor the window `resize` (which fires before the state IPC
// lands) re-measures — the reservation stayed stale until a sidebar toggle or
// reload (#108641). Chrome-state changes must schedule a re-measure.

let clusterLeft = 100

function rect(left: number, width: number): DOMRect {
  return { bottom: 34, height: 34, left, right: left + width, top: 0, width, x: left, y: 0 } as DOMRect
}

beforeEach(() => {
  // Re-stub per test: `vi.unstubAllGlobals()` in afterEach drops it too.
  stubResizeObserver()
  clusterLeft = 100
  const left = document.createElement('div')
  left.dataset.titlebarCluster = 'left'
  left.getBoundingClientRect = () => rect(clusterLeft, 60)
  const right = document.createElement('div')
  right.dataset.titlebarCluster = 'right'
  right.getBoundingClientRect = () => rect(900, 80)
  document.body.append(left, right)
  $connection.set({ isFullscreen: false, windowButtonPosition: { x: 20, y: 10 } } as unknown as HermesConnection)
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    cb(0)

    return 0
  })
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
})

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
  vi.unstubAllGlobals()
  $connection.set(null)
})

describe('usePanelTitlebar', () => {
  it('re-measures the cluster reservation when the window chrome state moves the clusters', () => {
    const element = document.createElement('div')
    element.getBoundingClientRect = () => rect(0, 1000)
    document.body.append(element)
    const ref = { current: element }

    renderHook(() => usePanelTitlebar(ref, true, false))
    expect(element.style.getPropertyValue('--panel-titlebar-left')).toBe('172px')

    // Fullscreen: the traffic lights vanish and the cluster pins to the edge —
    // a pure translate. No ResizeObserver callback, no `resize` event.
    clusterLeft = 14
    act(() => {
      $connection.set({ ...$connection.get()!, isFullscreen: true })
    })

    expect(element.style.getPropertyValue('--panel-titlebar-left')).toBe('86px')
  })

  it('ignores connection updates that leave the chrome where it is', () => {
    const element = document.createElement('div')
    element.getBoundingClientRect = () => rect(0, 1000)
    document.body.append(element)
    const ref = { current: element }

    renderHook(() => usePanelTitlebar(ref, true, false))
    // A stale-geometry read must not be re-committed by an unrelated update
    // (e.g. a status heartbeat): only chrome-moving fields trigger a measure.
    clusterLeft = 14
    act(() => {
      $connection.set({ ...$connection.get()!, connected: true } as HermesConnection)
    })

    expect(element.style.getPropertyValue('--panel-titlebar-left')).toBe('172px')
  })
})

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { usePanelTitlebar } from './panel-titlebar'

const chromeRevisionEvent = 'hermes:titlebar-chrome-revision'

function rect(left: number, width: number): DOMRect {
  return {
    bottom: 40,
    height: 40,
    left,
    right: left + width,
    top: 0,
    width,
    x: left,
    y: 0,
    toJSON: () => ({})
  }
}

afterEach(() => {
  cleanup()
  globalThis.document.body.replaceChildren()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('remeasures translated titlebar clusters when chrome geometry is revised', () => {
  let leftClusterX = 100
  const panel = globalThis.document.createElement('div')
  const leftCluster = globalThis.document.createElement('div')
  const rightCluster = globalThis.document.createElement('div')
  leftCluster.dataset.titlebarCluster = 'left'
  rightCluster.dataset.titlebarCluster = 'right'
  globalThis.document.body.append(leftCluster, rightCluster, panel)

  vi.spyOn(panel, 'getBoundingClientRect').mockImplementation(() => rect(0, 500))
  vi.spyOn(leftCluster, 'getBoundingClientRect').mockImplementation(() => rect(leftClusterX, 60))
  vi.spyOn(rightCluster, 'getBoundingClientRect').mockImplementation(() => rect(450, 30))
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )

  renderHook(() => usePanelTitlebar({ current: panel }, true, false))
  expect(panel.style.getPropertyValue('--panel-titlebar-left')).toBe('172px')

  // Fullscreen repositions fixed chrome without changing its dimensions, so
  // ResizeObserver has no entry to deliver for this translation.
  leftClusterX = 14
  act(() => window.dispatchEvent(new CustomEvent(chromeRevisionEvent, { detail: 1 })))

  expect(panel.style.getPropertyValue('--panel-titlebar-left')).toBe('86px')
})

it('retains the last safe reservation when a chrome cluster is missing', () => {
  const panel = globalThis.document.createElement('div')
  const leftCluster = globalThis.document.createElement('div')
  const rightCluster = globalThis.document.createElement('div')
  leftCluster.dataset.titlebarCluster = 'left'
  rightCluster.dataset.titlebarCluster = 'right'
  globalThis.document.body.append(leftCluster, rightCluster, panel)

  vi.spyOn(panel, 'getBoundingClientRect').mockImplementation(() => rect(0, 500))
  vi.spyOn(leftCluster, 'getBoundingClientRect').mockImplementation(() => rect(100, 60))
  vi.spyOn(rightCluster, 'getBoundingClientRect').mockImplementation(() => rect(450, 30))
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )

  renderHook(() => usePanelTitlebar({ current: panel }, true, false))
  expect(panel.style.getPropertyValue('--panel-titlebar-left')).toBe('172px')

  rightCluster.remove()
  act(() => window.dispatchEvent(new CustomEvent(chromeRevisionEvent, { detail: 1 })))

  expect(panel.style.getPropertyValue('--panel-titlebar-left')).toBe('172px')
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { stubResizeObserver } from '@/test/jsdom'

import { publishWorkspaceGeometry, useWindowControlsOverlap } from './geometry'
import { usePanelTitlebar } from './tree/renderer/panel-titlebar'

let panelRect: DOMRect
let stopGeometry: (() => void) | undefined

beforeEach(() => {
  stubResizeObserver()
  panelRect = new DOMRect(280, 0, 400, 600)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.titlebarCluster === 'left') {
      return new DOMRect(0, 0, 180, 34)
    }

    if (this.dataset.titlebarCluster === 'right') {
      return new DOMRect(window.innerWidth - 74, 0, 74, 34)
    }

    return panelRect
  })
})

afterEach(() => {
  cleanup()
  stopGeometry?.()
  stopGeometry = undefined
  $connection.set(null)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function Panel({ titlebar = false }: { titlebar?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const belowControls = usePanelTitlebar(ref, titlebar, false)
  const overlap = useWindowControlsOverlap(ref, !titlebar)

  return (
    <div aria-label="Panels" role="region">
      <div aria-label="Nested split" role="region">
        <div
          aria-label="Session"
          data-below-controls={belowControls}
          data-session-anchor="workspace"
          ref={ref}
          role="region"
          style={{ paddingTop: overlap ? overlap.y + overlap.height : 0 }}
        >
          <div aria-label="Transcript" role="log" />
        </div>
      </div>
    </div>
  )
}

it('keeps panel tabs clear of fixed titlebar controls when ancestor splits scroll', () => {
  render(
    <>
      <div data-titlebar-cluster="left" />
      <div data-titlebar-cluster="right" />
      <Panel titlebar />
    </>
  )

  const panel = screen.getByRole('region', { name: 'Session' })
  expect(panel.style.getPropertyValue('--panel-titlebar-left')).toBe('0px')
  expect(panel.dataset.belowControls).toBe('false')

  panelRect = new DOMRect(-100, 0, panelRect.width, panelRect.height)
  fireEvent.scroll(screen.getByRole('region', { name: 'Panels' }))

  const reserved = Number.parseFloat(panel.style.getPropertyValue('--panel-titlebar-left'))
  expect(panelRect.left + reserved).toBeGreaterThan(180)
  expect(panel.dataset.belowControls).toBe('true')

  panelRect = new DOMRect(280, 0, panelRect.width, panelRect.height)
  fireEvent.scroll(screen.getByRole('region', { name: 'Nested split' }))
  expect(panel.style.getPropertyValue('--panel-titlebar-left')).toBe('0px')
  expect(panel.dataset.belowControls).toBe('false')

  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockClear()
  fireEvent.scroll(screen.getByRole('log', { name: 'Transcript' }))
  expect(HTMLElement.prototype.getBoundingClientRect).not.toHaveBeenCalled()
})

it('updates workspace alignment and native control clearance without a resize', () => {
  vi.stubGlobal('hermesDesktop', {})
  $connection.set({
    baseUrl: 'http://localhost',
    isFullscreen: false,
    logs: [],
    mode: 'local',
    nativeOverlayWidth: 74,
    token: '',
    windowButtonPosition: null,
    wsUrl: 'ws://localhost'
  })
  render(<Panel />)
  stopGeometry = publishWorkspaceGeometry()

  const panel = screen.getByRole('region', { name: 'Session' })
  const rootStyle = globalThis.document.documentElement.style
  expect(panel.style.paddingTop).toBe('0px')

  panelRect = new DOMRect(window.innerWidth - 100, 0, panelRect.width, panelRect.height)
  fireEvent.scroll(screen.getByRole('region', { name: 'Panels' }))

  expect(panel.style.paddingTop).toBe('34px')
  expect(rootStyle.getPropertyValue('--workspace-left')).toBe(`${panelRect.left}px`)
  expect(rootStyle.getPropertyValue('--workspace-right')).toBe(`${window.innerWidth - panelRect.right}px`)

  panelRect = new DOMRect(280, 0, panelRect.width, panelRect.height)
  fireEvent.scroll(screen.getByRole('region', { name: 'Nested split' }))

  expect(panel.style.paddingTop).toBe('0px')
  expect(rootStyle.getPropertyValue('--workspace-left')).toBe(`${panelRect.left}px`)
  expect(rootStyle.getPropertyValue('--workspace-right')).toBe(`${window.innerWidth - panelRect.right}px`)

  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockClear()
  fireEvent.scroll(screen.getByRole('log', { name: 'Transcript' }))
  expect(HTMLElement.prototype.getBoundingClientRect).not.toHaveBeenCalled()
})

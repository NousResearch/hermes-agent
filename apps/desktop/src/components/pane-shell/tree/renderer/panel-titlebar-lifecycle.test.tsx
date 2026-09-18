import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { StrictMode, useRef } from 'react'
import { Link, MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ROUTES_AREA } from '@/app/routes'
import { TITLEBAR_CHROME_CHANGED_EVENT } from '@/app/shell/titlebar'
import { TitlebarControls } from '@/app/shell/titlebar-controls'
import { Contribute } from '@/contrib/react/contribute'
import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'

import { usePanelTitlebar } from './panel-titlebar'

const observers = new Set<ControlledResizeObserver>()

class ControlledResizeObserver {
  readonly targets = new Set<Element>()

  constructor(readonly callback: ResizeObserverCallback) {
    observers.add(this)
  }

  observe(target: Element) {
    this.targets.add(target)
  }

  unobserve(target: Element) {
    this.targets.delete(target)
  }

  disconnect() {
    this.targets.clear()
  }
}

function deliver(target: Element) {
  for (const observer of observers) {
    if (observer.targets.has(target)) {
      observer.callback([{ target } as ResizeObserverEntry], observer as unknown as ResizeObserver)
    }
  }
}

function rect(left: number, width: number): DOMRect {
  return { bottom: 34, height: 34, left, right: left + width, top: 0, width, x: left, y: 0 } as DOMRect
}

let panelWidth = 800
let chipWidth = 90
let rightEdge = 1200
let disposeRoutes: () => void

function Panel({ enabled = true, minimized = false }: { enabled?: boolean; minimized?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const below = usePanelTitlebar(ref, enabled, minimized)

  return <div data-below={below} data-testid="panel" data-tree-group="sessions" ref={ref} />
}

function Clusters({ branch }: { branch: string }) {
  return (
    <>
      <div data-titlebar-cluster="left" key={`left-${branch}`} />
      <div data-titlebar-cluster="right" key={`right-${branch}`} />
    </>
  )
}

function RouteChrome() {
  const location = useLocation()

  return (
    <>
      <TitlebarControls onOpenSettings={() => {}} />
      <Panel />
      <Link to="/">Chat</Link>
      <Link to="/kanban">Kanban</Link>
      <Link to="/settings">Settings</Link>
      {location.pathname === '/kanban' && (
        <Contribute area="titleBar.left" id="test:page-tools">
          <button type="button">Openplanet</button>
        </Contribute>
      )}
    </>
  )
}

beforeEach(() => {
  panelWidth = 800
  chipWidth = 90
  rightEdge = 1200
  observers.clear()
  vi.stubGlobal('ResizeObserver', ControlledResizeObserver)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.testid === 'panel') {
      return rect(0, panelWidth)
    }

    if (this.dataset.titlebarCluster === 'left') {
      return rect(98, chipWidth)
    }

    if (this.dataset.titlebarCluster === 'right') {
      return rect(rightEdge, 100)
    }

    return rect(0, 0)
  })
  disposeRoutes = registry.register({
    area: ROUTES_AREA,
    data: { path: '/kanban' },
    id: 'test:kanban-route',
    render: () => null
  })
})

afterEach(() => {
  cleanup()
  disposeRoutes()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const reservation = (side: 'left' | 'right') =>
  screen.getByTestId('panel').style.getPropertyValue(`--panel-titlebar-${side}`)

const chromeChanged = () => window.dispatchEvent(new Event(TITLEBAR_CHROME_CHANGED_EVENT))

describe('titlebar reservation lifecycle', () => {
  it('rebinds to replacement nodes and tracks a longer board name under StrictMode', () => {
    const fixture = (branch: string) => (
      <StrictMode>
        <Panel />
        <Clusters branch={branch} />
      </StrictMode>
    )

    const { rerender } = render(fixture('chat'))
    const previous = globalThis.document.querySelector('[data-titlebar-cluster="left"]')!
    expect(reservation('left')).toBe('200px')

    chipWidth = 200
    rerender(fixture('kanban'))
    const current = globalThis.document.querySelector('[data-titlebar-cluster="left"]')!
    expect(current).not.toBe(previous)
    expect(previous.isConnected).toBe(false)
    act(chromeChanged)
    expect(reservation('left')).toBe('310px')
    expect([...observers].some(observer => observer.targets.has(previous))).toBe(false)

    chipWidth = 260
    act(() => deliver(current))
    expect(reservation('left')).toBe('370px')
  })

  it('drops narrow panel tabs below the picker and restores them when room returns', () => {
    render(
      <>
        <Panel />
        <Clusters branch="kanban" />
      </>
    )
    const panel = screen.getByTestId('panel')
    expect(panel.dataset.below).toBe('false')

    panelWidth = 280
    act(() => deliver(panel))
    expect(panel.dataset.below).toBe('true')

    panelWidth = 800
    act(() => deliver(panel))
    expect(panel.dataset.below).toBe('false')
  })

  it('refreshes right-caption clearance after chrome translates without resizing', () => {
    render(
      <>
        <Panel />
        <Clusters branch="kanban" />
      </>
    )
    expect(reservation('right')).toBe('0px')

    rightEdge = 700
    act(chromeChanged)
    expect(reservation('right')).toBe('112px')
  })

  it('remeasures through real chat, contributed-page and overlay navigation', () => {
    render(
      <StrictMode>
        <MemoryRouter>
          <I18nProvider configClient={null} initialLocale="en">
            <RouteChrome />
          </I18nProvider>
        </MemoryRouter>
      </StrictMode>
    )
    expect(reservation('left')).toBe('200px')

    chipWidth = 200
    fireEvent.click(screen.getByText('Kanban'))
    expect(screen.getByText('Openplanet').closest('[data-titlebar-cluster="left"]')).not.toBeNull()
    expect(globalThis.document.querySelector('[data-titlebar-cluster="right"]')).not.toBeNull()
    expect(reservation('left')).toBe('310px')

    fireEvent.click(screen.getByText('Settings'))
    expect(globalThis.document.querySelector('[data-titlebar-cluster]')).toBeNull()
    expect(reservation('left')).toBe('310px')

    chipWidth = 90
    fireEvent.click(screen.getByText('Chat'))
    expect(reservation('left')).toBe('200px')
    expect(screen.queryByText('Openplanet')).toBeNull()
  })
})

import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import type { GroupNode } from '../model'
import { $treeDragging, NEW_SESSION_DRAG, SESSION_TILE_DRAG } from '../store'

import { TreeGroup } from './tree-group'

let root: null | Root = null
let container: HTMLDivElement | null = null
let disposePane: (() => void) | null = null

function render(ui: ReactNode) {
  if (!container) {
    container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    root = createRoot(container)
  }

  act(() => {
    root!.render(ui)
  })
}

function terminalGroup(minimized: boolean): GroupNode {
  return {
    active: 'terminal',
    id: 'terminal-zone',
    minimized,
    panes: ['terminal'],
    // The chevron lives in the strip, so this zone has to be showing one. A
    // lone unregistered pane is on auto and would render none.
    tabStrip: 'always',
    type: 'group'
  }
}

const toggle = (label: string) =>
  globalThis.document.querySelector<HTMLButtonElement>(
    `[data-tree-group="terminal-zone"] button[aria-label="${label}"]`
  )!

afterEach(() => {
  if (root) {
    act(() => root!.unmount())
  }

  container?.remove()
  disposePane?.()
  root = null
  container = null
  disposePane = null
  vi.unstubAllGlobals()
})

describe('TreeGroup', () => {
  it('keeps a top-edge strip inside its panel and yields native drag while moving a pane', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'terminal',
      title: 'Terminal',
      render: () => <div>Terminal</div>
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    render(<TreeGroup leftEdge node={terminalGroup(false)} rightEdge topEdge />)
    const zone = container!.querySelector('[data-tree-group]')!
    const strip = zone.querySelector<HTMLElement>('[data-zone-tabstrip]')!
    const header = zone.querySelector<HTMLElement>('[data-panel-header]')!
    expect(strip.closest('[data-tree-group]')).toBe(zone)
    expect(header.contains(strip)).toBe(true)
    expect(zone.querySelectorAll('[data-window-drag-handle]').length).toBeGreaterThan(0)
    act(() => $treeDragging.set('terminal'))
    expect(strip.style).toHaveProperty('WebkitAppRegion', 'no-drag')
    act(() => $treeDragging.set(null))
    expect(strip.style).toHaveProperty('WebkitAppRegion', '')
  })

  it('keeps panel tabs in the titlebar for a top-edge zone that is not the left edge', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'terminal',
      title: 'Terminal',
      render: () => <div>Terminal</div>
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    render(<TreeGroup node={terminalGroup(false)} topEdge />)
    const header = container!.querySelector('[data-panel-header]')!
    const strip = header.querySelector<HTMLElement>('[data-zone-tabstrip]')!
    expect(header.contains(strip)).toBe(true)
    expect(strip.className).toContain('h-full')
    expect(strip.className).toContain('[-webkit-app-region:drag]')
  })

  it('does not put a scroll-clipped tab strip in the left titlebar band beside Minimize', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'sessions',
      title: 'SESSIONS',
      render: () => <div>Sessions</div>
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    render(
      <TreeGroup
        leftEdge
        node={{
          active: 'sessions',
          id: 'sessions-zone',
          minimized: false,
          panes: ['sessions'],
          tabStrip: 'always',
          type: 'group'
        }}
        topEdge
      />
    )

    const header = container!.querySelector('[data-panel-header]')!
    const tab = container!.querySelector('[data-tree-tab="sessions"]')!
    const minimize = container!.querySelector('button[aria-label="Minimize"]')
    const strip = header.querySelector<HTMLElement>('[data-zone-tabstrip]')
    const tablist = tab.closest('[role="tablist"]')

    const clusterSpacer = [...header.querySelectorAll('div')].some(el =>
      el.className.includes('--titlebar-controls-width')
    )

    const titlebarStrip = Boolean(strip?.className.includes('h-full'))
    const tablistScrolls = Boolean(tablist?.className.includes('overflow-x-auto'))
    const truncates = Boolean(tab.querySelector('.truncate'))

    expect(tab.textContent).toContain('SESSIONS')
    // Left-edge titlebar must not host a truncated overflow-x-auto tablist in
    // the same flex row as the in-flow --titlebar-controls-width spacer.
    expect(clusterSpacer && titlebarStrip && tablistScrolls && truncates).toBe(false)

    if (minimize && strip?.contains(minimize)) {
      expect(titlebarStrip && clusterSpacer).toBe(false)
    }
  })

  it('cuts the titlebar drag filler so it cannot cover the window-control cluster', () => {
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    render(
      <TreeGroup
        node={{
          active: 'gone',
          id: 'empty-zone',
          minimized: false,
          panes: ['gone'],
          type: 'group'
        }}
        topEdge
      />
    )

    const header = container!.querySelector('[data-panel-header]')!

    const filler =
      header.querySelector('[data-titlebar-drag-fill]') ??
      [...header.querySelectorAll('div')].find(
        el =>
          el.className.includes('flex-1') &&
          el.className.includes('-webkit-app-region:drag') &&
          el.childElementCount === 0
      )

    expect(filler).toBeTruthy()
    expect(filler!.getAttribute('data-titlebar-drag-fill')).not.toBeNull()
    const css = `${filler!.className} ${filler!.getAttribute('style') ?? ''}`
    expect(css).toMatch(/--titlebar-controls-left/)
    expect(css).toMatch(/--titlebar-controls-width/)
  })

  it('points the docked-zone chevron in the collapse or restore action direction', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { height: '12rem' },
      id: 'terminal',
      render: () => <div>Terminal</div>,
      title: 'Terminal'
    })
    // jsdom does not implement CSS.escape, which the real tab-strip effect uses.
    vi.stubGlobal('CSS', { escape: (value: string) => value })

    render(<TreeGroup node={terminalGroup(false)} parentAxis="column" />)

    expect(toggle('Minimize').querySelector('i')!.className).toContain('codicon-chevron-down')

    render(<TreeGroup node={terminalGroup(true)} parentAxis="column" />)

    expect(toggle('Restore').querySelector('i')!.className).toContain('codicon-chevron-up')
  })

  // The invariant behind the shared eligibility predicate
  // (hostsSessionDropTarget): a session or new-session drag must paint the
  // SAME zones either drag resolver would accept, and stay dark everywhere
  // else. Standing chrome (terminal) is always dark; a zone hosting a chat
  // strip (workspace) paints for both sentinels; with no session-drag active
  // nothing paints even over an eligible zone.
  describe('session-drop overlay eligibility (one truth with the resolvers)', () => {
    const groupFor = (panes: string[], id = 'zone-a'): GroupNode => ({
      active: panes[0]!,
      id,
      minimized: false,
      panes,
      type: 'group'
    })

    const sheet = () => globalThis.document.querySelector('[data-tree-group="zone-a"] .pointer-events-none.absolute')

    async function withDragging(dragging: null | string, run: () => void) {
      await act(async () => {
        $treeDragging.set(dragging)
      })

      try {
        run()
      } finally {
        await act(async () => {
          $treeDragging.set(null)
        })
      }
    }

    it('stays dark over standing chrome (terminal) during a new-session drag', async () => {
      disposePane = registry.register({
        area: 'panes',
        data: { height: '12rem' },
        id: 'terminal',
        render: () => <div>Terminal</div>,
        title: 'Terminal'
      })
      vi.stubGlobal('CSS', { escape: (value: string) => value })

      render(<TreeGroup node={terminalGroup(false)} parentAxis="column" />)

      await withDragging(NEW_SESSION_DRAG, () => {
        expect(sheet()).toBeNull()
      })
    })

    it('lights a chat-strip zone for BOTH session and new-session drags, and only then', async () => {
      disposePane = registry.register({
        area: 'panes',
        data: {},
        id: 'workspace',
        render: () => <div>Chat</div>,
        title: 'Hermes'
      })
      vi.stubGlobal('CSS', { escape: (value: string) => value })

      render(<TreeGroup node={groupFor(['workspace'])} parentAxis="column" />)

      await withDragging(null, () => {
        expect(sheet()).toBeNull()
      })

      await withDragging(SESSION_TILE_DRAG, () => {
        expect(sheet()).not.toBeNull()
      })

      await withDragging(NEW_SESSION_DRAG, () => {
        expect(sheet()).not.toBeNull()
      })
    })
  })
})

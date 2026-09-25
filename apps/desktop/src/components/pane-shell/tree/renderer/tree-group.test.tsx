import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates } from '@/store/panes'
import { $tabStripDefault, setTabStripDefault } from '@/store/tabstrip-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { $layoutEditMode } from '../../edit-mode'
import type { GroupNode } from '../model'
import { $layoutTree, $treeDragging, NEW_SESSION_DRAG, SESSION_TILE_DRAG } from '../store'

import { TreeGroup } from './tree-group'

let root: null | Root = null
let container: HTMLDivElement | null = null
let disposePane: (() => void) | null = null
const disposers: (() => void)[] = []

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

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
  disposers.splice(0).forEach(d => d())
  $paneStates.set({})
  $layoutTree.set(null)
  $layoutEditMode.set(false)
  root = null
  container = null
  disposePane = null
  globalThis.document.querySelectorAll('[data-titlebar-cluster]').forEach(element => element.remove())
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('TreeGroup', () => {
  // Titlebar geometry (#112964): tabs sharing the native titlebar band are all
  // `no-drag`, so once the list overflows they can cover every draggable pixel.
  // A window must keep a drag target that no tab can occupy — a fixed-width
  // handle OUTSIDE the scrolling tablist. When the tabs instead drop below the
  // window controls, the free band above them stays the (flexible) handle.
  describe('top-edge window drag handle', () => {
    const paneIds = ['terminal', 'terminal-2', 'terminal-3', 'terminal-4', 'terminal-5']

    function mountCrowdedStrip(titlebarWidth: number, placement = 'main') {
      const disposers = paneIds.map(id =>
        registry.register({
          area: 'panes',
          data: { placement },
          id,
          render: () => <div>{id}</div>,
          title: id
        })
      )

      disposePane = () => disposers.forEach(dispose => dispose())
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      vi.stubGlobal(
        'ResizeObserver',
        class {
          observe() {}
          unobserve() {}
          disconnect() {}
        }
      )
      // jsdom has no layout: give usePanelTitlebar real chrome rects so it
      // picks the tabs-in-titlebar layout (wide) or below-controls (narrow).
      vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
        if (this.matches('[data-titlebar-cluster="left"]')) {
          return { left: 0, right: 100 } as DOMRect
        }

        if (this.matches('[data-titlebar-cluster="right"]')) {
          return { left: titlebarWidth - 100, right: titlebarWidth } as DOMRect
        }

        return { left: 0, right: titlebarWidth, width: titlebarWidth } as DOMRect
      })
      const leftControls = globalThis.document.createElement('div')
      leftControls.dataset.titlebarCluster = 'left'
      const rightControls = globalThis.document.createElement('div')
      rightControls.dataset.titlebarCluster = 'right'
      globalThis.document.body.append(leftControls, rightControls)

      render(<TreeGroup leftEdge node={{ ...terminalGroup(false), panes: paneIds }} rightEdge topEdge />)

      const zone = container!.querySelector<HTMLElement>('[data-tree-group]')!
      expect(zone.querySelectorAll('[data-tree-tab]').length).toBe(paneIds.length)

      return {
        handles: [...zone.querySelectorAll<HTMLElement>('[data-window-drag-handle]')],
        strip: zone.querySelector<HTMLElement>('[data-zone-tabstrip]')!
      }
    }

    it.each(['zone', 'default'] as const)('preserves the saved %s hide-tabs preference', source => {
      mountCrowdedStrip(800, 'left')
      const originalDefault = $tabStripDefault.get()

      const node: GroupNode = {
        ...terminalGroup(false),
        panes: paneIds,
        tabStrip: source === 'zone' ? 'never' : undefined
      }

      try {
        if (source === 'default') {
          act(() => setTabStripDefault('never'))
        }

        const savedDefault = localStorage.getItem('hermes.desktop.tabStripDefault')
        render(<TreeGroup leftEdge node={node} rightEdge topEdge />)
        expect(container!.querySelector('[data-zone-tabstrip]')).toBeNull()
        expect(node.tabStrip).toBe(source === 'zone' ? 'never' : undefined)
        expect(localStorage.getItem('hermes.desktop.tabStripDefault')).toBe(savedDefault)
      } finally {
        act(() => setTabStripDefault(originalDefault))
      }
    })

    it('renders page controls inside their own panel in the normal tab space', () => {
      mountCrowdedStrip(800)
      act(() => {
        registry.register({
          area: 'panes',
          id: 'terminal',
          title: 'Kanban',
          data: {
            placement: 'main',
            headerVeto: true,
            headerContent: () => <button type="button">Board picker</button>
          },
          render: () => <div>Board content</div>
        })
      })
      const header = container!.querySelector<HTMLElement>('[data-panel-page-header]')!
      expect(header.textContent).toBe('Board picker')
      expect(header.closest('[data-tree-group]')?.getAttribute('data-tree-group')).toBe('terminal-zone')
      expect(header.querySelector('[data-slot="pane-tab"]')).not.toBeNull()
      expect(container!.querySelector('[data-zone-tabstrip]')).toBeNull()
    })

    it('keeps a fixed drag handle outside the tablist when tabs share the titlebar', () => {
      const { handles } = mountCrowdedStrip(800)
      const fixed = handles.filter(handle => !handle.closest('[role="tablist"]') && handle.style.width !== '')

      expect(fixed.length).toBeGreaterThan(0)
    })
  })

  it('hides a keep-alive pane through hide/restore while a plain pane still parks', () => {
    const disposeBrowser = registry.register({
      area: 'panes',
      data: { lifecycleKeepAlive: true },
      id: 'terminal',
      title: 'Browser',
      render: () => <input data-live-page defaultValue="original" />
    })

    const disposePlain = registry.register({
      area: 'panes',
      id: 'plain',
      title: 'Plain',
      render: () => <input data-plain-page defaultValue="original" />
    })

    disposePane = () => {
      disposeBrowser()
      disposePlain()
    }

    vi.stubGlobal('CSS', { escape: (value: string) => value })
    stubResizeObserver()

    const both: GroupNode = { ...terminalGroup(false), panes: ['terminal', 'plain'] }

    render(<TreeGroup node={both} parentAxis="row" />)
    // Activate the plain tab once so it enters the hot-hidden cache too.
    render(<TreeGroup node={{ ...both, active: 'plain' }} parentAxis="row" />)
    render(<TreeGroup node={both} parentAxis="row" />)
    const page = container!.querySelector<HTMLInputElement>('[data-live-page]')!
    page.value = 'unsaved page state'
    expect(container!.querySelector('[data-plain-page]')).not.toBeNull()
    expect(toggle('Hide')).not.toBeNull()

    render(<TreeGroup node={{ ...both, minimized: true }} parentAxis="row" />)
    // The guest stays mounted (same node, same state) but hidden and inert;
    // the ordinary pane parks exactly as before.
    expect(container!.querySelector('[data-live-page]')).toBe(page)
    expect(page.closest('[data-pane-hidden]')?.hasAttribute('inert')).toBe(true)
    expect(container!.querySelector('[data-plain-page]')).toBeNull()

    render(<TreeGroup node={both} parentAxis="row" />)
    expect(container!.querySelector('[data-live-page]')).toBe(page)
    expect(page.value).toBe('unsaved page state')
    expect(page.closest('[data-pane-hidden]')).toBeNull()
  })

  it('keeps a top-edge strip inside its panel and yields native drag while moving a pane', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id: 'terminal',
      title: 'Terminal',
      render: () => <div>Terminal</div>
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    )
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

import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { TITLEBAR_HEIGHT, TITLEBAR_TABS_GAP, TITLEBAR_TABS_HEIGHT } from '@/app/shell/titlebar'
import { registry } from '@/contrib/registry'

import { group, type GroupNode, split } from '../model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  collapseTreePane,
  markCollapsePane,
  registerPaneCloser,
  setTreeGroupMinimized,
  setTreeGroupTabStrip,
  tabStripVisibleForGroup
} from '../store'

import { TreeGroup } from './tree-group'

/**
 * A collapsed pane must leave a visible way back. Two live repros, one
 * invariant:
 *   1. #91223 — double-clicking Sessions/Bots must not hide the strip (and
 *      an explicit `never` still cannot, because hide-only chrome's only
 *      handle lives on that strip).
 *   2. Clicking the active tab of a docked tool/plugin tile must not fold
 *      the zone; if the zone *is* collapsed (chevron), the tab chip stays.
 */

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const disposers: (() => void)[] = []

beforeEach(async () => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

function registerPane(id: string, data: Record<string, unknown>, title = id) {
  disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title }))
}

function LiveTreeGroup({
  index = 0,
  parentAxis,
  topEdge = false
}: {
  index?: number
  parentAxis: 'column' | 'row'
  topEdge?: boolean
}) {
  useStore($layoutTree)

  const node = $layoutTree.get()!
  const zone = (node.type === 'split' ? node.children[index] : node) as GroupNode

  return <TreeGroup node={zone} parentAxis={parentAxis} topEdge={topEdge} />
}

const tablist = () => globalThis.document.querySelector('[role="tablist"]')
const tabEl = (paneId: string) => globalThis.document.querySelector(`[data-tree-tab="${paneId}"]`)

const tap = (target: Element) => {
  fireEvent.pointerDown(target, { button: 0, clientX: 10, clientY: 10, pointerType: 'mouse' })
  fireEvent.pointerUp(window, { button: 0, clientX: 10, clientY: 10, pointerType: 'mouse' })
}

const doubleTap = (target: Element) => {
  tap(target)
  tap(target)
}

const zoneAt = (index: number) => {
  const node = $layoutTree.get()!

  return (node.type === 'split' ? node.children[index] : node) as GroupNode
}

describe('Sessions/Bots strip — #91223', () => {
  beforeEach(() => {
    registerPane('sessions', { hideOnly: true, placement: 'left' }, 'Sessions')
    registerPane('hermes-bots:pane', { hideOnly: true, placement: 'left' }, 'Bots')
    registerPane('workspace', { placement: 'main', uncloseable: true }, 'Chat')
    $layoutTree.set(
      split('row', [
        group(['sessions', 'hermes-bots:pane'], { active: 'sessions', id: 'g-side' }),
        group(['workspace'], { active: 'workspace', id: 'g-main' })
      ])
    )
  })

  it('keeps standing sidebar tabs below the titlebar across window widths', () => {
    const left = document.createElement('div')
    const right = document.createElement('div')
    left.dataset.titlebarCluster = 'left'
    right.dataset.titlebarCluster = 'right'
    left.getBoundingClientRect = () => ({ left: 0, right: 70 }) as DOMRect
    right.getBoundingClientRect = () => ({ left: 1000, right: 1100 }) as DOMRect
    document.body.append(left, right)

    try {
      const { container } = render(<LiveTreeGroup parentAxis="row" topEdge />)
      const zone = container.querySelector<HTMLElement>('[data-tree-group]')!
      const header = container.querySelector<HTMLElement>('[data-panel-header]')!
      let width = 237
      zone.getBoundingClientRect = () => ({ left: 0, right: width, width }) as DOMRect

      for (width of [237, 500, 237]) {
        act(() => window.dispatchEvent(new Event('resize')))
        expect(header.style.height).toBe(`${TITLEBAR_HEIGHT + TITLEBAR_TABS_HEIGHT + TITLEBAR_TABS_GAP}px`)
        expect(container.querySelector('[data-zone-tabstrip]')?.getAttribute('style')).toContain(
          `height: ${TITLEBAR_TABS_HEIGHT}px`
        )
        expect(container.querySelector('[data-zone-tabstrip]')?.classList.contains('absolute')).toBe(true)
      }
    } finally {
      left.remove()
      right.remove()
    }
  })

  it('double-clicking the Sessions tab leaves the strip and both chips', () => {
    render(<LiveTreeGroup parentAxis="row" />)

    doubleTap(tabEl('sessions')!)

    expect(tablist()).toBeTruthy()
    expect(tabEl('sessions')).toBeTruthy()
    expect(tabEl('hermes-bots:pane')).toBeTruthy()
    expect(zoneAt(0).minimized).toBeFalsy()
    expect(zoneAt(0).tabStrip).toBeUndefined()
  })

  it('double-clicking the Bots tab leaves the strip too', () => {
    render(<LiveTreeGroup parentAxis="row" />)

    tap(tabEl('hermes-bots:pane')!)
    doubleTap(tabEl('hermes-bots:pane')!)

    expect(tablist()).toBeTruthy()
    expect(tabEl('sessions')).toBeTruthy()
    expect(tabEl('hermes-bots:pane')).toBeTruthy()
  })

  it('tapping the strip gutter does not collapse the sidebar', () => {
    render(<LiveTreeGroup parentAxis="row" />)

    tap(globalThis.document.querySelector('[data-zone-tabstrip="g-side"]')!)

    expect(zoneAt(0).minimized).toBeFalsy()
    expect(tabEl('sessions')).toBeTruthy()
    expect(tabEl('hermes-bots:pane')).toBeTruthy()
  })

  it('an explicit never hides the sessions/Bots strip', () => {
    setTreeGroupTabStrip('g-side', 'never')
    expect(tabStripVisibleForGroup(zoneAt(0))).toBe(false)

    render(<LiveTreeGroup parentAxis="row" />)

    expect(tablist()).toBeNull()
  })
})

describe('docked tool tile — collapsing keeps the restore chip', () => {
  beforeEach(() => {
    registerPane('workspace', { placement: 'main', uncloseable: true }, 'Chat')
    registerPane('hermes-bots:routines', { placement: 'main', width: '250px' }, 'Cronjobs')
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['hermes-bots:routines'], { active: 'hermes-bots:routines', id: 'g-routines' })
      ])
    )
  })

  it('clicking the active tab does not collapse the tile or drop its label', () => {
    render(<LiveTreeGroup index={1} parentAxis="row" />)

    expect(tabEl('hermes-bots:routines')).toBeTruthy()
    tap(tabEl('hermes-bots:routines')!)

    expect(zoneAt(1).minimized).toBeFalsy()
    expect(tabEl('hermes-bots:routines')).toBeTruthy()
    expect(tabEl('hermes-bots:routines')?.textContent).toMatch(/cronjobs/i)
  })

  it('tapping the strip gutter does not collapse a lone docked tile', () => {
    render(<LiveTreeGroup index={1} parentAxis="row" />)

    tap(globalThis.document.querySelector('[data-zone-tabstrip="g-routines"]')!)

    expect(zoneAt(1).minimized).toBeFalsy()
    expect(tabEl('hermes-bots:routines')).toBeTruthy()
  })

  it('chevron-collapse of a row-docked tile keeps the tab as a restore handle', () => {
    render(<LiveTreeGroup index={1} parentAxis="row" />)

    fireEvent.click(globalThis.document.querySelector('[data-tree-group="g-routines"] button[aria-label="Minimize"]')!)

    expect(zoneAt(1).minimized).toBe(true)
    expect(tabEl('hermes-bots:routines')).toBeTruthy()
    expect(tabEl('hermes-bots:routines')?.textContent).toMatch(/cronjobs/i)
  })

  it('collapsing via the store does not dismiss the pane from the tree', () => {
    collapseTreePane('hermes-bots:routines')

    expect(zoneAt(1).minimized).toBe(true)
    expect($dismissedPanes.get().has('hermes-bots:routines')).toBe(false)
    expect(zoneAt(1).panes).toContain('hermes-bots:routines')

    render(<LiveTreeGroup index={1} parentAxis="row" />)

    expect(tabEl('hermes-bots:routines')).toBeTruthy()
  })
})

describe('a stacked tool zone collapsed in a row keeps a vertical restore rail', () => {
  beforeEach(() => {
    registerPane('workspace', { placement: 'main', uncloseable: true }, 'Chat')
    registerPane('terminal', { placement: 'bottom' }, 'Terminal')
    registerPane('logs', { placement: 'bottom' }, 'Logs')
    markCollapsePane('terminal')
    markCollapsePane('logs')
    registerPaneCloser('terminal', () => undefined)
    registerPaneCloser('logs', () => undefined)
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['terminal', 'logs'], { active: 'terminal', id: 'g-tools' })
      ])
    )
  })

  it('keeps both tab labels after minimize so either pane can be restored', () => {
    setTreeGroupMinimized('g-tools', true)
    render(<LiveTreeGroup index={1} parentAxis="row" />)

    expect(tabEl('terminal')).toBeTruthy()
    expect(tabEl('logs')).toBeTruthy()
    expect(tabEl('terminal')?.getAttribute('data-vertical')).toBe('true')
    expect(tabEl('logs')?.getAttribute('data-vertical')).toBe('true')
  })
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from '../model'
import {
  $layoutTree,
  declareDefaultTree,
  markCollapsePane,
  noteActiveTreeGroup,
  registerPaneCloser,
  setTreeGroupMinimized
} from '../store'

import { TreeGroup } from './tree-group'

// Ground truth for "right-clicking logs doesn't even show Close, and ⌘W
// doesn't close it". Renders the REAL zone renderer and opens the REAL
// context menu.

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  // jsdom lacks CSS.escape, which tab-strip-scroll uses in a layout effect.
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const disposers: (() => void)[] = []

beforeEach(async () => {
  window.localStorage.clear()

  // Per-test isolation: earlier cases dismiss / hide panes, and both records
  // live in module state that survives into the next test.
  const { $dismissedPanes, $hiddenTreePanes } = await import('../store')
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['terminal', { placement: 'bottom' }],
    ['logs', { placement: 'bottom' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }

  markCollapsePane('terminal')
  markCollapsePane('logs')
  registerPaneCloser('terminal', () => undefined)
  registerPaneCloser('logs', () => undefined)
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

/** Radix opens a ContextMenu on contextmenu after a pointerdown positions it. */
function openContextMenu(target: HTMLElement) {
  fireEvent.pointerDown(target, { button: 2, pointerType: 'mouse' })
  fireEvent.contextMenu(target, { button: 2 })
}

const zoneAt = (index: number) => {
  const node = $layoutTree.get()!

  return (node.type === 'split' ? node.children[index] : node) as never
}

const tabEl = (paneId: string) => document.querySelector<HTMLElement>(`[data-tree-tab="${paneId}"]`)

describe('right-clicking a tool panel tab', () => {
  it('offers Close when logs is STACKED with the terminal', async () => {
    declareDefaultTree(
      split('column', [
        group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        group(['terminal', 'logs'], { active: 'logs', id: 'grp-tools' })
      ])
    )
    render(<TreeGroup node={zoneAt(1)} parentAxis="column" />)

    openContextMenu(tabEl('logs')!)

    expect(await screen.findByRole('menuitem', { name: /^close$/i })).toBeTruthy()
  })

  it('offers Close when logs is ALONE in its zone', async () => {
    declareDefaultTree(
      split('column', [
        group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        group(['logs'], { active: 'logs', id: 'grp-logs' })
      ])
    )
    render(<TreeGroup node={zoneAt(1)} parentAxis="column" />)

    const tab = tabEl('logs')
    expect(tab).toBeTruthy()

    openContextMenu(tab!)

    expect(await screen.findByRole('menuitem', { name: /^close$/i })).toBeTruthy()
  })

  it('offers Close while the zone is MINIMIZED to its rail', async () => {
    declareDefaultTree(
      split('column', [
        group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        group(['terminal', 'logs'], { active: 'logs', id: 'grp-tools' })
      ])
    )
    setTreeGroupMinimized('grp-tools', true)
    render(<TreeGroup node={zoneAt(1)} parentAxis="column" />)

    openContextMenu(tabEl('logs')!)

    expect(await screen.findByRole('menuitem', { name: /^close$/i })).toBeTruthy()
  })
})

describe('⌘W over a focused tool panel', () => {
  it('closes the logs tab and the toggle brings it back', async () => {
    const { closeActiveTab } = await import('@/app/chat/close-tab')
    const { allPaneIds } = await import('../model')
    const { revealTreePane, setPaneCollapsed } = await import('../store')

    declareDefaultTree(
      split('column', [
        group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        group(['terminal', 'logs'], { active: 'logs', id: 'grp-tools' })
      ])
    )

    // Production wiring (bindPaneCollapse): a store drives visibility, its
    // closer collapses, its opener reveals.
    let open = true
    registerPaneCloser('logs', () => {
      open = false
      setPaneCollapsed('logs', true)
    })

    // The user clicked into the tools zone — ⌘W must act on ITS active tab.
    noteActiveTreeGroup('grp-tools')

    closeActiveTab()

    // The tab left the strip, and the toggle knows the pane is off.
    expect(allPaneIds($layoutTree.get()!)).not.toContain('logs')
    expect(open).toBe(false)

    // Toggling it back on re-adopts the pane (revealTreePane un-dismisses).
    open = true
    revealTreePane('logs')

    expect(allPaneIds($layoutTree.get()!)).toContain('logs')
  })

  it('leaves the uncloseable workspace zone to the chat rung', async () => {
    const { allPaneIds } = await import('../model')
    const { closeFocusedToolTab } = await import('../store')

    declareDefaultTree(
      split('column', [
        group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        group(['logs'], { active: 'logs', id: 'grp-logs' })
      ])
    )

    noteActiveTreeGroup('grp-main')

    // Pointer/focus on main: the tool rung must decline, not reach across and
    // close the logs pane the user isn't looking at.
    expect(closeFocusedToolTab()).toBe(false)
    expect(allPaneIds($layoutTree.get()!)).toContain('logs')
  })
})

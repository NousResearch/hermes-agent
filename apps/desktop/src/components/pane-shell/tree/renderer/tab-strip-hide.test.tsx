import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, type GroupNode, split } from '../model'
import {
  $layoutTree,
  markCollapsePane,
  registerPaneCloser,
  setTreeGroupTabStrip,
  tabStripVisibleForGroup,
  toggleTargetZoneTabStrip
} from '../store'

import { TreeGroup } from './tree-group'

/** TreeGroup reads its node from props; subscribe so store writes re-render. */
function LiveTreeGroup() {
  useStore($layoutTree)

  return <TreeGroup node={zoneAt(0)} parentAxis="column" />
}

// Pins the tab-strip hide grammar. Hiding is a COMMAND now, not a gesture: the
// pointer can no longer take the strip away by accident, and whatever does take
// it away leaves a way back that does not depend on the chrome it just removed.

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

  const { $dismissedPanes, $hiddenTreePanes } = await import('../store')
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['terminal', { placement: 'bottom' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }

  markCollapsePane('terminal')
  registerPaneCloser('terminal', () => undefined)

  $layoutTree.set(split('column', [group(['workspace', 'terminal'], { active: 'terminal', id: 'grp-main' })]))
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

const zoneAt = (index: number) => {
  const node = $layoutTree.get()!

  return (node.type === 'split' ? node.children[index] : node) as never
}

const groupNode = () => {
  const node = $layoutTree.get()!

  return (node.type === 'split' ? node.children[0] : node) as GroupNode
}

const tablist = () => globalThis.document.querySelector('[role="tablist"]')

/** Two sub-threshold taps: pointerdown on the target, pointerup on window
 *  (drag-session listens there), twice — the retired double-tap path. */
const doubleTap = (target: Element) => {
  for (let i = 0; i < 2; i++) {
    fireEvent.pointerDown(target, { button: 0, clientX: 10, clientY: 10, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { button: 0, clientX: 10, clientY: 10, pointerType: 'mouse' })
  }
}

describe('tab strip hide grammar', () => {
  it('no pointer gesture hides the strip', () => {
    render(<LiveTreeGroup />)

    // Both halves of the strip: the tab, which was always activate-only, and
    // the background, which used to answer a double-tap nothing announced.
    doubleTap(globalThis.document.querySelector('[data-tree-tab="terminal"]')!)
    doubleTap(globalThis.document.querySelector('[data-zone-tabstrip="grp-main"]')!)

    expect(tablist()).toBeTruthy()
    expect(groupNode().tabStrip).toBeUndefined()
  })

  it('renders no strip at all for a zone set to never', () => {
    setTreeGroupTabStrip('grp-main', 'never')
    render(<LiveTreeGroup />)

    expect(tablist()).toBeNull()
  })

  // The state that had no way out. The command targets the zone by
  // hover/focus/workspace fallback, so restoring the strip never depends on
  // the strip — or on any other chrome the hide took away.
  it('the toggle command reaches a zone that has no chrome left to click', () => {
    setTreeGroupTabStrip('grp-main', 'never')
    expect(tabStripVisibleForGroup(groupNode())).toBe(false)

    expect(toggleTargetZoneTabStrip()).toBe('always')
    expect(tabStripVisibleForGroup(groupNode())).toBe(true)
  })
})

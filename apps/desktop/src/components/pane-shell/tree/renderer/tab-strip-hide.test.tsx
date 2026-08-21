import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from '../model'
import { $layoutTree, markCollapsePane, registerPaneCloser } from '../store'

import { TreeGroup } from './tree-group'

/** TreeGroup reads its node from props; subscribe so store writes re-render. */
function LiveTreeGroup() {
  useStore($layoutTree)

  return <TreeGroup node={zoneAt(0)} parentAxis="column" />
}

// Pins the tab-strip hide grammar: the double-tap hide belongs to the STRIP
// BACKGROUND alone — tabs are activate-only and must never hide the bar.

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

  return (node.type === 'split' ? node.children[0] : node) as { headerHidden?: boolean; panes: string[] }
}

const tablist = () => globalThis.document.querySelector('[role="tablist"]')

/** Two sub-threshold taps: pointerdown on the target, pointerup on window
 *  (drag-session listens there), twice — the synthesized double-tap path. */
const doubleTap = (target: Element) => {
  for (let i = 0; i < 2; i++) {
    fireEvent.pointerDown(target, { button: 0, clientX: 10, clientY: 10, pointerType: 'mouse' })
    fireEvent.pointerUp(window, { button: 0, clientX: 10, clientY: 10, pointerType: 'mouse' })
  }
}

describe('tab strip hide grammar', () => {
  it('double-clicking a tab does NOT hide the strip', () => {
    // $layoutTree.set, not declareDefaultTree — the latter only adopts into an
    // existing tree, and the store is module state that survives between tests.
    $layoutTree.set(
      split('column', [group(['workspace', 'terminal'], { active: 'terminal', id: 'grp-main' })])
    )
    render(<LiveTreeGroup />)

    const tab = globalThis.document.querySelector<HTMLElement>('[data-tree-tab="terminal"]')
    expect(tab).toBeTruthy()

    doubleTap(tab!)

    // The strip is still there and the tree never recorded a hide.
    expect(tablist()).toBeTruthy()
    expect(groupNode().headerHidden).not.toBe(true)
  })

  it('double-tapping the strip background still hides the header (documented gesture)', () => {
    $layoutTree.set(
      split('column', [group(['workspace', 'terminal'], { active: 'terminal', id: 'grp-main' })])
    )
    render(<LiveTreeGroup />)

    const strip = globalThis.document.querySelector<HTMLElement>('[data-zone-tabstrip="grp-main"]')
    expect(strip).toBeTruthy()

    doubleTap(strip!)

    expect(groupNode().headerHidden).toBe(true)
  })
})

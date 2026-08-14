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

// Pins the tab-strip hide/recovery grammar: hiding the strip is an EXPLICIT
// verb (zone menu / main tab menu) — a double-click on a tab must NOT vanish
// the bar (it used to, stranding the zone with no ✕), and a double-tap on the
// zone body restores a strip that WAS hidden explicitly.

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

const tablist = () => document.querySelector('[role="tablist"]')

const doublePointerDown = (target: Element) => {
  fireEvent.pointerDown(target, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerDown(target, { button: 0, pointerType: 'mouse' })
}

describe('tab strip hide/recovery grammar', () => {
  it('double-clicking a tab does NOT hide the strip', () => {
    // $layoutTree.set, not declareDefaultTree — the latter only adopts into an
    // existing tree, and the store is module state that survives between tests.
    $layoutTree.set(
      split('column', [group(['workspace', 'terminal'], { active: 'terminal', id: 'grp-main' })])
    )
    render(<LiveTreeGroup />)

    const tab = document.querySelector<HTMLElement>('[data-tree-tab="terminal"]')
    expect(tab).toBeTruthy()

    doublePointerDown(tab!)

    // The strip is still there and the tree never recorded a hide.
    expect(tablist()).toBeTruthy()
    expect(groupNode().headerHidden).not.toBe(true)
  })

  it('double-tapping the zone body restores an explicitly hidden strip', () => {
    $layoutTree.set(
      split('column', [group(['terminal'], { active: 'terminal', headerHidden: true, id: 'grp-tools' })])
    )
    render(<LiveTreeGroup />)

    // Hidden strip: no tablist, no ✕ anywhere in the zone.
    expect(tablist()).toBeNull()

    const body = document.querySelector<HTMLElement>('[data-tree-group="grp-tools"]')
    expect(body).toBeTruthy()

    doublePointerDown(body!)

    expect(tablist()).toBeTruthy()
    expect(groupNode().headerHidden).toBe(false)
  })

  it('a single body tap does not toggle the strip back', () => {
    $layoutTree.set(
      split('column', [group(['terminal'], { active: 'terminal', headerHidden: true, id: 'grp-tools' })])
    )
    render(<LiveTreeGroup />)

    const body = document.querySelector<HTMLElement>('[data-tree-group="grp-tools"]')!
    fireEvent.pointerDown(body, { button: 0, pointerType: 'mouse' })

    expect(tablist()).toBeNull()
    expect(groupNode().headerHidden).toBe(true)
  })
})

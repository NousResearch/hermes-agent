import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { allPaneIds, group, split } from '../model'
import { $layoutTree } from '../store'

import { TreeGroup } from './tree-group'

// The hover ✕ and middle-click are ONE affordance in two shapes: whichever tabs
// the pointer gesture can close must advertise it. A tab that closes on
// middle-click but hides its ✕ is a close verb the user cannot discover, and a
// tab that shows a ✕ it will not honor is a dead control. This asserts the
// equivalence over every tab kind the app registers, so the ✕ can never be
// wired (or un-wired) on its own again.

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

/** Every tab kind that shares a strip, paired with what its chrome declares. */
const PANES: readonly (readonly [string, Record<string, unknown>])[] = [
  // Standing chrome: no close gesture of any kind.
  ['sessions', { hideOnly: true, placement: 'left' }],
  // A plain side pane: closes, so it must say so.
  ['files', { placement: 'right' }],
  // The one surface that cannot leave the tree.
  ['workspace', { placement: 'main', uncloseable: true }],
  // A mirrored session tile — `placement: 'main'` but closeable.
  ['session-tile:abc', { placement: 'main' }]
]

beforeEach(async () => {
  window.localStorage.clear()

  const { $dismissedPanes, $hiddenTreePanes } = await import('../store')
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of PANES) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

/** All four panes in ONE zone, so every tab renders in the same strip. */
function renderOneStrip() {
  $layoutTree.set(
    split('row', [
      group(
        PANES.map(([id]) => id),
        { active: 'workspace', id: 'grp-all' }
      ),
      group(['spacer'], { id: 'grp-spacer' })
    ])
  )

  const node = $layoutTree.get()!
  const zone = (node.type === 'split' ? node.children[0] : node) as never

  render(<TreeGroup node={zone} parentAxis="row" />)
}

const tabEl = (paneId: string) => document.querySelector<HTMLElement>(`[data-tree-tab="${paneId}"]`)

/** Does this tab advertise a ✕? */
const hasCloseButton = (paneId: string) => Boolean(tabEl(paneId)?.querySelector('button[aria-label]'))

/** Does the middle-click gesture actually close this tab? Observed through the
 *  tree, not through a spy — a pane that is gone stopped being a tab. */
function middleClickCloses(paneId: string): boolean {
  const tab = tabEl(paneId)!
  fireEvent.pointerDown(tab, { button: 1 })
  fireEvent.pointerUp(tab, { button: 1 })

  return !allPaneIds($layoutTree.get()!).includes(paneId)
}

describe('a tab advertises exactly the close gesture it honors', () => {
  for (const [paneId] of PANES) {
    it(`${paneId}: ✕ presence matches middle-click`, () => {
      renderOneStrip()

      expect(tabEl(paneId)).toBeTruthy()

      const advertised = hasCloseButton(paneId)

      expect(advertised).toBe(middleClickCloses(paneId))
    })
  }
})

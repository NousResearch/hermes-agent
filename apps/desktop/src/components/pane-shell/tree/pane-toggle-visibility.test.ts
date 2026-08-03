import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from './model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  bindPaneVisibility,
  isPaneVisible,
  setTreeGroupMinimized,
  togglePaneVisible
} from './store'

// The bug class, across EVERY pane kind — not just the terminal she reported.
//
// A toggle that flips its own boolean diverges from the tree the moment
// anything else moves the pane: stacked behind a sibling tab, folded into a
// minimized zone, closed with ⌘W. The store then says "open" while nothing is
// on screen, the press re-asserts a value it already held, and the key reads
// as dead. Same shape for the COLLAPSE-style tool panels (terminal, logs) and
// the HIDE-style panes (files, review) — proven here with one table.

const disposers: (() => void)[] = []

/** Bind through the REAL production function, with the closer/opener pair the
 *  controller gives it. */
function bindVisibility(paneId: string, $open: ReturnType<typeof atom<boolean>>) {
  bindPaneVisibility(
    paneId,
    $open,
    () => $open.set(false),
    () => $open.set(true)
  )
}

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['files', { placement: 'right' }],
    ['review', { placement: 'right' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('a hide-style pane stacked behind a sibling tab', () => {
  it('comes forward on the first press instead of swallowing it', () => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['files', 'review'], { active: 'files', id: 'g-right' })
      ])
    )

    const $review = atom(true)
    bindVisibility('review', $review)

    // The column is open but showing FILES, so the diff is not on screen —
    // and $reviewOpen is true, which is what made ⌘G a no-op.
    expect(isPaneVisible('review')).toBe(false)

    togglePaneVisible('review')

    expect(isPaneVisible('review')).toBe(true)
  })
})

describe('a hide-style pane inside a minimized zone', () => {
  it('un-minimizes on the first press', () => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['files'], { active: 'files', id: 'g-files' })
      ])
    )

    const $files = atom(true)
    bindVisibility('files', $files)
    setTreeGroupMinimized('g-files', true)

    expect(isPaneVisible('files')).toBe(false)

    togglePaneVisible('files')

    expect(isPaneVisible('files')).toBe(true)
  })
})

describe('the toggle round-trip', () => {
  it('closes a visible hide-style pane through its own store', () => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'g-main' }),
        group(['files'], { active: 'files', id: 'g-files' })
      ])
    )

    const $files = atom(true)
    bindVisibility('files', $files)

    expect(isPaneVisible('files')).toBe(true)

    togglePaneVisible('files')

    // Close routes through the registered closer, so the pane's own toggle
    // store stays truthful rather than being bypassed by a tree edit.
    expect($files.get()).toBe(false)
    expect(isPaneVisible('files')).toBe(false)

    togglePaneVisible('files')

    expect($files.get()).toBe(true)
    expect(isPaneVisible('files')).toBe(true)
  })
})

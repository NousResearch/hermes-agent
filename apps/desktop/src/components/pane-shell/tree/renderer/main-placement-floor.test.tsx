import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { stubResizeObserver } from '@/test/jsdom'

import { group, split } from '../model'
import { $hiddenTreePanes, $layoutTree } from '../store'

import { subtreeGone, type TrackContext } from './track-model'
import { TreeSplit } from './tree-split'

// Ground truth for "the main pane always shows, no matter what". Emptying the
// main strip must never let a sibling absorb the center — the failure Bot Mode
// shipped, where closing the last bot chat left the roster sidebar stretched
// across the whole window. Chrome toggles over a tool panel must still collapse.

beforeAll(() => {
  stubResizeObserver()
  // jsdom lacks CSS.escape, which tab-strip-scroll uses in a layout effect.
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
})

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['sessions', { placement: 'left', width: '237px' }],
    ['workspace', { placement: 'main', uncloseable: true }],
    ['session-tile:a', { placement: 'main' }],
    ['terminal', { placement: 'bottom', height: '38vh' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  cleanup()
  $hiddenTreePanes.set(new Set())
  $layoutTree.set(null)
  disposers.splice(0).forEach(dispose => dispose())
})

/** The split-child WRAPPER the renderer sizes — `display:none` on it is what
 *  hands a zone's space to its siblings. */
const zoneWrapper = (groupId: string) =>
  document.querySelector<HTMLElement>(`[data-tree-group="${groupId}"]`)?.closest<HTMLElement>('[style]')

const ctx = (gone: string[]): TrackContext => ({
  overrides: {},
  paneFor: id => registry.getArea('panes').find(pane => pane.id === id),
  paneGone: id => gone.includes(id)
})

describe('subtreeGone', () => {
  it('keeps a main-bearing subtree even when every one of its panes is gone', () => {
    const main = group(['workspace', 'session-tile:a'], { id: 'grp-main' })

    expect(subtreeGone(main, ctx(['workspace', 'session-tile:a']))).toBe(false)
  })

  it('still collapses a tool zone whose panes a chrome toggle hid', () => {
    expect(subtreeGone(group(['terminal'], { id: 'grp-tools' }), ctx(['terminal']))).toBe(true)
  })

  it('still collapses a main pane whose plugin has not registered yet', () => {
    // No contribution means no placement to read — the zone stays folded until
    // the plugin loads rather than flashing an empty placeholder.
    expect(subtreeGone(group(['plugin-workspace:late'], { id: 'grp-late' }), ctx(['plugin-workspace:late']))).toBe(true)
  })
})

describe('the main placement floor in the renderer', () => {
  it('renders the main zone with every main tab hidden, and leaves tool zones absorbable', () => {
    const tree = split('row', [
      group(['sessions'], { id: 'grp-sessions' }),
      split('column', [
        group(['workspace', 'session-tile:a'], { id: 'grp-main' }),
        group(['terminal'], { id: 'grp-tools' })
      ])
    ])

    $layoutTree.set(tree)
    $hiddenTreePanes.set(new Set(['workspace', 'session-tile:a', 'terminal']))

    render(<TreeSplit node={tree} root rootRow />)

    const main = zoneWrapper('grp-main')

    expect(main).toBeTruthy()
    expect(main!.style.display).not.toBe('none')
    expect(zoneWrapper('grp-sessions')?.style.display).not.toBe('none')
    expect(zoneWrapper('grp-tools')?.style.display).toBe('none')
  })
})

describe('vertical sash vs leading-pane scrollbar (#99867)', () => {
  it('keeps the grab band entirely in the trailing pane', () => {
    const tree = split('row', [
      group(['sessions'], { id: 'grp-sessions' }),
      group(['workspace'], { id: 'grp-main' })
    ])

    $layoutTree.set(tree)
    render(<TreeSplit node={tree} root rootRow />)

    const sashes = document.querySelectorAll('[data-sash-overlap="trailing"]')

    expect(sashes.length).toBeGreaterThan(0)
    for (const sash of sashes) {
      expect(sash.className).toContain('w-[8px]')
      expect(sash.className).not.toContain('-translate-x-[1px]')
    }
  })
})

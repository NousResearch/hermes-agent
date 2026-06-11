import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarSelection,
  clearSidebarSelection,
  pruneSidebarSelection,
  rangeSelectSessions,
  toggleSessionSelected
} from './sidebar-selection'

const ORDER = ['a', 'b', 'c', 'd', 'e']

beforeEach(() => {
  $sidebarSelection.set({ ids: [], section: null })
})

describe('toggleSessionSelected', () => {
  it('starts a selection and toggles membership off back to empty', () => {
    toggleSessionSelected('sessions', 'a')
    expect($sidebarSelection.get()).toEqual({ ids: ['a'], section: 'sessions' })

    toggleSessionSelected('sessions', 'b')
    expect($sidebarSelection.get().ids).toEqual(['a', 'b'])

    toggleSessionSelected('sessions', 'a')
    expect($sidebarSelection.get().ids).toEqual(['b'])

    toggleSessionSelected('sessions', 'b')
    expect($sidebarSelection.get()).toEqual({ ids: [], section: null })
  })

  it('restarts in the new section instead of merging cross-section rows', () => {
    toggleSessionSelected('sessions', 'a')
    toggleSessionSelected('pinned', 'p1')

    expect($sidebarSelection.get()).toEqual({ ids: ['p1'], section: 'pinned' })
  })

  it('scopes messaging selections per platform', () => {
    toggleSessionSelected('messaging:telegram', 't1')
    toggleSessionSelected('messaging:discord', 'd1')

    expect($sidebarSelection.get()).toEqual({ ids: ['d1'], section: 'messaging:discord' })
  })
})

describe('rangeSelectSessions', () => {
  it('selects the contiguous run from the anchor, in either direction', () => {
    toggleSessionSelected('sessions', 'b')
    rangeSelectSessions('sessions', 'd', ORDER)
    expect($sidebarSelection.get().ids).toEqual(['b', 'c', 'd'])

    // New anchor is the clicked id; range back up the list unions in place.
    rangeSelectSessions('sessions', 'a', ORDER)
    expect(new Set($sidebarSelection.get().ids)).toEqual(new Set(['a', 'b', 'c', 'd']))
  })

  it('falls back to a plain toggle without a usable anchor', () => {
    rangeSelectSessions('sessions', 'c', ORDER)
    expect($sidebarSelection.get()).toEqual({ ids: ['c'], section: 'sessions' })

    // Anchor from another section doesn't leak into this one.
    toggleSessionSelected('pinned', 'p1')
    rangeSelectSessions('sessions', 'b', ORDER)
    expect($sidebarSelection.get()).toEqual({ ids: ['b'], section: 'sessions' })
  })

  it('keeps the clicked id as the next range anchor', () => {
    toggleSessionSelected('sessions', 'a')
    rangeSelectSessions('sessions', 'c', ORDER)
    rangeSelectSessions('sessions', 'e', ORDER)

    expect($sidebarSelection.get().ids).toEqual(['a', 'b', 'c', 'd', 'e'])
  })

  it('seeds a cold range from the active row so the starting point stays selected', () => {
    // No selection yet; the open session ("b") is the implicit anchor.
    rangeSelectSessions('sessions', 'd', ORDER, 'b')

    expect($sidebarSelection.get().ids).toEqual(['b', 'c', 'd'])
  })

  it('ignores the seed when it equals the target or is not in the list', () => {
    rangeSelectSessions('sessions', 'c', ORDER, 'c')
    expect($sidebarSelection.get()).toEqual({ ids: ['c'], section: 'sessions' })

    clearSidebarSelection()
    rangeSelectSessions('sessions', 'c', ORDER, 'not-in-list')
    expect($sidebarSelection.get()).toEqual({ ids: ['c'], section: 'sessions' })
  })

  it('prefers the in-section anchor over the seed once a selection exists', () => {
    toggleSessionSelected('sessions', 'd')
    rangeSelectSessions('sessions', 'e', ORDER, 'a')

    expect($sidebarSelection.get().ids).toEqual(['d', 'e'])
  })
})

describe('pruneSidebarSelection', () => {
  const rows = (...ids: string[]) => ids.map(id => ({ id }))

  it('drops ids that left the section and clears when none remain', () => {
    toggleSessionSelected('sessions', 'a')
    toggleSessionSelected('sessions', 'b')

    pruneSidebarSelection('sessions', rows('b', 'c'))
    expect($sidebarSelection.get()).toEqual({ ids: ['b'], section: 'sessions' })

    pruneSidebarSelection('sessions', rows('c'))
    expect($sidebarSelection.get()).toEqual({ ids: [], section: null })
  })

  it('ignores prunes aimed at a different section', () => {
    toggleSessionSelected('archived', 'x')
    pruneSidebarSelection('sessions', rows('y'))

    expect($sidebarSelection.get()).toEqual({ ids: ['x'], section: 'archived' })
  })

  it('keeps the selection intact across a transient EMPTY row list', () => {
    // A background refresh can hand the section zero rows for a beat; nuking
    // the selection then is the deselects-everything-but-the-last bug.
    toggleSessionSelected('sessions', 'a')
    toggleSessionSelected('sessions', 'b')

    pruneSidebarSelection('sessions', [])

    expect($sidebarSelection.get().ids).toEqual(['a', 'b'])
  })

  it('remaps a compression-rotated id to the new live tip instead of dropping it', () => {
    toggleSessionSelected('sessions', 'a')
    toggleSessionSelected('sessions', 'b')

    // Auto-compression rotated b → b-tip (lineage root b) mid-selection.
    pruneSidebarSelection('sessions', [{ id: 'a' }, { id: 'b-tip', _lineage_root_id: 'b' }, { id: 'c' }])

    expect($sidebarSelection.get().ids).toEqual(['a', 'b-tip'])

    // The remapped id stays a usable range anchor.
    rangeSelectSessions('sessions', 'c', ['a', 'b-tip', 'c'])
    expect($sidebarSelection.get().ids).toEqual(['a', 'b-tip', 'c'])
  })

  it('collapses duplicates when a rotation lands on an already-selected tip', () => {
    toggleSessionSelected('sessions', 'b')
    toggleSessionSelected('sessions', 'b-tip')

    pruneSidebarSelection('sessions', [{ id: 'b-tip', _lineage_root_id: 'b' }])

    expect($sidebarSelection.get().ids).toEqual(['b-tip'])
  })
})

describe('clearSidebarSelection', () => {
  it('resets to the empty selection', () => {
    toggleSessionSelected('archived', 'x')
    clearSidebarSelection()

    expect($sidebarSelection.get()).toEqual({ ids: [], section: null })
  })
})

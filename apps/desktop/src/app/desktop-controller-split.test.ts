import { describe, expect, it } from 'vitest'

import { resolveOpenSessionDispatch } from './desktop-controller-utils'

// The open-in-active-pane dispatcher's decision table (design §3.5). The
// invariant under test: the same session may never be open in both panes —
// opening a duplicate focuses the pane that already shows it.

const base = {
  activePaneId: 'main' as const,
  mainSelectedStoredSessionId: 'stored-main',
  splitOpen: true,
  splitStoredSessionId: 'stored-split',
  storedSessionId: 'stored-other'
}

describe('resolveOpenSessionDispatch', () => {
  it('split closed: always the plain navigate path (single-pane behavior)', () => {
    expect(
      resolveOpenSessionDispatch({ ...base, splitOpen: false, storedSessionId: 'stored-split' })
    ).toEqual({ kind: 'navigate' })

    expect(
      resolveOpenSessionDispatch({
        ...base,
        activePaneId: 'split',
        splitOpen: false,
        splitStoredSessionId: null
      })
    ).toEqual({ kind: 'navigate' })
  })

  it('main active: a fresh session opens via the router', () => {
    expect(resolveOpenSessionDispatch(base)).toEqual({ kind: 'navigate' })
  })

  it('split active: a fresh session opens pane-locally in the split (URL untouched)', () => {
    expect(resolveOpenSessionDispatch({ ...base, activePaneId: 'split' })).toEqual({ kind: 'open-in-split' })
  })

  it("duplicate of the split's session focuses the split — from either active pane", () => {
    expect(resolveOpenSessionDispatch({ ...base, storedSessionId: 'stored-split' })).toEqual({ kind: 'focus-split' })

    expect(
      resolveOpenSessionDispatch({ ...base, activePaneId: 'split', storedSessionId: 'stored-split' })
    ).toEqual({ kind: 'focus-split' })
  })

  it("split active: duplicate of the MAIN pane's session focuses the main pane instead of double-opening", () => {
    expect(
      resolveOpenSessionDispatch({ ...base, activePaneId: 'split', storedSessionId: 'stored-main' })
    ).toEqual({ kind: 'focus-main' })
  })

  it("main active: re-opening the main pane's own session keeps today's re-navigate", () => {
    expect(resolveOpenSessionDispatch({ ...base, storedSessionId: 'stored-main' })).toEqual({ kind: 'navigate' })
  })
})

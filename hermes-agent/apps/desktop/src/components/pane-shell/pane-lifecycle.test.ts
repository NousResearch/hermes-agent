import { describe, expect, it } from 'vitest'

import { emptyPaneLifecycleState, reconcilePaneLifecycle } from './pane-lifecycle'

const visit = (state: ReturnType<typeof emptyPaneLifecycleState>, activeId: string, paneIds: string[]) =>
  reconcilePaneLifecycle(state, { activeId, paneIds })

describe('per-zone pane lifecycle', () => {
  it('keeps a small recent hidden set and parks older panes', () => {
    let state = emptyPaneLifecycleState()

    state = visit(state, 'a', ['a', 'b', 'c', 'd'])
    state = visit(state, 'b', ['a', 'b', 'c', 'd'])
    state = visit(state, 'c', ['a', 'b', 'c', 'd'])
    state = visit(state, 'd', ['a', 'b', 'c', 'd'])

    expect(state.entries).toMatchObject({
      a: { lifecycle: 'parked' },
      b: { lifecycle: 'hot-hidden' },
      c: { lifecycle: 'hot-hidden' },
      d: { lifecycle: 'visible' }
    })
  })

  it('tracks recency independently for each zone state', () => {
    const zoneA = visit(visit(emptyPaneLifecycleState(), 'a', ['a', 'b']), 'b', ['a', 'b'])
    const zoneB = visit(emptyPaneLifecycleState(), 'x', ['x', 'y'])

    expect(zoneA.entries.a.lifecycle).toBe('hot-hidden')
    expect(zoneA.entries.b.lifecycle).toBe('visible')
    expect(zoneB.entries.x.lifecycle).toBe('visible')
    expect(zoneB.entries.y).toBeUndefined()
  })

  it('keeps a hidden terminal alive outside the normal cap', () => {
    let state = emptyPaneLifecycleState()
    const paneIds = ['terminal', 'a', 'b', 'c']

    const reconcile = (activeId: string) => {
      state = reconcilePaneLifecycle(state, {
        activeId,
        hotHiddenCap: 1,
        keepAlive: id => id === 'terminal',
        paneIds
      })
    }

    reconcile('terminal')
    reconcile('a')
    reconcile('b')
    reconcile('c')

    expect(state.entries.terminal.lifecycle).toBe('hot-hidden')
    expect(state.entries.b.lifecycle).toBe('hot-hidden')
    expect(state.entries.a.lifecycle).toBe('parked')
  })

  it('forgets panes that leave a zone and remounts a parked pane when selected', () => {
    let state = visit(emptyPaneLifecycleState(), 'a', ['a', 'b', 'c', 'd'])

    for (const active of ['b', 'c', 'd']) {
      state = visit(state, active, ['a', 'b', 'c', 'd'])
    }

    expect(state.entries.a.lifecycle).toBe('parked')

    state = visit(state, 'a', ['a', 'b', 'c'])

    expect(state.entries.a.lifecycle).toBe('visible')
    expect(state.entries.d).toBeUndefined()
  })
})

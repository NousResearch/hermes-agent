import { describe, expect, it } from 'vitest'

import { group, type LayoutNode, setSplitOrientation, split, type SplitNode, swapPanes } from './model'

const tree = (): SplitNode =>
  split(
    'row',
    [group(['workspace']), split('column', [group(['session-tile:a']), group(['session-tile:b'])], [1, 1])],
    [1, 1]
  )

describe('setSplitOrientation', () => {
  it('flips one split; a split now matching its parent flattens with proportional weights', () => {
    const t = tree()
    const inner = t.children[1] as SplitNode
    const next = setSplitOrientation(t, inner.id, 'row') as SplitNode

    expect(next.orientation).toBe('row')
    expect(next.children).toHaveLength(3)
    expect(next.weights).toEqual([1, 0.5, 0.5])
  })

  it('flips the root split (its same-orientation child flattens in)', () => {
    const t = tree()
    const next = setSplitOrientation(t, t.id, 'column') as SplitNode

    expect(next.orientation).toBe('column')
    expect(next.children).toHaveLength(3)
    expect(next.weights).toEqual([1, 0.5, 0.5])
  })

  it('leaves the tree alone for an unknown split id', () => {
    const t = tree()

    expect(setSplitOrientation(t, 'nope', 'column')).toEqual(t)
  })
})

describe('swapPanes', () => {
  it('exchanges two panes across groups and keeps active pointers', () => {
    const next = swapPanes(tree(), 'workspace', 'session-tile:b') as SplitNode
    const left = next.children[0] as Extract<LayoutNode, { type: 'group' }>
    const lower = (next.children[1] as SplitNode).children[1] as Extract<LayoutNode, { type: 'group' }>

    expect(left.panes).toEqual(['session-tile:b'])
    expect(left.active).toBe('session-tile:b')
    expect(lower.panes).toEqual(['workspace'])
    expect(lower.active).toBe('workspace')
  })

  it('is a no-op for a missing pane or a self-swap', () => {
    const t = tree()

    expect(swapPanes(t, 'workspace', 'nope')).toBe(t)
    expect(swapPanes(t, 'workspace', 'workspace')).toBe(t)
  })
})

import { describe, expect, it } from 'vitest'

import {
  arrangeChatGroups,
  type ChatArrangement,
  group,
  type GroupNode,
  type LayoutNode,
  setSplitOrientation,
  split,
  type SplitNode,
  swapPanes
} from './model'

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

const isChat = (id: string) => id === 'workspace' || id.startsWith('session-tile:')

/** Shape of a tree as text: (r a b) / (c a b) with group panes joined by |. */
const shape = (n: LayoutNode | null): string =>
  !n ? 'null' : n.type === 'group' ? n.panes.join('|') : `(${n.orientation[0]} ${n.children.map(shape).join(' ')})`

const app = () =>
  split(
    'row',
    [
      group(['sessions'], { id: 'side' }),
      group(['workspace'], { id: 'main' }),
      group(['session-tile:a'], { id: 'ga' }),
      group(['session-tile:b', 'session-tile:c'], { id: 'gb' }),
      split('column', [group(['review']), group(['terminal'])])
    ],
    [1, 2, 2, 2, 1],
    'root'
  )

const grid: ChatArrangement = {
  orientation: 'column',
  children: [
    { orientation: 'row', children: [{ pane: 'workspace' }, { pane: 'session-tile:a' }] },
    { pane: 'session-tile:c' }
  ]
}

describe('arrangeChatGroups', () => {
  it('rebuilds only the chat run; sidebar and rails keep their slots and weights', () => {
    const next = arrangeChatGroups(app(), grid, isChat) as SplitNode

    expect(shape(next)).toBe(
      '(r sessions (c (r workspace session-tile:a) session-tile:b|session-tile:c) (c review terminal))'
    )
    expect(next.weights).toEqual([1, 6, 1])
  })

  it('moves whole zones: stacked tabs and the active tab travel together', () => {
    const next = arrangeChatGroups(app(), grid, isChat) as SplitNode
    const moved = (next.children[1] as SplitNode).children[1] as GroupNode

    expect(moved.id).toBe('gb')
    expect(moved.panes).toEqual(['session-tile:b', 'session-tile:c'])
  })

  it('applies explicit weights, and flattens into a same-orientation parent proportionally', () => {
    const columns: ChatArrangement = {
      orientation: 'row',
      weights: [2, 1, 1],
      children: [{ pane: 'workspace' }, { pane: 'session-tile:a' }, { pane: 'session-tile:b' }]
    }

    const next = arrangeChatGroups(app(), columns, isChat) as SplitNode

    expect(shape(next)).toBe('(r sessions workspace session-tile:a session-tile:b|session-tile:c (c review terminal))')
    expect(next.weights).toEqual([1, 3, 1.5, 1.5, 1])
  })

  it('refuses a spec that misses a chat zone, repeats one, or names a non-chat pane', () => {
    const missing: ChatArrangement = {
      orientation: 'row',
      children: [{ pane: 'workspace' }, { pane: 'session-tile:a' }]
    }

    const twice: ChatArrangement = {
      orientation: 'row',
      children: [
        { pane: 'workspace' },
        { pane: 'session-tile:a' },
        { pane: 'session-tile:b' },
        { pane: 'session-tile:c' }
      ]
    }

    const chrome: ChatArrangement = {
      orientation: 'row',
      children: [{ pane: 'workspace' }, { pane: 'session-tile:a' }, { pane: 'session-tile:b' }, { pane: 'terminal' }]
    }

    expect(arrangeChatGroups(app(), missing, isChat)).toBeNull()
    expect(arrangeChatGroups(app(), twice, isChat)).toBeNull()
    expect(arrangeChatGroups(app(), chrome, isChat)).toBeNull()
  })

  it('refuses when a user-placed zone sits between chats', () => {
    const t = split('row', [group(['workspace']), group(['terminal']), group(['session-tile:a'])])

    const spec: ChatArrangement = {
      orientation: 'column',
      children: [{ pane: 'workspace' }, { pane: 'session-tile:a' }]
    }

    expect(arrangeChatGroups(t, spec, isChat)).toBeNull()
  })

  it('arranges a chat area nested below the root', () => {
    const t = split(
      'row',
      [
        group(['sessions']),
        split('column', [group(['workspace']), group(['session-tile:a']), group(['session-tile:b'])])
      ],
      [1, 3]
    )

    const spec: ChatArrangement = {
      orientation: 'row',
      children: [
        { pane: 'session-tile:b' },
        { orientation: 'column', children: [{ pane: 'workspace' }, { pane: 'session-tile:a' }] }
      ]
    }

    const next = arrangeChatGroups(t, spec, isChat) as SplitNode

    expect(shape(next)).toBe('(r sessions session-tile:b (c workspace session-tile:a))')
    expect(next.weights).toEqual([1, 1.5, 1.5])
  })
})

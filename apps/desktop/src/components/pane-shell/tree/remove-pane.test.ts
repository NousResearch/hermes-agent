import { describe, expect, it } from 'vitest'

import { findGroup, group, removePane, split } from './model'

describe('removePane close-neighbor selection', () => {
  it('closing a middle active tab keeps you on the tab that slides into its slot', () => {
    // [a, b, c] active=b → close b → [a, c] active=c (was right of b), not a
    const next = removePane(group(['a', 'b', 'c'], { active: 'b', id: 'g' }), 'b')

    expect(next).toMatchObject({ type: 'group', panes: ['a', 'c'], active: 'c' })
  })

  it('closing the first active tab still advances to the former next tab', () => {
    const next = removePane(group(['a', 'b', 'c'], { active: 'a', id: 'g' }), 'a')

    expect(next).toMatchObject({ type: 'group', panes: ['b', 'c'], active: 'b' })
  })

  it('closing the last active tab falls back to its left neighbor', () => {
    const next = removePane(group(['a', 'b', 'c'], { active: 'c', id: 'g' }), 'c')

    expect(next).toMatchObject({ type: 'group', panes: ['a', 'b'], active: 'b' })
  })

  it('closing a non-active tab leaves the selection alone', () => {
    const next = removePane(group(['a', 'b', 'c'], { active: 'c', id: 'g' }), 'b')

    expect(next).toMatchObject({ type: 'group', panes: ['a', 'c'], active: 'c' })
  })

  it('walks nested splits so session stacks still get the slot rule', () => {
    const tree = split('row', [
      group(['workspace'], { active: 'workspace', id: 'main' }),
      group(['tile:1', 'tile:2', 'tile:3'], { active: 'tile:2', id: 'stack' })
    ])

    const next = removePane(tree, 'tile:2')
    const stack = next ? findGroup(next, 'stack') : null

    expect(stack).toMatchObject({ panes: ['tile:1', 'tile:3'], active: 'tile:3' })
  })
})

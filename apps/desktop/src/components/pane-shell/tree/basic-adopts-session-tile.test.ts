import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { findGroupOfPane, group, split } from './model'
import { $dismissedPanes, $hiddenTreePanes, $layoutTree, applyTree } from './store'

// Repro: a session tile living in its OWN zone (split beside the chat) has no
// sibling the next preset places, so adoption falls back to the FIRST group —
// the sessions sidebar. The tile is main-placed, so the sidebar turns into a
// flex track and swallows the window ("Basic is broken").

const TILE = 'session-tile:abc'

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['sessions', { placement: 'left' }],
    ['workspace', { placement: 'main', uncloseable: true }],
    ['files', { placement: 'right' }],
    ['review', { placement: 'right' }],
    ['terminal', { placement: 'bottom' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('preset adoption of a lone session tile', () => {
  it('lands the tile with the main zone, not the sessions sidebar', () => {
    $layoutTree.set(
      split('row', [
        group(['sessions']),
        group(['workspace']),
        group([TILE]),
        split('column', [group(['review', 'files']), group(['terminal'])])
      ])
    )

    const basic = split('row', [
      group(['sessions']),
      split('column', [group(['workspace']), group(['terminal'])], [3, 1]),
      split('row', [group(['review']), group(['files'])], [1, 1.2])
    ])

    applyTree(basic, 'basic')

    const tree = $layoutTree.get()!
    expect(findGroupOfPane(tree, TILE)?.panes).not.toContain('sessions')
    expect(findGroupOfPane(tree, TILE)?.panes).toContain('workspace')
  })
})

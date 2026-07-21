import { describe, expect, it } from 'vitest'

import { group, split, visiblePaneIds } from './model'

describe('visiblePaneIds', () => {
  it('includes the active pane from each side-by-side split group', () => {
    const tree = split('row', [
      group(['main'], { id: 'main-group' }),
      group(['session-tile:a'], { id: 'tile-group' }),
    ])

    expect(visiblePaneIds(tree)).toEqual(['main', 'session-tile:a'])
  })

  it('excludes an inactive tab stacked behind the active pane', () => {
    const tree = group(['main', 'session-tile:a'], {
      active: 'main',
      id: 'stack-group',
    })

    expect(visiblePaneIds(tree)).toEqual(['main'])
  })

  it('excludes panes hidden or dismissed by chrome', () => {
    const tree = split('row', [
      group(['main'], { id: 'main-group' }),
      group(['session-tile:a'], { id: 'tile-group' }),
    ])

    expect(visiblePaneIds(tree, new Set(['main', 'session-tile:a']))).toEqual([])
  })

  it('excludes the active pane in a minimized group', () => {
    const tree = group(['session-tile:a'], {
      id: 'tile-group',
      minimized: true,
    })

    expect(visiblePaneIds(tree)).toEqual([])
  })
})

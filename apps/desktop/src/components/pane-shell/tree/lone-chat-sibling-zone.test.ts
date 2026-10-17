import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { findGroupOfPane, group } from './model'
import {
  $dismissedPanes,
  $hiddenTreePanes,
  $layoutTree,
  moveTreePane,
  removeTreePane,
  tabStripVisibleForGroup
} from './store'

// The report: two session tabs in main, drag the second one to the right edge,
// and "the tabs disappear". Main was left a lone uncloseable workspace, which
// auto treated as "not a tab" — while the tile it had just split from kept its
// own strip. Pins the store's answer (the one ⌘⌥T toggles against) across the
// whole gesture and its undo.

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['workspace', { placement: 'main', uncloseable: true }],
    ['session-tile:b', { placement: 'main' }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }

  $layoutTree.set(group(['workspace', 'session-tile:b'], { active: 'session-tile:b', id: 'grp-main' }))
})

afterEach(() => disposers.splice(0).forEach(dispose => dispose()))

const zoneOf = (paneId: string) => findGroupOfPane($layoutTree.get()!, paneId)!

describe('dragging a session tab into its own zone', () => {
  it('leaves BOTH chat zones with a strip, and main keeps its own once the tile is gone', () => {
    expect(tabStripVisibleForGroup(zoneOf('workspace'))).toBe(true)

    moveTreePane('session-tile:b', { groupId: 'grp-main', pos: 'right' })

    expect(zoneOf('workspace').id).not.toBe(zoneOf('session-tile:b').id)
    expect(tabStripVisibleForGroup(zoneOf('workspace'))).toBe(true)
    expect(tabStripVisibleForGroup(zoneOf('session-tile:b'))).toBe(true)
    // Nothing was written to the zone to get there — it is still on auto.
    expect(zoneOf('workspace').tabStrip).toBeUndefined()

    removeTreePane('session-tile:b')

    // A lone workspace is the session switcher's home: tab and "+" stay (#89350).
    expect(tabStripVisibleForGroup(zoneOf('workspace'))).toBe(true)
  })
})

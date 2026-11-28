import { beforeEach, describe, expect, it } from 'vitest'

import {
  $titlebarAppActionsSide,
  setTitlebarAppActionsSide,
  TITLEBAR_APP_ACTIONS_DEFAULT,
  titlebarAppActionsClusterCounts
} from './titlebar-app-actions'

describe('titlebarAppActionsClusterCounts', () => {
  it('adds extras to the cluster they belong to', () => {
    for (const side of ['left', 'right'] as const) {
      const base = titlebarAppActionsClusterCounts(side)
      expect(titlebarAppActionsClusterCounts(side, 1, 2)).toEqual({ left: base.left + 1, right: base.right + 2 })
    }
  })

  it('releases the space of every tool Simple mode hides, on both sides', () => {
    // Sidebar toggle + what Simple keeps of the app actions; nothing fixed on the right.
    expect(titlebarAppActionsClusterCounts('right', 0, 0, 'simple')).toEqual({ left: 1, right: 2 })
    expect(titlebarAppActionsClusterCounts('left', 0, 0, 'simple')).toEqual({ left: 3, right: 0 })
  })
})

describe('$titlebarAppActionsSide', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setTitlebarAppActionsSide(TITLEBAR_APP_ACTIONS_DEFAULT)
  })

  it('persists left', () => {
    setTitlebarAppActionsSide('left')
    expect($titlebarAppActionsSide.get()).toBe('left')
    expect(window.localStorage.getItem('hermes.desktop.titlebarAppActions')).toBe('left')
  })
})

it('counts only the redesigned band tools that are not hidden', () => {
  const hidden = new Set(['settings', 'hud', 'flip-panes'] as const)

  // Back, Forward, sidebar | layout, right sidebar, Agent settings.
  expect(titlebarAppActionsClusterCounts('right', 0, 0, 'advanced', hidden)).toEqual({ left: 3, right: 3 })
  expect(titlebarAppActionsClusterCounts('left', 0, 0, 'advanced', hidden)).toEqual({ left: 4, right: 2 })
  expect(
    titlebarAppActionsClusterCounts('right', 0, 0, 'advanced', new Set([...hidden, 'history-back', 'history-forward', 'sidebar']))
  ).toEqual({ left: 0, right: 3 })
})

import { beforeEach, describe, expect, it } from 'vitest'

import {
  $rightSidebarWorkspaceLayouts,
  moveRightSidebarPanel,
  rightSidebarLayoutFor,
  selectRightSidebarPanel,
  setSidebarSplitRatio
} from './store'

describe('right sidebar workspace layouts', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $rightSidebarWorkspaceLayouts.set({})
  })

  it('keeps docked panel state isolated by workspace', () => {
    moveRightSidebarPanel('workspace:/one', 'git', 'secondary')
    setSidebarSplitRatio('workspace:/one', 0.7)

    expect(rightSidebarLayoutFor('workspace:/one')).toEqual({
      primary: { active: 'files', tabs: ['files', 'terminal'] },
      secondary: { active: 'git', tabs: ['git'] },
      splitRatio: 0.7
    })
    expect(rightSidebarLayoutFor('workspace:/two')).toEqual({
      primary: { active: 'files', tabs: ['files', 'git', 'terminal'] },
      secondary: { active: null, tabs: [] },
      splitRatio: 0.6
    })
  })

  it('supports moving any panel between islands and reordering it', () => {
    moveRightSidebarPanel('workspace:/one', 'terminal', 'secondary')
    moveRightSidebarPanel('workspace:/one', 'files', 'secondary', 0)

    expect(rightSidebarLayoutFor('workspace:/one')).toMatchObject({
      primary: { active: 'git', tabs: ['git'] },
      secondary: { active: 'files', tabs: ['files', 'terminal'] }
    })
  })

  it('does not reorder an icon dropped on itself', () => {
    moveRightSidebarPanel('workspace:/one', 'files', 'primary', 0)

    expect(rightSidebarLayoutFor('workspace:/one').primary.tabs).toEqual(['files', 'git', 'terminal'])
  })

  it('collapses the secondary island when its final panel returns', () => {
    moveRightSidebarPanel('workspace:/one', 'git', 'secondary')
    moveRightSidebarPanel('workspace:/one', 'git', 'primary', 1)

    expect(rightSidebarLayoutFor('workspace:/one')).toMatchObject({
      primary: { active: 'git', tabs: ['files', 'git', 'terminal'] },
      secondary: { active: null, tabs: [] }
    })
  })

  it('selects a panel inside the island that owns it', () => {
    moveRightSidebarPanel('workspace:/one', 'git', 'secondary')
    moveRightSidebarPanel('workspace:/one', 'terminal', 'secondary')
    selectRightSidebarPanel('workspace:/one', 'git')

    expect(rightSidebarLayoutFor('workspace:/one').secondary.active).toBe('git')
  })

  it('clamps the island split ratio', () => {
    setSidebarSplitRatio('workspace:/one', 0.95)
    expect(rightSidebarLayoutFor('workspace:/one').splitRatio).toBe(0.75)

    setSidebarSplitRatio('workspace:/one', 0.05)
    expect(rightSidebarLayoutFor('workspace:/one').splitRatio).toBe(0.25)
  })
})

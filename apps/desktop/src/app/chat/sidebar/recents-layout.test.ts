import { describe, expect, it } from 'vitest'

import { recentsFillsSidebar, recentsSectionClassNames, type RecentsFillInput } from './recents-layout'

const emptyRecents: RecentsFillInput = {
  inProject: false,
  showSessionSkeletons: false,
  sessionCount: 0,
  projectCount: 0,
  projectTreeLoading: false,
  worktreeGroupingActive: false,
  groupedSessionCount: 0
}

describe('recentsFillsSidebar', () => {
  it('does not fill when every recent session is pinned', () => {
    expect(recentsFillsSidebar(emptyRecents)).toBe(false)
  })

  it('fills while rows, a project tree, or a load are on screen', () => {
    expect(recentsFillsSidebar({ ...emptyRecents, sessionCount: 2 })).toBe(true)
    expect(recentsFillsSidebar({ ...emptyRecents, projectCount: 1 })).toBe(true)
    expect(recentsFillsSidebar({ ...emptyRecents, inProject: true })).toBe(true)
    expect(recentsFillsSidebar({ ...emptyRecents, showSessionSkeletons: true })).toBe(true)
    expect(recentsFillsSidebar({ ...emptyRecents, groupedSessionCount: 1 })).toBe(true)
    expect(
      recentsFillsSidebar({ ...emptyRecents, worktreeGroupingActive: true, projectTreeLoading: true })
    ).toBe(true)
  })
})

describe('recentsSectionClassNames', () => {
  it('shrinks an empty open list to its content', () => {
    const { rootClassName, contentClassName } = recentsSectionClassNames(false, false)

    expect(rootClassName).toContain('shrink-0')
    expect(rootClassName).not.toContain('flex-1')
    expect(rootClassName).not.toContain('min-h-32')
    expect(contentClassName).not.toContain('flex-1')
  })

  it('keeps a populated list as the flexible scroller', () => {
    const { rootClassName, contentClassName } = recentsSectionClassNames(true, false)

    expect(rootClassName).toContain('flex-1')
    expect(rootClassName).toContain('min-h-32')
    expect(contentClassName).toContain('flex-1')
    expect(contentClassName).toContain('overflow-y-auto')
  })
})

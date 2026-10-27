import { describe, expect, it } from 'vitest'

import { buildCommitChangelog } from './commit-changelog'

describe('changelog display copy', () => {
  it('overrides labels without changing grouping, ordering or authored subjects', () => {
    const commits = ['feat: A', 'fix: B', 'perf: C', 'refactor: D', 'misc: E'].map(summary => ({ summary }))
    const labels = { new: '新增', fixed: '修复', faster: '更快', improved: '改进', other: '其他改进' }
    const defaults = buildCommitChangelog(commits, { maxGroups: 5 })
    const localized = buildCommitChangelog(commits, { maxGroups: 5, labels })
    expect(localized).toEqual(defaults.map(group => ({ ...group, label: labels[group.id] })))
    const partial = buildCommitChangelog(commits, { maxGroups: 5, labels: { new: labels.new } })
    expect(partial).toEqual(defaults.map(group => ({ ...group, label: group.id === 'new' ? labels.new : group.label })))
  })

  it('uses caller fallback copy only when no user-facing commits remain', () => {
    const fallback = { label: '本次更新', item: '改进与修复' }

    for (const commits of [undefined, [], [{ summary: 'chore: internal' }]]) {
      expect(buildCommitChangelog(commits, { fallback })).toEqual([
        { id: 'other', label: fallback.label, items: [fallback.item] }
      ])
    }

    expect(buildCommitChangelog([{ summary: 'fix: real change' }], { fallback })[0].items).toEqual(['Real change'])
  })
})

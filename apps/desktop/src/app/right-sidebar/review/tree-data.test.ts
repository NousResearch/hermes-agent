import { describe, expect, it } from 'vitest'

import type { HermesReviewFile } from '@/global'

import {
  buildReviewSmartList,
  buildReviewTree,
  countAllNodes,
  flattenReviewRows,
  reconcileReviewOrder,
  smartReviewOrder
} from './tree-data'

const file = (path: string, added = 1, removed = 0): HermesReviewFile => ({
  path,
  added,
  removed,
  status: 'M',
  staged: false
})

describe('buildReviewTree', () => {
  it('nests files under their folders and sorts dirs before files', () => {
    const tree = buildReviewTree([file('src/a.ts'), file('readme.md'), file('src/b.ts')], false)

    expect(tree.map(n => n.name)).toEqual(['src', 'readme.md'])
    const src = tree[0]
    expect(src.isDir).toBe(true)
    expect(src.children?.map(n => n.name)).toEqual(['a.ts', 'b.ts'])
  })

  it('aggregates +/- onto directories', () => {
    const tree = buildReviewTree([file('src/a.ts', 5, 2), file('src/b.ts', 3, 1)], false)

    expect(tree[0].added).toBe(8)
    expect(tree[0].removed).toBe(3)
  })

  it('compacts single-child directory chains', () => {
    const tree = buildReviewTree([file('a/b/c/deep.ts')], true)

    expect(tree[0].name).toBe('a/b/c')
    expect(tree[0].children?.[0].name).toBe('deep.ts')
  })

  it('does not compact when a directory has multiple children', () => {
    const tree = buildReviewTree([file('a/b/one.ts'), file('a/other.ts')], true)

    expect(tree[0].name).toBe('a')
    expect(tree[0].children?.map(n => n.name).sort()).toEqual(['b', 'other.ts'])
  })
})

describe('countAllNodes', () => {
  it('counts every node including descendants', () => {
    const tree = buildReviewTree([file('src/a.ts'), file('src/b.ts'), file('readme.md')], false)

    // src + its two files + readme.
    expect(countAllNodes(tree)).toBe(4)
  })
})

describe('flattenReviewRows', () => {
  it('includes a directory children only while it is open', () => {
    const tree = buildReviewTree([file('src/a.ts'), file('readme.md')], false)

    const collapsed = flattenReviewRows(tree, () => false)
    expect(collapsed.map(r => r.node.id)).toEqual(['src', 'readme.md'])

    const expanded = flattenReviewRows(tree, () => true)
    expect(expanded.map(r => r.node.id)).toEqual(['src', 'src/a.ts', 'readme.md'])
    expect(expanded[1].depth).toBe(1)
  })

  it('recurses into nested open directories with increasing depth', () => {
    const tree = buildReviewTree([file('a/b/c.ts')], false)

    const rows = flattenReviewRows(tree, () => true)

    expect(rows.map(r => r.node.id)).toEqual(['a', 'a/b', 'a/b/c.ts'])
    expect(rows.map(r => r.depth)).toEqual([0, 1, 2])
  })

  it('stops descending into a collapsed nested directory', () => {
    const tree = buildReviewTree([file('a/b/c.ts')], false)

    const rows = flattenReviewRows(tree, id => id !== 'a/b')

    expect(rows.map(r => r.node.id)).toEqual(['a', 'a/b'])
  })
})

describe('buildReviewSmartList', () => {
  it('orders change-explaining files by churn before supporting files, which are muted', () => {
    const rows = buildReviewSmartList([
      file('src/store/review.test.ts', 40, 10),
      file('package-lock.json', 900, 900),
      file('src/store/review.ts', 30, 5),
      file('src/app/pane.tsx', 80, 20)
    ])

    // Source files first, biggest churn first; supporting files trail (their
    // own group also churn-ordered), muted.
    expect(rows.map(r => r.id)).toEqual([
      'src/app/pane.tsx',
      'src/store/review.ts',
      'package-lock.json',
      'src/store/review.test.ts'
    ])
    expect(rows.map(r => Boolean(r.muted))).toEqual([false, false, true, true])
  })

  it('classifies supporting paths by directory segment and filename shape', () => {
    const rows = buildReviewSmartList([
      file('tests/agent/test_loop.py', 5, 0),
      file('agent/__fixtures__/sample.json', 5, 0),
      file('agent/loop.py', 1, 0)
    ])

    expect(rows[0].id).toBe('agent/loop.py')
    expect(rows.filter(r => r.muted).map(r => r.id).sort()).toEqual([
      'agent/__fixtures__/sample.json',
      'tests/agent/test_loop.py'
    ])
  })
})

describe('stable smart order while reviewing', () => {
  it('keeps the shown order when the diff changes: new files appended, removed files dropped', () => {
    const before = [file('src/a.ts', 50, 0), file('src/b.ts', 10, 0), file('src/gone.ts', 5, 0)]
    const shown = smartReviewOrder(before)
    expect(shown).toEqual(['src/a.ts', 'src/b.ts', 'src/gone.ts'])

    // b.ts grew past a.ts, gone.ts was reverted, new.ts appeared with top churn.
    const after = [file('src/a.ts', 50, 0), file('src/b.ts', 400, 0), file('src/new.ts', 900, 0)]

    expect(smartReviewOrder(after)).toEqual(['src/new.ts', 'src/b.ts', 'src/a.ts'])
    expect(buildReviewSmartList(after, shown).map(r => r.id)).toEqual(['src/a.ts', 'src/b.ts', 'src/new.ts'])
  })

  it('adopts the fresh order once applied (no pinned order)', () => {
    const files = [file('src/a.ts', 50, 0), file('src/b.ts', 400, 0)]
    const fresh = smartReviewOrder(files)

    expect(reconcileReviewOrder(['src/a.ts', 'src/b.ts'], fresh)).toEqual(['src/a.ts', 'src/b.ts'])
    expect(buildReviewSmartList(files, fresh).map(r => r.id)).toEqual(fresh)
    expect(buildReviewSmartList(files).map(r => r.id)).toEqual(['src/b.ts', 'src/a.ts'])
  })
})

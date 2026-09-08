import { describe, expect, it } from 'vitest'

import type { HermesReviewFile } from '@/global'

import { buildReviewTree, countAllNodes, flattenReviewRows } from './tree-data'

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

  it('counts one per top-level leaf', () => {
    const tree = buildReviewTree([file('a.ts'), file('b.ts')], false)

    expect(countAllNodes(tree)).toBe(2)
  })

  it('counts a single deep folder holding thousands of files as heavy', () => {
    const files = Array.from({ length: 40_000 }, (_, i) => file(`publish/lib-${i}.so`))
    const tree = buildReviewTree(files)

    // One top-level node, but the total is what matters for virtualization.
    expect(tree.length).toBe(1)
    expect(countAllNodes(tree)).toBe(40_001)
  })
})

describe('flattenReviewRows', () => {
  it('flattens top-level leaves in order', () => {
    const tree = buildReviewTree([file('b.ts'), file('a.ts')], false)

    const rows = flattenReviewRows(tree, () => true)

    expect(rows.map(r => r.node.id)).toEqual(['a.ts', 'b.ts'])
    expect(rows.map(r => r.depth)).toEqual([0, 0])
  })

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

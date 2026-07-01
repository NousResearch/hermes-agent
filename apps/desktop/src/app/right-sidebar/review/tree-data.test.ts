import { describe, expect, it } from 'vitest'

import type { HermesReviewFile } from '@/global'

import { buildReviewTree } from './tree-data'

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

import { describe, expect, it, vi } from 'vitest'

import { memoArgs } from './memo-args'

describe('memoArgs', () => {
  it('derives once per distinct argument identity, across callers', () => {
    const derive = vi.fn((rows: number[], kind: string) => rows.map(row => `${kind}${row}`))
    const memo = memoArgs(derive)
    const feed = [1, 2, 3]

    const first = memo(feed, 'a')
    expect(memo(feed, 'a')).toBe(first)
    expect(derive).toHaveBeenCalledOnce()

    // An equal but distinct list is new input: identity, not deep equality, is the key.
    memo([1, 2, 3], 'a')
    expect(derive).toHaveBeenCalledTimes(2)
  })

  it('keeps a few recent inputs so alternating between them does not recompute', () => {
    const derive = vi.fn((rows: number[]) => rows.length)
    const memo = memoArgs(derive, 2)
    const skills = [1]
    const plugins = [1, 2]

    memo(skills)
    memo(plugins)
    memo(skills)
    memo(plugins)
    expect(derive).toHaveBeenCalledTimes(2)

    // A third input evicts the least recent one.
    memo([3])
    memo(skills)
    expect(derive).toHaveBeenCalledTimes(4)
  })
})

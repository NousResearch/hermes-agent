import { describe, expect, it } from 'vitest'
import { canonicalWaves } from './wave-preview'

describe('managed rollout canonical draft', () => {
  it('keeps the stable canary first and excludes it from successor rows', () => {
    expect(canonicalWaves({ mode: 'manual', concurrency: 1, canaryInstallId: 'b', selectedInstallIds: ['a', 'b', 'c'] })).toEqual([['b'], ['a', 'c']])
  })
})

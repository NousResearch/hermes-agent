import { describe, expect, it } from 'vitest'

import { parseSelfImprovementReview } from './system-message'

describe('parseSelfImprovementReview', () => {
  it('extracts the review text without the emoji prefix', () => {
    expect(parseSelfImprovementReview('💾 Self-improvement review: Patched one skill.')).toBe(
      'Self-improvement review: Patched one skill.'
    )
  })

  it('ignores unrelated system messages', () => {
    expect(parseSelfImprovementReview('model → openai/gpt-5')).toBeNull()
  })
})

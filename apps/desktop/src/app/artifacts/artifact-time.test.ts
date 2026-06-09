import { describe, expect, it } from 'vitest'

import { artifactTimestampToDate } from './artifact-time'

describe('artifactTimestampToDate', () => {
  it('treats session timestamps from the backend as epoch seconds', () => {
    const seconds = 1_765_230_300

    expect(artifactTimestampToDate(seconds).getTime()).toBe(seconds * 1000)
    expect(artifactTimestampToDate(seconds).getUTCFullYear()).toBeGreaterThan(2024)
  })

  it('preserves millisecond timestamps from Date.now fallbacks', () => {
    const millis = 1_765_230_300_000

    expect(artifactTimestampToDate(millis).getTime()).toBe(millis)
  })
})

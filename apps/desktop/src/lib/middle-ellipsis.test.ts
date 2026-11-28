import { describe, expect, it } from 'vitest'

import { middleEllipsis } from './middle-ellipsis'

// One unit per character: the width is the length.
const measure = (value: string) => value.length

describe('middleEllipsis', () => {
  it('leaves a fitting text alone', () => {
    expect(middleEllipsis('~/code/portal', 20, measure)).toBe('~/code/portal')
  })

  it('cuts the middle, keeps both ends and fills the width exactly', () => {
    const shown = middleEllipsis('~/.cache/omp-tmp/claude-1000/apps/desktop/portal', 24, measure)

    expect(shown).toHaveLength(24)
    expect(shown.startsWith('~/.cache')).toBe(true)
    expect(shown.endsWith('desktop/portal')).toBe(true)
    expect(shown).toContain('…')
  })

  it('degrades to the ellipsis alone when nothing else fits', () => {
    expect(middleEllipsis('~/a/b/c', 1, measure)).toBe('…')
  })
})

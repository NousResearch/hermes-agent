import { describe, expect, it } from 'vitest'

import { truncateText } from './job-state'

describe('truncateText', () => {
  it('truncates on Unicode code-point boundaries', () => {
    const result = truncateText('😀😀😀😀', 3)

    expect(result).toBe('😀😀😀…')
    expect(Array.from(result)).toEqual(['😀', '😀', '😀', '…'])
  })

  it('does not add an ellipsis at the limit', () => {
    expect(truncateText('😀😀😀', 3)).toBe('😀😀😀')
  })
})

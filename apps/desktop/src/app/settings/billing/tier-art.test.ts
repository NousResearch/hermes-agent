import { describe, expect, it } from 'vitest'

import { resolveTierArt } from './tier-art'

describe('resolveTierArt', () => {
  it('keys art by lowercase tier name, case-insensitively', () => {
    for (const name of ['Free', 'starter', 'Plus', 'SUPER', 'ultra']) {
      expect(resolveTierArt(name)).not.toBeNull()
    }
  })

  it('returns null for unknown or missing names so the card renders text-only', () => {
    expect(resolveTierArt('Mystery')).toBeNull()
    expect(resolveTierArt('')).toBeNull()
    expect(resolveTierArt(null)).toBeNull()
    expect(resolveTierArt(undefined)).toBeNull()
  })
})

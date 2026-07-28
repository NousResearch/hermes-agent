import { describe, expect, it } from 'vitest'

import { resolveTierArt } from './tier-art'

describe('resolveTierArt', () => {
  it('keys art by lowercase tier name, case-insensitively', () => {
    for (const name of ['Free', 'starter', 'Plus', 'SUPER', 'ultra']) {
      expect(resolveTierArt(name)).not.toBeNull()
    }
  })

  it('maps each named tier to its NAS blend mode', () => {
    expect(resolveTierArt('free')?.blend).toBe('screen')
    expect(resolveTierArt('plus')?.blend).toBe('screen')
    expect(resolveTierArt('super')?.blend).toBe('lighten')
    expect(resolveTierArt('ultra')?.blend).toBe('normal')
  })

  it('returns null for unknown or missing names so the card renders text-only', () => {
    expect(resolveTierArt('Mystery')).toBeNull()
    expect(resolveTierArt('')).toBeNull()
    expect(resolveTierArt(null)).toBeNull()
    expect(resolveTierArt(undefined)).toBeNull()
  })
})

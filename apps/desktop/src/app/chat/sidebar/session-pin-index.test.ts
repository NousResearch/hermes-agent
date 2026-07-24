import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { buildPinnedSessionIndex } from './session-pin-index'

const messaging = (id: string, profile: string): SessionInfo =>
  ({ id, profile, source: 'telegram' }) as SessionInfo

describe('buildPinnedSessionIndex', () => {
  it('resolves only Messaging pins owned by the concrete profile', () => {
    const index = buildPinnedSessionIndex(
      'alma',
      false,
      [],
      [],
      [messaging('alma-telegram', 'alma'), messaging('aegis-telegram', 'aegis_h-01')]
    )

    expect(index.has('alma-telegram')).toBe(true)
    expect(index.has('aegis-telegram')).toBe(false)
  })

  it('resolves Messaging pins from every profile in All Profiles', () => {
    const index = buildPinnedSessionIndex(
      '__all__',
      true,
      [],
      [],
      [messaging('alma-telegram', 'alma'), messaging('aegis-telegram', 'aegis_h-01')]
    )

    expect([...index.keys()]).toEqual(['alma-telegram', 'aegis-telegram'])
  })
})

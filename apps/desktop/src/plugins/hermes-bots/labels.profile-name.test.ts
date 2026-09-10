import { describe, expect, it } from 'vitest'

import { botProfileIdentity, slugifyProfileName } from './labels'

describe('profile-name identity', () => {
  it('pins the documented ASCII-lookalike collision', () => {
    const encoded = 'u5c0f-u52a9-u624b'

    expect(slugifyProfileName('小助手')).toBe(encoded)
    expect(slugifyProfileName(encoded)).toBe(encoded)
  })

  it('preserves case-only display identity', () => {
    expect(botProfileIdentity('Test', '')).toEqual({ slug: 'test', title: 'Test' })
    expect(botProfileIdentity('test', '')).toEqual({ slug: 'test', title: '' })
  })

  it('stops before a Unicode token that would cross the profile limit', () => {
    const prefix = 'a'.repeat(60)
    const slug = slugifyProfileName(`${prefix}助`)

    expect(slug).toBe(prefix)
    expect(slug).toHaveLength(60)
  })

  it('rejects symbols-only identity', () => {
    expect(botProfileIdentity('🤖', '')).toEqual({ slug: '', title: '🤖' })
  })
})

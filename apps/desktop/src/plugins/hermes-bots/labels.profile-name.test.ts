import { describe, expect, it } from 'vitest'

import { botProfileIdentity, slugifyProfileName } from './labels'

describe('profile-name identity', () => {
  it('derives a stable ASCII profile id from a CJK name and keeps the name as the title', () => {
    expect(slugifyProfileName('小助手')).toBe('u5c0f-u52a9-u624b')
    expect(slugifyProfileName('test机器人')).toBe('test-u673a-u5668-u4eba')
    // The encoded form is its own fixed point, and ASCII input slugs as before.
    expect(slugifyProfileName('u5c0f-u52a9-u624b')).toBe('u5c0f-u52a9-u624b')
    expect(slugifyProfileName('Inbox Triage')).toBe('inbox-triage')

    expect(botProfileIdentity('小助手', '')).toEqual({ slug: 'u5c0f-u52a9-u624b', title: '小助手' })
    expect(botProfileIdentity('小助手', 'Helper')).toEqual({ slug: 'u5c0f-u52a9-u624b', title: 'Helper' })
    expect(botProfileIdentity('Test', '')).toEqual({ slug: 'test', title: '' })
    // Symbols carry no letters or digits: still no id, Create stays disabled.
    expect(botProfileIdentity('🤖', '')).toEqual({ slug: '', title: '🤖' })
  })

  it('folds accented Latin to its base letters and keeps the accented name as the title', () => {
    // NFKD alone tokenised the detached accent: 'Résumé' → 're-sume'.
    expect(slugifyProfileName('Café Résumé')).toBe('cafe-resume')
    expect(botProfileIdentity('Café', '')).toEqual({ slug: 'cafe', title: 'Café' })
  })

  it('keeps one token per NFC code point and never splits a token at the 64-char cap', () => {
    // Hangul must not decompose to jamo (3 syllables → 6 tokens) nor ガ collapse onto カ.
    expect(slugifyProfileName('한글')).toBe('ud55c-uae00')
    expect(slugifyProfileName('ガ')).not.toBe(slugifyProfileName('カ'))

    const long = slugifyProfileName('小助手'.repeat(7))
    expect(long).toBe('u5c0f-u52a9-u624b-u5c0f-u52a9-u624b-u5c0f-u52a9-u624b-u5c0f')
    // Every id still satisfies hermes_cli.profiles._PROFILE_ID_RE.
    expect(long).toMatch(/^[a-z0-9][a-z0-9_-]{0,63}$/)
  })
})

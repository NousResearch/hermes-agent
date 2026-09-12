import { describe, expect, it } from 'vitest'

import { DEFAULT_LOCALE, osPreferredLocale, resolveInitialLocale } from './languages'

describe('Ukrainian locale and OS language precedence', () => {
  it.each(['uk', 'uk-UA', 'uk_UA', 'uk-CA'])('recognizes the Ukrainian OS tag %s', tag => {
    expect(osPreferredLocale(tag)).toBe('uk')
    expect(resolveInitialLocale(undefined, tag)).toBe('uk')
  })

  it('keeps an explicit English choice over a Ukrainian OS language', () => {
    expect(resolveInitialLocale('en', 'uk-UA')).toBe('en')
  })

  it('keeps an explicit Ukrainian choice over another OS language', () => {
    expect(resolveInitialLocale(' uk_UA ', 'ru-RU')).toBe('uk')
  })

  it('infers Ukrainian only when the saved choice is unsupported or absent', () => {
    expect(resolveInitialLocale('unsupported', 'uk-UA')).toBe('uk')
    expect(resolveInitialLocale(null, 'uk-UA')).toBe('uk')
  })

  it('retains the English fallback when neither source is supported', () => {
    expect(resolveInitialLocale(undefined, 'unsupported')).toBe(DEFAULT_LOCALE)
    expect(resolveInitialLocale(undefined, undefined)).toBe(DEFAULT_LOCALE)
  })
})

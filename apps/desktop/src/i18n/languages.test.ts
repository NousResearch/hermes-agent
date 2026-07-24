import { describe, expect, it } from 'vitest'

import {
  DEFAULT_LOCALE,
  isLocale,
  isSupportedLocaleValue,
  localeConfigValue,
  matchPreferredLocale,
  normalizeLocale,
  resolveInitialLocale
} from './languages'

describe('desktop i18n languages', () => {
  it('normalizes supported locale aliases', () => {
    expect(normalizeLocale('en')).toBe('en')
    expect(normalizeLocale('EN-US')).toBe('en')
    expect(normalizeLocale('zh')).toBe('zh')
    expect(normalizeLocale('zh-CN')).toBe('zh')
    expect(normalizeLocale('zh-Hans')).toBe('zh')
    expect(normalizeLocale(' zh_hans_cn ')).toBe('zh')
    expect(normalizeLocale('zh-Hant')).toBe('zh-hant')
    expect(normalizeLocale('zh-TW')).toBe('zh-hant')
    expect(normalizeLocale('zh_HK')).toBe('zh-hant')
    expect(normalizeLocale('ja')).toBe('ja')
    expect(normalizeLocale('ja-JP')).toBe('ja')
  })

  it('falls back to English for empty or unsupported values', () => {
    expect(normalizeLocale(null)).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('')).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('de')).toBe(DEFAULT_LOCALE)
  })

  it('distinguishes exact locale ids from supported config aliases', () => {
    expect(isSupportedLocaleValue('zh-CN')).toBe(true)
    expect(isSupportedLocaleValue('zh-TW')).toBe(true)
    expect(isSupportedLocaleValue('ja-JP')).toBe(true)
    expect(isSupportedLocaleValue('de')).toBe(false)
    expect(isLocale('zh-CN')).toBe(false)
    expect(isLocale('zh')).toBe(true)
    expect(isLocale('zh-hant')).toBe(true)
    expect(isLocale('ja')).toBe(true)
  })

  it('returns the persisted config value for supported locales', () => {
    expect(localeConfigValue('en')).toBe('en')
    expect(localeConfigValue('zh')).toBe('zh')
    expect(localeConfigValue('zh-hant')).toBe('zh-hant')
    expect(localeConfigValue('ja')).toBe('ja')
  })

  it('matches the first supported locale from the platform preference list', () => {
    expect(matchPreferredLocale(['ja-JP', 'en-US'])).toBe('ja')
    expect(matchPreferredLocale(['zh-TW', 'zh-CN'])).toBe('zh-hant')
    // Unsupported entries are skipped in favor of the first supported one.
    expect(matchPreferredLocale(['de-DE', 'fr-FR', 'zh-Hans'])).toBe('zh')
  })

  it('returns null when no preferred locale is supported', () => {
    expect(matchPreferredLocale(['de-DE', 'fr-FR'])).toBeNull()
    expect(matchPreferredLocale([])).toBeNull()
    expect(matchPreferredLocale(null)).toBeNull()
    expect(matchPreferredLocale(undefined)).toBeNull()
  })

  it('prefers an explicit configured language over the system locale', () => {
    expect(resolveInitialLocale('ja-JP', ['zh-CN'])).toBe('ja')
    // An unsupported but explicit configured value still wins and falls back to
    // English, so a manual selection is never overridden by the system locale.
    expect(resolveInitialLocale('de', ['zh-CN'])).toBe(DEFAULT_LOCALE)
  })

  it('falls back to the system locale only when no language is configured', () => {
    expect(resolveInitialLocale(undefined, ['zh-CN', 'en-US'])).toBe('zh')
    expect(resolveInitialLocale('', ['ja-JP'])).toBe('ja')
    expect(resolveInitialLocale('   ', ['zh-TW'])).toBe('zh-hant')
  })

  it('falls back to English when neither config nor system locale is supported', () => {
    expect(resolveInitialLocale(undefined, ['de-DE'])).toBe(DEFAULT_LOCALE)
    expect(resolveInitialLocale(null, [])).toBe(DEFAULT_LOCALE)
  })
})

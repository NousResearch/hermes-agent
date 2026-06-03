import { describe, expect, it } from 'vitest'

import {
  DEFAULT_DESKTOP_LANGUAGE,
  desktopLanguageConfigValue,
  isDesktopLanguage,
  isSupportedDesktopLanguageValue,
  normalizeDesktopLanguage
} from './languages'

describe('desktop i18n languages', () => {
  it('normalizes supported desktop language aliases', () => {
    expect(normalizeDesktopLanguage('en')).toBe('en')
    expect(normalizeDesktopLanguage('EN-US')).toBe('en')
    expect(normalizeDesktopLanguage('zh')).toBe('zh')
    expect(normalizeDesktopLanguage('zh-CN')).toBe('zh')
    expect(normalizeDesktopLanguage('zh-Hans')).toBe('zh')
    expect(normalizeDesktopLanguage(' zh_hans_cn ')).toBe('zh')
  })

  it('falls back to English for empty or unsupported values', () => {
    expect(normalizeDesktopLanguage(null)).toBe(DEFAULT_DESKTOP_LANGUAGE)
    expect(normalizeDesktopLanguage('')).toBe(DEFAULT_DESKTOP_LANGUAGE)
    expect(normalizeDesktopLanguage('ja')).toBe(DEFAULT_DESKTOP_LANGUAGE)
  })

  it('distinguishes normalized display fallback from supported config values', () => {
    expect(isSupportedDesktopLanguageValue('zh-CN')).toBe(true)
    expect(isSupportedDesktopLanguageValue('ja')).toBe(false)
    expect(isDesktopLanguage('zh-CN')).toBe(false)
    expect(isDesktopLanguage('zh')).toBe(true)
  })

  it('returns the persisted config value for supported desktop languages', () => {
    expect(desktopLanguageConfigValue('en')).toBe('en')
    expect(desktopLanguageConfigValue('zh')).toBe('zh')
  })
})

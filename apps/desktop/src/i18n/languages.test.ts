import { describe, expect, it } from 'vitest'

import { DEFAULT_LOCALE, isLocale, isSupportedLocaleValue, localeConfigValue, normalizeLocale } from './languages'

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
    expect(normalizeLocale('ar')).toBe('ar')
    expect(normalizeLocale('AR-SA')).toBe('ar')
    expect(normalizeLocale(' ar_eg ')).toBe('ar')
    expect(normalizeLocale('ru')).toBe('ru')
    expect(normalizeLocale('RU-RU')).toBe('ru')
    expect(normalizeLocale(' ru_ru ')).toBe('ru')
    expect(normalizeLocale('Русский')).toBe('ru')
    expect(normalizeLocale('id')).toBe('id')
    expect(normalizeLocale('ID-ID')).toBe('id')
    expect(normalizeLocale(' bahasa ')).toBe('id')
    expect(normalizeLocale('bahasa-indonesia')).toBe('id')
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
    expect(isSupportedLocaleValue('ru-RU')).toBe(true)
    expect(isSupportedLocaleValue('id-ID')).toBe(true)
    expect(isSupportedLocaleValue('de')).toBe(false)
    expect(isLocale('zh-CN')).toBe(false)
    expect(isLocale('zh')).toBe(true)
    expect(isLocale('zh-hant')).toBe(true)
    expect(isLocale('ja')).toBe(true)
    expect(isLocale('ar')).toBe(true)
    expect(isLocale('ru')).toBe(true)
    expect(isLocale('id')).toBe(true)
  })

  it('returns the persisted config value for supported locales', () => {
    expect(localeConfigValue('en')).toBe('en')
    expect(localeConfigValue('zh')).toBe('zh')
    expect(localeConfigValue('zh-hant')).toBe('zh-hant')
    expect(localeConfigValue('ja')).toBe('ja')
    expect(localeConfigValue('ar')).toBe('ar')
    expect(localeConfigValue('ru')).toBe('ru')
    expect(localeConfigValue('id')).toBe('id')
  })
})

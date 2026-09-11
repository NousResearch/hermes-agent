import { describe, expect, it } from 'vitest'

import {
  DEFAULT_LOCALE,
  isLocale,
  isSupportedLocaleValue,
  localeConfigValue,
  normalizeLocale,
  resolveInitialLocale
} from './languages'

describe('desktop i18n languages', () => {
  it('prefers a saved choice over the OS locale and falls back to English when neither is supported', () => {
    expect(resolveInitialLocale('en', 'ja-JP')).toBe('en')
    expect(resolveInitialLocale('zh-TW', 'ja-JP')).toBe('zh-hant')
    expect(resolveInitialLocale(undefined, 'ja-JP')).toBe('ja')
    expect(resolveInitialLocale(null, 'ru-UA')).toBe('ru')
    expect(resolveInitialLocale('de', 'ar_MA')).toBe('ar')
    expect(resolveInitialLocale(undefined, 'zh-Hant-HK')).toBe('zh-hant')
    expect(resolveInitialLocale(undefined, 'zh-Hans-CN')).toBe('zh')
    expect(resolveInitialLocale(undefined, 'pt-BR')).toBe(DEFAULT_LOCALE)
    expect(resolveInitialLocale(null, undefined)).toBe(DEFAULT_LOCALE)
    expect(resolveInitialLocale('de', '')).toBe(DEFAULT_LOCALE)
  })

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
    expect(isSupportedLocaleValue('de')).toBe(false)
    expect(isLocale('zh-CN')).toBe(false)
    expect(isLocale('zh')).toBe(true)
    expect(isLocale('zh-hant')).toBe(true)
    expect(isLocale('ja')).toBe(true)
    expect(isLocale('ar')).toBe(true)
    expect(isLocale('ru')).toBe(true)
  })

  it('returns the persisted config value for supported locales', () => {
    expect(localeConfigValue('en')).toBe('en')
    expect(localeConfigValue('zh')).toBe('zh')
    expect(localeConfigValue('zh-hant')).toBe('zh-hant')
    expect(localeConfigValue('ja')).toBe('ja')
    expect(localeConfigValue('ar')).toBe('ar')
    expect(localeConfigValue('ru')).toBe('ru')
  })
})

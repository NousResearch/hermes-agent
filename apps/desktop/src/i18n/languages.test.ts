import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import {
  DEFAULT_LOCALE,
  isLocale,
  isSupportedLocaleValue,
  localeConfigValue,
  localeDirection,
  normalizeLocale
} from './languages'

function missingTranslationPaths(source: unknown, target: unknown, prefix = ''): string[] {
  if (typeof source !== 'object' || source === null || Array.isArray(source)) {
    return []
  }

  const targetRecord =
    typeof target === 'object' && target !== null && !Array.isArray(target)
      ? (target as Record<string, unknown>)
      : {}

  return Object.entries(source as Record<string, unknown>).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key

    if (!(key in targetRecord)) {
      return [path]
    }

    return missingTranslationPaths(value, targetRecord[key], path)
  })
}

describe('desktop i18n languages', () => {
  it('normalizes supported locale aliases', () => {
    expect(normalizeLocale('en')).toBe('en')
    expect(normalizeLocale('EN-US')).toBe('en')
    expect(normalizeLocale('ar')).toBe('ar')
    expect(normalizeLocale('Arabic')).toBe('ar')
    expect(normalizeLocale('العربية')).toBe('ar')
    expect(normalizeLocale('ar-EG')).toBe('ar')
    expect(normalizeLocale('ar_SA')).toBe('ar')
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
    expect(isSupportedLocaleValue('ar-AE')).toBe(true)
    expect(isSupportedLocaleValue('zh-TW')).toBe(true)
    expect(isSupportedLocaleValue('ja-JP')).toBe(true)
    expect(isSupportedLocaleValue('de')).toBe(false)
    expect(isLocale('zh-CN')).toBe(false)
    expect(isLocale('ar')).toBe(true)
    expect(isLocale('zh')).toBe(true)
    expect(isLocale('zh-hant')).toBe(true)
    expect(isLocale('ja')).toBe(true)
  })

  it('returns the persisted config value for supported locales', () => {
    expect(localeConfigValue('en')).toBe('en')
    expect(localeConfigValue('ar')).toBe('ar')
    expect(localeConfigValue('zh')).toBe('zh')
    expect(localeConfigValue('zh-hant')).toBe('zh-hant')
    expect(localeConfigValue('ja')).toBe('ja')
  })

  it('returns the writing direction for each locale', () => {
    expect(localeDirection('ar')).toBe('rtl')
    expect(localeDirection('en')).toBe('ltr')
    expect(localeDirection('zh')).toBe('ltr')
  })

  it('keeps provider and model brands in their original script in Arabic UI', () => {
    const arabic = TRANSLATIONS.ar

    const maps = [
      arabic.settings.memoryProvider!.providerNames,
      arabic.settings.providers.providerNames,
      arabic.settings.toolsets.providerNames,
      arabic.onboarding.providerNames
    ]

    for (const map of maps) {
      for (const label of Object.values(map ?? {})) {
        expect(label).not.toMatch(/\p{Script=Arabic}/u)
      }
    }

    expect(arabic.onboarding.openRouterName).toBe('OpenRouter')
    expect(arabic.commandCenter.generatePet.openAi).toBe('OpenAI')
  })

  it('provides an Arabic value for every English translation path', () => {
    expect(missingTranslationPaths(TRANSLATIONS.en, TRANSLATIONS.ar)).toEqual([])
  })
})

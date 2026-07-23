import { describe, expect, it } from 'vitest'

import { normalizeLocale } from './i18n'

describe('normalizeLocale', () => {
  it('maps Traditional Chinese system locales to zh-Hant', () => {
    expect(normalizeLocale('zh-TW')).toBe('zh-Hant')
    expect(normalizeLocale('zh-HK')).toBe('zh-Hant')
    expect(normalizeLocale('zh-MO')).toBe('zh-Hant')
  })

  it('maps other Chinese system locales to zh-CN', () => {
    expect(normalizeLocale('zh-CN')).toBe('zh-CN')
    expect(normalizeLocale('zh')).toBe('zh-CN')
    expect(normalizeLocale('zh-SG')).toBe('zh-CN')
  })

  it('maps supported non-Chinese language families', () => {
    expect(normalizeLocale('ja-JP')).toBe('ja')
    expect(normalizeLocale('ko-KR')).toBe('ko')
    expect(normalizeLocale('pt-BR')).toBe('pt-BR')
  })

  it('falls back to English for unsupported languages', () => {
    expect(normalizeLocale('it-IT')).toBe('en')
  })
})

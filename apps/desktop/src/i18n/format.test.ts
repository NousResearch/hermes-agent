import { beforeEach, describe, expect, it, vi } from 'vitest'

import { catalogs } from './catalogs'
import {
  clearMissingTranslationDiagnostics,
  createTranslator,
  getMissingTranslationDiagnostics,
  interpolate,
  translate,
  type TranslationCatalogs
} from './format'

describe('desktop i18n formatting', () => {
  beforeEach(() => {
    clearMissingTranslationDiagnostics()
    vi.restoreAllMocks()
  })

  it('translates catalog keys for the active language', () => {
    expect(translate(catalogs, 'en', 'common.save')).toBe('Save')
    expect(translate(catalogs, 'zh', 'common.save')).toBe('保存')
  })

  it('interpolates named values and leaves unknown placeholders intact', () => {
    expect(interpolate('Open {name} in {count} tabs: {missing}', { count: 2, name: 'Hermes' })).toBe(
      'Open Hermes in 2 tabs: {missing}'
    )
  })

  it('falls back to English when the active language is missing a key', () => {
    const partialCatalogs: TranslationCatalogs = {
      en: { greeting: 'Hello {name}' },
      zh: {}
    }

    expect(translate(partialCatalogs, 'zh', 'greeting', { name: 'Kai' })).toBe('Hello Kai')
    expect(getMissingTranslationDiagnostics()).toEqual([
      { key: 'greeting', language: 'zh', reason: 'missing-active' }
    ])
  })

  it('records missing default keys and returns the key as the final fallback', () => {
    const partialCatalogs: TranslationCatalogs = {
      en: {},
      zh: {}
    }

    expect(translate(partialCatalogs, 'zh', 'missing.key')).toBe('missing.key')
    expect(getMissingTranslationDiagnostics()).toEqual([
      { key: 'missing.key', language: 'zh', reason: 'missing-active' },
      { key: 'missing.key', language: 'en', reason: 'missing-default' }
    ])
  })

  it('creates stable translators for a selected language', () => {
    const t = createTranslator(catalogs, 'zh')

    expect(t('common.cancel')).toBe('取消')
  })
})

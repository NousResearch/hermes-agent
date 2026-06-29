import { describe, expect, it, vi } from 'vitest'

// @tabler/icons-react is a native dep that may be absent in partial installs.
// Stub it so the transitive import chain (en.ts → settings/constants → icons)
// doesn't block i18n unit tests. Each named export becomes a no-op component.
vi.mock('@tabler/icons-react', () => ({
  IconActivity: () => null,
  IconAlertCircle: () => null,
}))

import { TRANSLATIONS } from './catalog'
import { setRuntimeI18nLocale, translateNow } from './runtime'

/**
 * Tests for the "Copy File Content" context-menu action added to the file
 * browser and git-review file trees. Verifies the i18n keys exist in every
 * locale and resolve at runtime.
 */
describe('fileMenu.copyContent i18n', () => {
  const locales = ['en', 'zh', 'zh-hant', 'ja'] as const

  it('has copyContent and contentCopied keys in every locale', () => {
    for (const locale of locales) {
      const fm = TRANSLATIONS[locale].fileMenu
      expect(fm.copyContent, `${locale}.fileMenu.copyContent`).toBeTypeOf('string')
      expect(fm.copyContent.length, `${locale}.fileMenu.copyContent non-empty`).toBeGreaterThan(0)
      expect(fm.contentCopied, `${locale}.fileMenu.contentCopied`).toBeTypeOf('string')
      expect(fm.contentCopied.length, `${locale}.fileMenu.contentCopied non-empty`).toBeGreaterThan(0)
    }
  })

  it('translateNow resolves fileMenu.copyContent at runtime for every locale', () => {
    for (const locale of locales) {
      setRuntimeI18nLocale(locale)
      const value = translateNow('fileMenu.copyContent')
      expect(value, `${locale} fileMenu.copyContent`).not.toBe('fileMenu.copyContent')
      expect(value.length).toBeGreaterThan(0)
    }
  })

  it('translateNow resolves fileMenu.contentCopied at runtime for every locale', () => {
    for (const locale of locales) {
      setRuntimeI18nLocale(locale)
      const value = translateNow('fileMenu.contentCopied')
      expect(value, `${locale} fileMenu.contentCopied`).not.toBe('fileMenu.contentCopied')
      expect(value.length).toBeGreaterThan(0)
    }
  })

  it('falls back to English when a locale is missing the key', () => {
    // Simulate a missing key by deleting it from ja, then restoring.
    const jaFm = TRANSLATIONS.ja.fileMenu as { copyContent?: string }
    const original = jaFm.copyContent
    try {
      jaFm.copyContent = undefined
      setRuntimeI18nLocale('ja')
      // Should fall back to the English string, not the raw key.
      expect(translateNow('fileMenu.copyContent')).toBe('Copy File Content')
    } finally {
      jaFm.copyContent = original
    }
  })
})

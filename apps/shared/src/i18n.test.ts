import { describe, expect, it } from 'vitest'

import { applyDocumentLocale, LOCALE_ENDONYMS, mergeTranslations, RTL_LOCALES, type TranslationOverride } from './i18n'

describe('mergeTranslations', () => {
  it('keeps untouched sibling keys on a nested partial override and replaces functions/arrays wholesale', () => {
    interface Catalog {
      menu: { close: string; count: (n: number) => string; open: string }
      steps: string[]
    }

    const base: Catalog = {
      menu: { open: 'Open', close: 'Close', count: n => `${n} items` },
      steps: ['one', 'two', 'three']
    }

    const overrides: TranslationOverride<Catalog> = {
      menu: { close: 'Schließen', count: n => `${n} Einträge` },
      steps: ['eins']
    }

    const merged = mergeTranslations<Catalog>(base, overrides)

    expect(merged.menu.open).toBe('Open')
    expect(merged.menu.close).toBe('Schließen')
    expect(merged.menu.count(2)).toBe('2 Einträge')
    expect(merged.steps).toEqual(['eins'])
    expect(base.menu.close).toBe('Close')
  })
})

describe('RTL_LOCALES', () => {
  it('only names locales that have an endonym', () => {
    for (const locale of RTL_LOCALES) {
      expect(Object.keys(LOCALE_ENDONYMS)).toContain(locale)
    }
  })
})

describe('applyDocumentLocale', () => {
  it('is a no-op without a document', () => {
    expect(typeof document).toBe('undefined')
    expect(() => applyDocumentLocale('ar')).not.toThrow()
  })
})

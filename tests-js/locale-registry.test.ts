import { describe, expect, it } from 'vitest'

import { LOCALE_ENDONYMS, RTL_LOCALES } from '../apps/shared/src/i18n'
import { LOCALE_METADATA, localeDirection, LOCALES, normalizeLocaleInput } from '../apps/shared/src/locale-registry'
import registry from '../locales/registry.json'

describe('shared locale boundary', () => {
  it('normalizes every registered identity and alias while rejecting inherited object keys', () => {
    for (const [input, expected] of Object.entries({
      ...Object.fromEntries(LOCALES.map(locale => [locale, locale])),
      ...registry.aliases,
      ...registry.compatibilityAliases
    })) {
      expect(normalizeLocaleInput(input)).toBe(expected)
      expect(normalizeLocaleInput(` ${input.toUpperCase().replaceAll('-', '_')} `)).toBe(expected)
    }

    for (const value of ['constructor', '__proto__', 'toString', 'unknown', {}, null]) {
      expect(normalizeLocaleInput(value)).toBeNull()
    }
  })
  it('derives all picker and document metadata from the registry', () => {
    expect(Object.keys(LOCALE_ENDONYMS)).toEqual(LOCALES)

    for (const locale of LOCALES) {
      expect(LOCALE_ENDONYMS[locale]).toBe(LOCALE_METADATA[locale].name)
      expect(RTL_LOCALES.has(locale)).toBe(localeDirection(locale) === 'rtl')
    }
  })
})

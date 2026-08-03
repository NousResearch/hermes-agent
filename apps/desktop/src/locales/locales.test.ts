import { describe, expect, it } from 'vitest'

import ar from './ar.json'
import de from './de.json'
import en from './en.json'
import es from './es.json'
import fr from './fr.json'
import hi from './hi.json'
import itLocale from './it.json'
import ja from './ja.json'
import ko from './ko.json'
import ptBR from './pt-BR.json'
import th from './th.json'
import vi from './vi.json'
import zhCN from './zh-CN.json'
import zhHant from './zh-Hant.json'

type LocaleCatalog = Record<string, unknown>

const catalogs: Record<string, LocaleCatalog> = {
  ar,
  de,
  en,
  es,
  fr,
  hi,
  it: itLocale,
  ja,
  ko,
  'pt-BR': ptBR,
  th,
  vi,
  'zh-CN': zhCN,
  'zh-Hant': zhHant
}

function flattenKeys(value: LocaleCatalog, prefix = ''): string[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key
    if (child && typeof child === 'object' && !Array.isArray(child)) {
      return flattenKeys(child as LocaleCatalog, path)
    }
    return path
  })
}

describe('desktop locale catalogs', () => {
  it('all non-English catalogs are subsets of the English catalog (partial fallback is expected)', () => {
    const englishKeys = flattenKeys(en).sort()

    for (const [locale, catalog] of Object.entries(catalogs)) {
      if (locale === 'en') continue

      const localeKeys = flattenKeys(catalog).sort()
      const extraKeys = localeKeys.filter(k => !englishKeys.includes(k))

      // Self-reference keys (language.xx) are expected — each locale may
      // declare its own name. Filter them out before checking for unexpected extras.
      const unexpectedExtra = extraKeys.filter(
        k => !k.startsWith('language.') && k !== 'artifacts'
      )

      expect(
        unexpectedExtra,
        `${locale}: keys not present in en.json — these would never be translated`
      ).toEqual([])
    }
  })

  it('all catalogs are valid JSON objects (parseable at import time)', () => {
    for (const [locale, catalog] of Object.entries(catalogs)) {
      expect(typeof catalog, `${locale}: catalog must be an object`).toBe('object')
      expect(catalog, `${locale}: catalog must not be null`).not.toBeNull()
    }
  })
})

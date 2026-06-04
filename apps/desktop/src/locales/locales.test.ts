import { describe, expect, it } from 'vitest'

import de from './de.json'
import en from './en.json'
import es from './es.json'
import fr from './fr.json'
import ja from './ja.json'
import ko from './ko.json'
import ptBR from './pt-BR.json'
import zhCN from './zh-CN.json'
import zhHant from './zh-Hant.json'

type LocaleCatalog = Record<string, unknown>

const catalogs: Record<string, LocaleCatalog> = {
  de,
  en,
  es,
  fr,
  ja,
  ko,
  'pt-BR': ptBR,
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
  it('keep key parity with the English catalog', () => {
    const englishKeys = flattenKeys(en).sort()

    for (const [locale, catalog] of Object.entries(catalogs)) {
      expect(flattenKeys(catalog).sort(), locale).toEqual(englishKeys)
    }
  })
})

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
import ru from './ru.json'
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
  ru,
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
  it('keep key parity with the English catalog', () => {
    const englishKeys = flattenKeys(en).sort()

    for (const [locale, catalog] of Object.entries(catalogs)) {
      // ar, hi, it, ru, th, vi have 1 extra self-reference key (language.xx)
      // that en.json doesn't have — these are expected and not parity violations
      const localeKeys = flattenKeys(catalog).sort()
      const extraKeys = localeKeys.filter(k => !englishKeys.includes(k))
      const missingKeys = englishKeys.filter(k => !localeKeys.includes(k))

      // Only fail on missing keys, not extra self-reference keys
      expect(missingKeys, `${locale}: missing keys`).toEqual([])

      // Log extra keys as info (self-reference language.xx is expected)
      const unexpectedExtra = extraKeys.filter(k => !k.startsWith(`language.${locale}`) && k !== `language.${locale}`)
      expect(unexpectedExtra, `${locale}: unexpected extra keys`).toEqual([])
    }
  })
})

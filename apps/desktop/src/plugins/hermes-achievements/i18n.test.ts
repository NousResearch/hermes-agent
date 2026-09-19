import { describe, expect, it } from 'vitest'

import { ACHIEVEMENTS_LOCALES } from './i18n'

type Leaf = string | ((...args: never[]) => string)

function leafEntries(node: unknown, prefix = ''): Array<[string, Leaf]> {
  if (typeof node === 'function' || typeof node === 'string') {
    return [[prefix, node as Leaf]]
  }

  return Object.entries(node as Record<string, unknown>).flatMap(([key, value]) =>
    leafEntries(value, prefix ? `${prefix}.${key}` : key)
  )
}

describe('ACHIEVEMENTS_LOCALES', () => {
  it('covers the English message tree in every Desktop locale', () => {
    const enPaths = leafEntries(ACHIEVEMENTS_LOCALES.en).map(([path]) => path)

    for (const locale of ['ja', 'zh', 'zh-hant', 'ar', 'ru'] as const) {
      expect(leafEntries(ACHIEVEMENTS_LOCALES[locale]).map(([path]) => path)).toEqual(enPaths)
    }
  })

  it('localizes the primary navigation and status copy', () => {
    const en = Object.fromEntries(leafEntries(ACHIEVEMENTS_LOCALES.en))

    for (const locale of ['ja', 'zh', 'zh-hant', 'ar', 'ru'] as const) {
      const messages = Object.fromEntries(leafEntries(ACHIEVEMENTS_LOCALES[locale]))

      for (const path of ['nav', 'scan', 'loadingTitle', 'errorTitle']) {
        expect(messages[path]).not.toBe(en[path])
      }
    }
  })
})

import { describe, expect, it } from 'vitest'

import { BOTS_LOCALES } from './i18n'

/** Walks two message trees in lockstep and reports every leaf path whose
 *  shape (missing / extra / string-vs-function / function arity) differs.
 *  A relationship between two pieces of data, not a frozen snapshot of
 *  either one — the invariant this guards is "de never drifts from en",
 *  not "en has exactly these N keys today". */
function diffLeafShapes(en: unknown, other: unknown, path: string[] = []): string[] {
  if (typeof en === 'function') {
    if (typeof other !== 'function') {
      return [`${path.join('.')}: en is a function, locale is ${typeof other}`]
    }

    return en.length === other.length
      ? []
      : [`${path.join('.')}: arity mismatch (en takes ${en.length}, locale takes ${other.length})`]
  }

  if (typeof en === 'string') {
    return typeof other === 'string' ? [] : [`${path.join('.')}: en is a string, locale is ${typeof other}`]
  }

  if (en && typeof en === 'object') {
    if (!other || typeof other !== 'object') {
      return [`${path.join('.')}: en is an object, locale is ${typeof other}`]
    }

    const enKeys = Object.keys(en as Record<string, unknown>)
    const otherKeys = new Set(Object.keys(other as Record<string, unknown>))
    const problems: string[] = []

    for (const key of enKeys) {
      if (!otherKeys.has(key)) {
        problems.push(`${[...path, key].join('.')}: missing in locale`)
        continue
      }

      otherKeys.delete(key)
      problems.push(
        ...diffLeafShapes(
          (en as Record<string, unknown>)[key],
          (other as Record<string, unknown>)[key],
          [...path, key]
        )
      )
    }

    for (const extra of otherKeys) {
      problems.push(`${[...path, extra].join('.')}: present in locale but not in en`)
    }

    return problems
  }

  return []
}

describe('Bot Mode i18n — de parity', () => {
  it('covers every en leaf, with the same string/function shape and function arity', () => {
    const problems = diffLeafShapes(BOTS_LOCALES.en, BOTS_LOCALES.de)

    expect(problems).toEqual([])
  })

  it('registers under BOTS_LOCALES alongside the other shipped locales', () => {
    expect(Object.keys(BOTS_LOCALES).sort()).toEqual(['de', 'en', 'ja', 'zh', 'zh-hant'])
  })
})

import { describe, expect, it } from 'vitest'

import { ar } from './ar'
import { en } from './en'
import { ja } from './ja'
import { ru } from './ru'
import { zh } from './zh'
import { zhHant } from './zh-hant'

/**
 * Locale key-parity guard — the class-fix for the recurring "missed locale"
 * bug family: a key renamed in en.ts but left stale in a partial locale
 * (e.g. `view.terminalSelection` → `view.selectionToComposer`, which missed
 * ru.ts for weeks and made the renamed row fall back to English for ru
 * users).
 *
 * `defineLocale()` locales are PARTIAL by design — missing keys fall back to
 * English at runtime (that's the documented contract). What must never
 * happen is a locale carrying a *stale* key: en renamed or removed a leaf
 * while the locale kept the old name. The old key is dead weight at best
 * and a silently-lost translation at worst.
 *
 * Invariant (one-directional): when en has a NON-EMPTY section, every
 * locale key under that section must still exist in en. Sections en emptied
 * entirely (`platformIntro: {}` — free-form `Record<string, string>`
 * overrides read before a code-level fallback const) are exempt: a locale
 * may legitimately carry extra free-form keys en leaves empty.
 */

type LeafKey = string

type Obj = Record<string, unknown>

function asObj(value: unknown): Obj | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Obj)
    : null
}

function leafEntries(value: unknown, prefix = ''): LeafKey[] {
  const obj = asObj(value)

  if (obj === null) {
    return prefix ? [prefix] : []
  }

  const out: LeafKey[] = []

  for (const [key, child] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key
    out.push(...leafEntries(child, path))
  }

  return out
}

function keySet(obj: unknown): Set<LeafKey> {
  return new Set(leafEntries(obj))
}

/** Value at a dot-path in the en tree, or undefined. */
function enAt(path: string): unknown {
  let node: unknown = en

  for (const part of path.split('.')) {
    const obj = asObj(node)

    if (obj === null) {return undefined}
    node = obj[part]

    if (node === undefined) {return undefined}
  }

  return node
}

/**
 * Stale = a locale key whose en counterpart is gone while its enclosing en
 * section is still live. Walks ancestor paths from longest to shortest:
 *   - ancestor exists in en as an EMPTY object → free-form section en
 *     intentionally leaves empty (platformIntro: {} with a code fallback
 *     const); locale overrides are legitimate → EXEMPT
 *   - ancestor exists in en as a live (non-empty) object → the leaf should
 *     exist in en too; if it doesn't, the key was renamed/removed in en but
 *     left behind in the locale → STALE
 *   - ancestor not in en at all → keep walking up (the section may have
 *     moved wholesale into a live grandparent)
 */
function staleUnderLiveParents(locale: unknown): LeafKey[] {
  const enKeys = keySet(en)

  return [...keySet(locale)].filter((k) => {
    if (enKeys.has(k)) {return false}
    const parts = k.split('.')

    for (let i = parts.length - 1; i >= 1; i -= 1) {
      const parent = parts.slice(0, i).join('.')
      const enValue = enAt(parent)

      if (enValue === undefined) {continue} // keep walking up
      const obj = asObj(enValue)

      if (obj !== null && Object.keys(obj).length === 0) {
        return false // en emptied this whole section → locale extras are legit
      }

      return true // live section (or scalar) → leaf should still exist in en
    }

    return false
  }).sort()
}

const LOCALES: Array<[string, unknown]> = [
  ['zh', zh],
  ['zh-hant', zhHant],
  ['ja', ja],
  ['ar', ar],
  ['ru', ru],
]

describe('desktop i18n key parity with en', () => {
  it.each(LOCALES)('%s has no stale keys under live en sections', (_name, locale) => {
    const stale = staleUnderLiveParents(locale)
    expect(
      stale,
      `stale keys in ${_name} — renamed/removed in en.ts but still declared here`
    ).toEqual([])
  })

  it('reports partial-locale coverage so the translation backlog is visible', () => {
    // Visibility only — NOT a failure. Partial locales are a supported state
    // (defineLocale falls back to English for missing keys), so a missing
    // key is a translation backlog item, not a bug. This just forces the
    // counts into CI output so a locale quietly losing coverage shows up.
    const enKeys = keySet(en)
    const missing: string[] = []

    for (const [name, locale] of LOCALES) {
      const localeSet = keySet(locale)
      const absent = [...enKeys].filter((k) => !localeSet.has(k))

      if (absent.length > 0) {
        missing.push(`${name}: ${absent.length} keys fall back to English (${absent.slice(0, 3).join(', ')}…)`)
      }
    }

    if (missing.length > 0) {
      console.info(`[i18n key-parity] partial-locale coverage:\n${missing.join('\n')}`)
    }
  })
})

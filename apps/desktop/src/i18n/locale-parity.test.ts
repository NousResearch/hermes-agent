import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import { en } from './en'

/**
 * Locale parity guard.
 *
 * `defineLocale()` deep-merges a locale's overrides onto `en`, and the
 * override type makes every key optional. That is deliberate - a partially
 * translated locale should fall back to English rather than break.
 *
 * The failure mode it hides is a locale key that English NO LONGER HAS.
 * TypeScript cannot catch it: blocks like `composer.commandDescs` are typed
 * `Record<string, string>`, so any key satisfies the type. Verified against
 * this repo - re-adding a removed key to a locale leaves `tsc --noEmit`
 * exiting 0.
 *
 * That is not cosmetic. #66646 removed terminal-only commands (`/clear`,
 * `/details`, `/copy`, `/quit`) from the desktop quick-help drawer because
 * the desktop refuses them, but a locale that still defined those keys kept
 * advertising commands that do not work. The stale entries survive the merge
 * and win over English, so the very bug the change fixed stayed live in that
 * language while every other locale was corrected.
 *
 * This guard fails on exactly that: a key defined by a locale that does not
 * exist in English. Missing keys are NOT flagged - falling back is the
 * design.
 */

/**
 * Blocks whose keys are supplied at runtime rather than fixed by English.
 *
 * `sidebar.nav` is a contribution registry (`SIDEBAR_NAV_AREA` in
 * `app/routes.ts`) - plugins register nav items, so a locale carrying a label
 * for a contributed item is legitimate rather than stale. Keep this list
 * short: every entry is a place the guard cannot protect.
 */
const REGISTRY_PATHS = new Set(['sidebar.nav'])

type Node = Record<string, unknown>

const isRecord = (value: unknown): value is Node =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/** Key paths present in `node` but absent from `reference`, depth-first. */
function extraKeyPaths(node: unknown, reference: unknown, trail: string[] = []): string[] {
  if (!isRecord(node) || !isRecord(reference)) {
    return []
  }

  const found: string[] = []

  // A block English leaves deliberately empty (e.g. `platformIntro: {}`) is an
  // extension point locales are meant to populate, not a set they must match.
  if (Object.keys(reference).length === 0) {
    return []
  }

  if (REGISTRY_PATHS.has(trail.join('.'))) {
    return []
  }

  for (const [key, value] of Object.entries(node)) {
    const path = [...trail, key]

    if (!(key in reference)) {
      found.push(path.join('.'))
      continue
    }

    found.push(...extraKeyPaths(value, reference[key], path))
  }

  return found
}

describe('locale parity', () => {
  const locales = Object.entries(TRANSLATIONS).filter(([id]) => id !== 'en')

  it('covers every registered locale', () => {
    // A locale added to the catalog without being added here would silently
    // escape the guard, which is the same class of miss this test exists for.
    expect(locales.length).toBeGreaterThan(0)
    expect(Object.keys(TRANSLATIONS)).toContain('en')
  })

  it.each(locales)('%s defines no key English has dropped', (id, translations) => {
    const stale = extraKeyPaths(translations, en)

    expect(
      stale,
      stale.length
        ? `${id} defines ${stale.length} key(s) English no longer has, so they win over the ` +
            `English value and can surface removed or renamed UI:\n  ${stale.join('\n  ')}`
        : ''
    ).toEqual([])
  })
})

describe('extraKeyPaths', () => {
  it('reports a key the reference does not have', () => {
    expect(extraKeyPaths({ a: 1, gone: 2 }, { a: 1 })).toEqual(['gone'])
  })

  it('reports nested keys with a dotted trail', () => {
    expect(extraKeyPaths({ outer: { kept: 1, stale: 2 } }, { outer: { kept: 1 } })).toEqual([
      'outer.stale'
    ])
  })

  it('does not report missing keys - falling back to English is by design', () => {
    expect(extraKeyPaths({ a: 1 }, { a: 1, translatedLater: 2 })).toEqual([])
  })

  it('ignores a block English leaves deliberately empty', () => {
    // `en.messaging.platformIntro` is `{}` on purpose - locales fill it.
    expect(extraKeyPaths({ platformIntro: { slack: 'x' } }, { platformIntro: {} })).toEqual([])
  })

  it('ignores a documented runtime registry block', () => {
    // Plugins contribute sidebar nav items, so extra locale labels there are
    // not evidence of a removed key.
    expect(
      extraKeyPaths({ nav: { plugin: 'x' } }, { nav: { core: 'y' } }, ['sidebar'])
    ).toEqual([])
  })

  it('ignores functions and arrays rather than walking into them', () => {
    const fn = (n: number) => `${n}`
    expect(extraKeyPaths({ fn, list: [1, 2] }, { fn, list: [1] })).toEqual([])
  })
})

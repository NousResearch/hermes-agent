import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import type { TranslationOverrides } from './define-locale'
import { en } from './en'
import { LOCALE_OPTIONS } from './languages'

/** Every message path in a catalog, mapped to the kind of value it holds (a leaf
 *  kind, or `array` — locally-sized lists like placeholder sets are opaque). */
function leafKinds(value: unknown, prefix = '', out: Record<string, string> = {}): Record<string, string> {
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${key}` : key

    if (Array.isArray(child)) {
      out[path] = 'array'
    } else if (typeof child === 'object' && child !== null) {
      leafKinds(child, path, out)
    } else {
      out[path] = typeof child
    }
  }

  return out
}

describe('desktop i18n catalog', () => {
  it('registers a table for every locale in the picker', () => {
    expect(Object.keys(TRANSLATIONS).sort()).toEqual(LOCALE_OPTIONS.map(locale => locale.id).sort())
  })

  // The English catalog is the message schema (`types.ts` used to spell it out by
  // hand). A locale is a partial override merged over `en`, so every path English
  // defines must resolve to the same kind of value in every locale — a locale may
  // still translate ids English ships no copy for. This is the completeness guard
  // the handwritten schema used to be.
  it.each(Object.keys(TRANSLATIONS))('%s resolves every English message', locale => {
    const table = leafKinds(TRANSLATIONS[locale as keyof typeof TRANSLATIONS])
    const broken = Object.entries(leafKinds(en)).filter(([path, kind]) => table[path] !== kind)

    expect(broken).toEqual([])
  })

  it('keeps a locale override key English does not know a compile error', () => {
    // @ts-expect-error — a typo'd key would otherwise vanish into the fallback
    const overrides: TranslationOverrides = { notAMessageKey: 'nope' }

    expect(Object.keys(overrides)).toEqual(['notAMessageKey'])
  })
})

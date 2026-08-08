import { describe, expect, it } from 'vitest'

// Source of truth for builtin personality IDs is the same array the
// settings picker renders from — import it directly so a future rename
// or addition breaks the test instead of silently shipping stale IDs.
import { BUILTIN_PERSONALITIES } from '@/app/settings/constants'

import { TRANSLATIONS } from './catalog'
import type { Locale } from './types'

// All locales the type system promises. The catalog completeness test
// iterates over this so adding a new locale (e.g. `fr`, `ko`) automatically
// gets the personalities-map assertion for free.
const ALL_LOCALES: Locale[] = ['en', 'zh', 'zh-hant', 'ja', 'ar']

describe('desktop i18n — builtin personality labels', () => {
  // Every builtin ID the renderer can show in the `display.personality`
  // dropdown must have a Simplified Chinese label, otherwise zh users still
  // see the raw English ID in the dropdown.
  it('zh provides a localized label for every builtin personality', () => {
    const labels = TRANSLATIONS.zh.settings.personalities

    for (const id of BUILTIN_PERSONALITIES) {
      const label = labels[id]
      expect(label, `missing zh label for builtin personality "${id}"`).toBeTypeOf('string')
      expect((label ?? '').trim().length, `zh label for "${id}" is empty`).toBeGreaterThan(0)
      // Defence in depth: a label that is identical to the raw ID means the
      // translator forgot this entry — surface it loudly instead of silently
      // shipping English in the Simplified Chinese dropdown.
      expect(label, `zh label for "${id}" is just the raw ID`).not.toBe(id)
    }
  })

  it('zh-hant mirrors zh coverage so 繁體 dropdown is not partially English', () => {
    const labels = TRANSLATIONS['zh-hant'].settings.personalities
    const zhLabels = TRANSLATIONS.zh.settings.personalities

    for (const id of BUILTIN_PERSONALITIES) {
      const label = labels[id]
      expect(label, `missing zh-hant label for builtin personality "${id}"`).toBeTypeOf('string')
      expect((label ?? '').trim().length, `zh-hant label for "${id}" is empty`).toBeGreaterThan(0)
      expect(label, `zh-hant label for "${id}" is just the raw ID`).not.toBe(id)
    }

    // The Simplified and Traditional sets should cover exactly the same keys.
    expect(Object.keys(labels).sort()).toEqual(Object.keys(zhLabels).sort())
  })

  it('every locale exposes the personalities map (may be empty for fallback)', () => {
    // Catalog completeness: the type system guarantees this, but assert at
    // runtime so a future locale addition can't ship without the field.
    for (const locale of ALL_LOCALES) {
      const labels = TRANSLATIONS[locale].settings.personalities
      expect(labels, `${locale} missing settings.personalities`).toBeDefined()
      expect(typeof labels).toBe('object')
    }
  })

  it('en falls back to the raw builtin ID when no override exists', () => {
    // English UI users see the raw personality ID (helpful, concise, …) in
    // the dropdown today. Documenting that contract here keeps the contract
    // intentional: any future `en.personalities` map is an additive change.
    const labels = TRANSLATIONS.en.settings.personalities

    for (const id of BUILTIN_PERSONALITIES) {
      const label = labels[id] ?? id
      expect(label).toBe(id)
    }
  })
})
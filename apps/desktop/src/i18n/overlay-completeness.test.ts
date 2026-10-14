import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { flattenMessageKeys } from '@hermes/shared/i18n'
import { expect, it } from 'vitest'

import { arOverrides } from './ar'
import { deOverrides } from './de'
import type { TranslationOverrides } from './define-locale'
import { en } from './en'
import { esOverrides } from './es'
import { frOverrides } from './fr'
import { jaOverrides } from './ja'
import { koOverrides } from './ko'
import knownGaps from './overlay-gaps.json'
import { ruOverrides } from './ru'
import type { BundledLocale } from './types'
import { zhOverrides } from './zh'
import { zhHantOverrides } from './zh-hant'

// Overlays are partial, so a missing key renders English without failing typecheck
// or catalog-completeness.test.ts (which reads the merged catalog). This reads each
// locale's own overlay. overlay-gaps.json maps each English key to the locales that
// lacked it when this check landed. It only ratchets: translating a listed key passes
// untouched, while a key missing from a locale and not listed fails. Prune it with
// `OVERLAY_GAPS_UPDATE=1 npx vitest run src/i18n/overlay-completeness.test.ts`.
const OVERLAYS = {
  ar: arOverrides,
  de: deOverrides,
  es: esOverrides,
  fr: frOverrides,
  ja: jaOverrides,
  ko: koOverrides,
  ru: ruOverrides,
  zh: zhOverrides,
  'zh-hant': zhHantOverrides
} satisfies Record<Exclude<BundledLocale, 'en'>, TranslationOverrides>

// `intro` is display-only; intro.test.tsx covers its translated rotation.
const translatable = (tree: TranslationOverrides) => flattenMessageKeys(tree).filter(key => !key.startsWith('intro.'))

function currentGaps(): Record<string, string[]> {
  const present = Object.entries(OVERLAYS).map(([locale, overlay]) => [locale, new Set(translatable(overlay))] as const)

  return Object.fromEntries(
    translatable(en)
      .map(key => [key, present.filter(([, keys]) => !keys.has(key)).map(([locale]) => locale)] as const)
      .filter(([, locales]) => locales.length > 0)
  )
}

it('adds no untranslated key to any locale overlay beyond the known gaps', () => {
  const gaps = currentGaps()

  if (process.env.OVERLAY_GAPS_UPDATE) {
    const lines = Object.entries(gaps).map(([key, locales]) => `  ${JSON.stringify(key)}: ${JSON.stringify(locales)}`)
    writeFileSync(resolve(__dirname, 'overlay-gaps.json'), `{\n${lines.join(',\n')}\n}\n`)
  }

  const known: Record<string, string[] | undefined> = knownGaps

  const newGaps = Object.entries(gaps).flatMap(([key, locales]) =>
    locales.filter(locale => !known[key]?.includes(locale)).map(locale => `${locale}: ${key}`)
  )

  expect(newGaps).toEqual([])
})

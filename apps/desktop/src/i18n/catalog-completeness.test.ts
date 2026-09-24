import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import type { Locale } from './types'

// Locales that shipped fully translated. They are `defineLocale` overlays like
// ja/ru, so an English key added later falls back to English instead of
// failing typecheck; these checks keep the translated copy structurally sound.
const COMPLETE_LOCALES = ['fr', 'de', 'es'] as const satisfies readonly Locale[]

type Leaf = { path: string; value: unknown }

function leaves(value: unknown, path = ''): Leaf[] {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return Object.entries(value).flatMap(([key, child]) => leaves(child, path ? `${path}.${key}` : key))
  }

  return [{ path, value }]
}

// Arguments that are identifiers rather than display text; translations
// branch on them (`capability === 'search' ? … : …`) instead of echoing them.
const IDENTIFIER_ARGS: Record<string, number[]> = {
  'settings.toolsets.webCapabilitySelectedMessage': [1]
}

const kindOf = (value: unknown) => (Array.isArray(value) ? 'array' : typeof value)

// `intro` is display-only: English lives in intro-copy.jsonl, so its catalog
// entry is an empty shell. intro.test.tsx covers the translated rotation.
const catalogLeaves = (locale: Locale) =>
  new Map(
    leaves(TRANSLATIONS[locale])
      .filter(leaf => !leaf.path.startsWith('intro.'))
      .map(leaf => [leaf.path, leaf.value])
  )

const english = catalogLeaves('en')

describe.each(COMPLETE_LOCALES)('%s desktop catalog', locale => {
  const catalog = catalogLeaves(locale)

  it('covers exactly the English key set with matching value kinds', () => {
    expect([...catalog.keys()].sort()).toEqual([...english.keys()].sort())

    for (const [path, value] of english) {
      expect({ path, kind: kindOf(catalog.get(path)) }).toEqual({ path, kind: kindOf(value) })
    }
  })

  it('keeps every interpolated argument that English renders', () => {
    for (const [path, value] of english) {
      if (typeof value !== 'function') {
        continue
      }

      const translated = catalog.get(path) as (...args: unknown[]) => unknown
      const probes = Array.from({ length: value.length }, (_, index) => `⟦${index}⟧`)
      let englishOut: string

      try {
        englishOut = JSON.stringify(value(...probes))
      } catch {
        continue // needs structured arguments; the type checker covers the signature
      }

      expect(translated.length, path).toBe(value.length)
      const translatedOut = JSON.stringify(translated(...probes))

      const identifiers = new Set(IDENTIFIER_ARGS[path]?.map(index => probes[index]))

      for (const probe of probes.filter(probe => englishOut.includes(probe) && !identifiers.has(probe))) {
        expect(translatedOut, `${path} drops ${probe}`).toContain(probe)
      }
    }
  })

  it('keeps list-shaped copy the same length as English', () => {
    for (const [path, value] of english) {
      if (Array.isArray(value)) {
        expect((catalog.get(path) as unknown[]).length, path).toBe(value.length)
      }
    }
  })
})

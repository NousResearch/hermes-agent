/**
 * Parity guard for the renderer-side translation files.
 *
 * Mirrors the Python-side convention (locales/*.yaml + tests/agent/test_i18n.py):
 * every leaf key in the source-of-truth file (en) must exist in every other
 * language, every value must be a string, and the {{placeholder}} tokens
 * interpolated by i18next must line up. This keeps the tree green even when
 * contributors add or rename keys.
 */

import { describe, expect, it } from 'vitest'

import en from './en/translation.json'
import zhCN from './zh-CN/translation.json'

const PLACEHOLDER_RE = /\{\{(\w+)\}\}/g

type Flat = Record<string, string>

function isFlat(value: unknown): value is Flat {
  if (!value || typeof value !== 'object') {
    return false
  }

  return Object.values(value).every(v => typeof v === 'string')
}

function placeholders(value: string): string[] {
  return [...value.matchAll(PLACEHOLDER_RE)].map(m => m[1]).sort()
}

describe('locales parity', () => {
  // The source file uses flat keys (common.close, settings.appearance.mode.dark)
  // rather than nested JSON. That makes them grep-friendly and matches the
  // i18next convention of treating keys as opaque strings.
  it('en is a flat string→string map', () => {
    expect(isFlat(en)).toBe(true)
  })

  it('zh-CN is a flat string→string map', () => {
    expect(isFlat(zhCN)).toBe(true)
  })

  const enFlat = en as unknown as Flat
  const zhFlat = zhCN as unknown as Flat
  const sourceKeys = Object.keys(enFlat)

  it('source (en) has at least one key', () => {
    expect(sourceKeys.length).toBeGreaterThan(0)
  })

  describe('zh-CN/translation.json', () => {
    it('has the same keys as en', () => {
      const enSet = new Set(sourceKeys)
      const zhSet = new Set(Object.keys(zhFlat))

      const missing = [...enSet].filter(k => !zhSet.has(k))
      const extra = [...zhSet].filter(k => !enSet.has(k))

      expect(missing, `keys missing in zh-CN`).toEqual([])
      expect(extra, `extra keys in zh-CN not present in en`).toEqual([])
    })

    it('placeholder tokens match en for every key', () => {
      for (const key of sourceKeys) {
        const expected = placeholders(enFlat[key])
        const actual = placeholders(zhFlat[key])

        expect(actual, `${key}: zh-CN placeholders must match en`).toEqual(expected)
      }
    })
  })
})

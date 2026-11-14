import { describe, expect, it } from 'vitest'

import stockRaw from '@/components/chat/intro-copy.jsonl?raw'

import { BUNDLED_LOCALES, TRANSLATIONS } from './catalog'

// The catalog-completeness checks skip `intro` (English editorial copy lives in
// the shipped JSONL, not in en.ts), so a translated rotation could drift: a
// renamed preset id, a short rotation, a custom() that lost its label arg.
// This pins every bundled locale that ships personalities to the JSONL's shape.
const stockIds = stockRaw
  .trim()
  .split('\n')
  .map(line => JSON.parse(line).personality as string)

const uniqueStockIds = [...new Set(stockIds)]

describe('bundled intro personalities', () => {
  it('covers exactly the stock rotation ids, five lines each, with a one-arg custom()', () => {
    for (const id of uniqueStockIds) {
      expect(stockIds.filter(s => s === id), `the JSONL rotation carries five lines per personality (${id})`).toHaveLength(5)
    }

    for (const locale of BUNDLED_LOCALES) {
      const { intro } = TRANSLATIONS[locale]

      if (Object.keys(intro.stock).length === 0) {
        continue // en ships an empty shell; the rotation renders from the JSONL
      }

      expect(Object.keys(intro.stock).sort(), `${locale}: stock ids`).toEqual([...uniqueStockIds].sort())

      for (const [id, lines] of Object.entries(intro.stock)) {
        expect(lines, `${locale}.${id}: rotation length`).toHaveLength(5)
      }

      expect(intro.custom, `${locale}: custom()`).toBeTypeOf('function')
      expect(intro.custom.length, `${locale}: custom() takes the label`).toBe(1)
    }
  })
})

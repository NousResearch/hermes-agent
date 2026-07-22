/**
 * Fact-Forcing Gate (pre-create):
 * 1. Importers: vitest only — `npx vitest run src/lib/statusbar-cache-cost.test.ts`
 *    under apps/desktop. Production code imports `./statusbar`, not this file.
 * 2. External API: none (test module). Exercises pure helpers
 *    `formatCacheFresh` / `formatStatusCost` from `./statusbar`.
 * 3. Data schema: none — synthetic UsageStats fixtures only
 *    (cache_read, last_prompt, cost_usd numbers).
 * 4. User instruction (verbatim): "底栏 cache与费用，需要保留，出稳定方案"
 */
import { describe, expect, it } from 'vitest'

import { formatCacheFresh, formatStatusCost } from './statusbar'

import type { UsageStats } from '@/types/hermes'

const baseUsage = (patch: Partial<UsageStats> = {}): UsageStats => ({
  calls: 1,
  input: 0,
  output: 0,
  total: 0,
  ...patch
})

describe('statusbar cache + cost helpers', () => {
  describe('formatCacheFresh', () => {
    it('returns empty when there is no cache_read (no fabricated hits)', () => {
      expect(formatCacheFresh(baseUsage({ context_used: 10_000, last_prompt: 10_000 }))).toBe('')
    })

    it('uses last_prompt as hit denominator', () => {
      const full = formatCacheFresh(
        baseUsage({
          cache_read: 71_168,
          last_prompt: 71_614,
          context_used: 71_614
        })
      )
      expect(full).toContain('cache R=')
      expect(full).toContain('99%')
      expect(full).toContain('fresh=')

      const compact = formatCacheFresh(
        baseUsage({
          cache_read: 71_168,
          last_prompt: 71_614,
          context_used: 71_614
        }),
        true
      )
      expect(compact).toMatch(/^R=/)
      expect(compact).toContain('99%')
      expect(compact).not.toContain('fresh=')
    })
  })

  describe('formatStatusCost', () => {
    it('hides non-positive costs', () => {
      expect(formatStatusCost(undefined)).toBe('')
      expect(formatStatusCost(0)).toBe('')
      expect(formatStatusCost(-1)).toBe('')
    })

    it('formats sub-cent and larger sessions', () => {
      expect(formatStatusCost(0.004)).toBe('<$0.01')
      expect(formatStatusCost(1.2)).toBe('$1.20')
      expect(formatStatusCost(127.093)).toBe('$127.093')
    })

    it('marks estimated costs with a tilde', () => {
      expect(formatStatusCost(1.2, 'estimated')).toBe('~$1.20')
      expect(formatStatusCost(0.004, 'estimated')).toBe('~<$0.01')
    })
  })
})

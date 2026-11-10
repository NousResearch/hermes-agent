import { describe, expect, it } from 'vitest'

import { ar } from './ar'
import { arBatch1 } from './ar_batch_1'
import { en } from './en'

describe('Arabic batch 1 overlay', () => {
  it('renders batch values and keeps existing Arabic siblings', () => {
    expect(ar.common.tryHint('Hermes')).toBe('جرّب «Hermes»')
    expect(ar.sidebar.toolCallCount(2)).toBe('استدعاءا أداة')
    expect(ar.sidebar.messageCount(11)).toBe('11 رسالة')
    expect(ar.shell.statusbar.contextUsagePanel.tokenSummary('1k', '8k')).toBe('1k / 8k Tokens')
    expect(ar.assistant.catalogInstall.serverNotConnected('github', '')).toContain('github')
    // A sibling owned by the section overlays must survive the deep merge.
    expect(ar.composer.redirect).toBe('إعادة توجيه التشغيل الحالي')
    expect(ar.composer.redirect).not.toBe(en.composer.redirect)
  })

  it('only overrides keys the English catalog defines', () => {
    const walk = (node: object, base: object, path: string[]): string[] =>
      Object.entries(node).flatMap(([key, value]) => {
        const next = (base as Record<string, unknown>)[key]

        if (next === undefined) {
          return [[...path, key].join('.')]
        }

        return typeof value === 'object' && value !== null ? walk(value, next as object, [...path, key]) : []
      })

    expect(walk(arBatch1, en, [])).toEqual([])
  })
})

import { describe, expect, it } from 'vitest'

import { ar } from './ar'
import { arBatch2 } from './ar_batch_2'
import { en } from './en'

describe('Arabic batch 2 overlay', () => {
  it('localizes same-build engine repairs without dropping the build tag', () => {
    const tag = 'fixture-build-tag'
    const detail = ar.settings.localModels.updateSameBuildDetail(tag)

    expect(detail).not.toBe(en.settings.localModels.updateSameBuildDetail(tag))
    expect(detail).toContain(tag)
    expect(detail).toContain('Hermes')
    expect(detail).toContain('llama.cpp')
  })

  it('renders batch values and keeps existing Arabic siblings', () => {
    expect(ar.agents.ageDays(2)).toBe('منذ 2 يوم')
    expect(ar.zones.newTab).toBe('تبويب جديد')
    // A sibling owned by an earlier overlay must survive the deep merge.
    expect(ar.common.tryHint('Hermes')).toBe('جرّب «Hermes»')
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

    expect(walk(arBatch2, en, [])).toEqual([])
  })
})

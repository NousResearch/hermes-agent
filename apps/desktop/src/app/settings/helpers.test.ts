import { describe, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { getNested, setNested } from './helpers'

describe('settings helpers', () => {
  it('reads and writes nested config paths', () => {
    const config: HermesConfigRecord = { display: { theme: 'mono' } }
    const next = setNested(config, 'display.theme', 'slate')

    expect(getNested(next, 'display.theme')).toBe('slate')
    expect(getNested(config, 'display.theme')).toBe('mono')
  })

  it('rejects prototype-polluting config paths', () => {
    const config: HermesConfigRecord = {}

    expect(() => setNested(config, '__proto__.polluted', true)).toThrow('Unsafe config path')
    expect(() => setNested(config, 'constructor.prototype.polluted', true)).toThrow('Unsafe config path')
    expect(({} as Record<string, unknown>).polluted).toBeUndefined()
  })
})
